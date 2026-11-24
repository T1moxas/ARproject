"""Direct dual-camera stereo. Run --help; calibrate with stereo_calibrate.py."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from rendering import Renderer
from calibration_tools import summarize_distance
from depth_appearance import DepthAppearance, STYLES
from surface_updates import SurfaceUpdates
from work_area import WorkArea, calibrate_area, projector_bounds
from surface_geometry import FloorPlane, HeightRenderer, ProjectorModel, project_points


CAPTURE_SIZE = (1280, 720)
SIZES = ((640, 360), (1280, 720))
PROJECTOR = "Stereo projector"
PREVIEW = "Stereo cameras: 1=360p 2=720p B=BM G=SGBM Esc=exit"
ARUCO_INTERVAL = 1 / 15


def dpi_awareness():
    if sys.platform == "win32":
        import ctypes
        try:
            setter = ctypes.windll.user32.SetProcessDpiAwarenessContext
            setter.argtypes = [ctypes.c_void_p]
            setter(ctypes.c_void_p(-4))
        except AttributeError:
            ctypes.windll.user32.SetProcessDPIAware()


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float  # software receipt time, not sensor exposure time
    image: np.ndarray


class Camera:
    """One capture thread per camera; bounded recent frames, no growing queue."""

    def __init__(self, index, focus=None):
        self.index = index
        self.focus = focus
        self.frames = deque(maxlen=8)
        self.timestamps = deque(maxlen=90)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = ""
        self.info = {}
        self.thread = threading.Thread(target=self._read, name=f"camera-{index}", daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Camera {self.index}: cannot open")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            # C920 DirectShow renegotiates format on size changes: set MJPG LAST.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            autofocus_disabled = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if self.focus is not None:
                cap.set(cv2.CAP_PROP_FOCUS, self.focus)
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.info = {"index": self.index,
                         "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                         "fourcc": "".join(chr((code >> (8 * i)) & 255) for i in range(4)),
                         "reported_fps": cap.get(cv2.CAP_PROP_FPS),
                         "focus": cap.get(cv2.CAP_PROP_FOCUS),
                         "autofocus_disable_requested": bool(autofocus_disabled)}
            return cap
        except BaseException:
            cap.release()
            raise

    def _read(self):
        sequence = 0
        while not self.stop.is_set():
            cap = None
            try:
                cap = self._open()
                failures = 0
                while not self.stop.is_set():
                    ok, image = cap.read()
                    stamp = time.perf_counter()
                    if not ok:
                        failures += 1
                        if failures >= 10:
                            raise OSError(f"Camera {self.index}: no frames; reconnecting")
                        self.stop.wait(0.02)
                        continue
                    if image.shape[:2] != (CAPTURE_SIZE[1], CAPTURE_SIZE[0]):
                        raise OSError(f"Camera {self.index}: expected 1280x720, got {image.shape}")
                    failures = 0
                    self.error = ""
                    sequence += 1
                    with self.lock:
                        self.frames.append(Frame(sequence, stamp, image))
                        self.timestamps.append(stamp)
            except (OSError, cv2.error) as exc:
                self.error = str(exc)
                with self.lock:
                    self.frames.clear()
                    self.timestamps.clear()
                self.stop.wait(1)
            finally:
                if cap is not None:
                    cap.release()

    def snapshot(self):
        with self.lock:
            return list(self.frames)

    def fps(self):
        with self.lock:
            times = list(self.timestamps)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


def choose_pair(left_frames, right_frames, previous=(0, 0), max_skew=0.05, now=None):
    """Prefer a fresh pair; receipt timestamps approximate synchronization."""
    now = time.perf_counter() if now is None else now
    choices = [(left, right) for left in left_frames for right in right_frames
               if left.sequence > previous[0] and right.sequence > previous[1]
               and 0 <= now - min(left.timestamp, right.timestamp) < 0.5
               and abs(left.timestamp - right.timestamp) <= max_skew]
    return max(choices, key=lambda pair: (min(pair[0].timestamp, pair[1].timestamp),
                                          -abs(pair[0].timestamp - pair[1].timestamp)), default=None)


class StereoEngine:
    def __init__(self, size, method="bm", calibration=None, disparities=None, low=0.3, high=4.0,
                 lr_check=True, lr_threshold=1.5, pad_edges=True):
        self.size, self.method = size, method
        self.calibrated = calibration is not None
        self.renderer = Renderer(low, high)
        self.low, self.high = low, high
        self.lr_check = lr_check
        self.lr_threshold = lr_threshold * size[0] / 640
        self.maps = None
        self.rectified_masks = None
        self.sign = 1
        self.fx_baseline = None
        self.rectified_intrinsic = None
        self.right_to_left_offset = np.zeros(3, dtype=np.float64)
        if calibration is not None:
            with np.load(calibration, allow_pickle=False) as saved:
                if tuple(saved["capture_size"].tolist()) != CAPTURE_SIZE:
                    raise ValueError("Calibration capture size must be 1280x720")
                k1, k2 = saved["K1"].copy(), saved["K2"].copy()
                for k in (k1, k2):
                    k[0] *= size[0] / CAPTURE_SIZE[0]
                    k[1] *= size[1] / CAPTURE_SIZE[1]
                r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                    k1, saved["D1"], k2, saved["D2"], size, saved["R"], saved["T"].reshape(3, 1),
                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                if abs(p2[1, 3]) > abs(p2[0, 3]) or abs(p2[0, 3]) < 1e-9:
                    raise ValueError("Use a horizontal stereo rig (cameras side by side)")
                self.maps = (cv2.initUndistortRectifyMap(k1, saved["D1"], r1, p1, size, cv2.CV_16SC2),
                             cv2.initUndistortRectifyMap(k2, saved["D2"], r2, p2, size, cv2.CV_16SC2))
                ones = np.ones(size[::-1], np.uint8)
                self.rectified_masks = [cv2.remap(ones, *maps, interpolation=cv2.INTER_NEAREST).astype(bool)
                                        for maps in self.maps]
                self.sign = 1 if p2[0, 3] < 0 else -1
                self.fx_baseline = abs(float(p2[0, 3]))
                self.rectified_intrinsic = p1[:, :3].copy()
                # Projection matrices for stereo triangulation in rectified_left.
                self.rectified_projections = (p1.copy(), p2.copy())
                # P2 = K [I | T] in the rectified left-camera frame.
                # Convert right-camera PnP coordinates back into left-camera coordinates.
                self.right_to_left_offset = np.array(
                    [-p2[0, 3] / p2[0, 0], 0.0, 0.0], dtype=np.float64)
        if self.calibrated and disparities is None:
            # d = f*B/Z. A fixed 64-pixel search previously excluded everything
            # nearer than ~94 cm on this rig, regardless of the colour scale.
            smallest = max(0, math.floor(self.fx_baseline / high) - 1)
            largest = math.ceil(self.fx_baseline / low) + 1
            self.disparities = max(16, math.ceil((largest - smallest + 1) / 16) * 16)
            self.minimum = smallest if self.sign > 0 else -(smallest + self.disparities - 1)
        else:
            self.disparities = max(16, math.ceil((disparities or 64) * size[0] / 640 / 16) * 16)
            self.minimum = 0 if self.sign > 0 else -self.disparities
        if self.disparities + max(0, self.minimum) >= size[0] - 16:
            raise ValueError("Depth search exceeds image width: increase --near or reduce --disparities")
        self.maximum = self.minimum + self.disparities - 1
        # OpenCV otherwise discards an entire max-disparity-wide strip even
        # where the actual (smaller) disparity has a valid correspondence.
        # Padding enables that search; actual-image bounds and patch masks
        # below still reject matches outside the original observations.
        self.padding = max(abs(self.minimum), abs(self.maximum)) + 16 if pad_edges else 0
        if self.rectified_masks is None:
            self.rectified_masks = [np.ones(size[::-1], bool) for _ in range(2)]
        block = 15 if method == "bm" else 5
        self.rectified_masks = [cv2.erode(mask.astype(np.uint8), np.ones((block, block), np.uint8),
                                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                                for mask in self.rectified_masks]
        max_positive_disparity = max(self.sign * self.minimum, self.sign * self.maximum)
        self.effective_near = (self.fx_baseline / max_positive_disparity
                               if self.calibrated else None)
        self.matcher = self._make_matcher(self.minimum)
        self.reverse_minimum = -self.maximum
        self.reverse_matcher = self._make_matcher(self.reverse_minimum) if lr_check else None
        self.last_rectified = None

    def _make_matcher(self, minimum):
        if self.method == "bm":
            matcher = cv2.StereoBM_create(self.disparities, 15)
            matcher.setMinDisparity(minimum)
            matcher.setUniquenessRatio(12)
            matcher.setTextureThreshold(10)
            matcher.setSpeckleWindowSize(50)
            matcher.setSpeckleRange(2)
            return matcher
        return cv2.StereoSGBM_create(
            minDisparity=minimum, numDisparities=self.disparities, blockSize=5,
            P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
            speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify(self, left, right):
        gray = []
        for index, image in enumerate((left, right)):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if (image.shape[1], image.shape[0]) != self.size:
                image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            if self.maps:
                image = cv2.remap(image, *self.maps[index], interpolation=cv2.INTER_LINEAR)
            gray.append(image)
        return tuple(gray)

    def compute(self, left, right):
        gray = self.last_rectified = self.rectify(left, right)
        matching_gray = ([cv2.copyMakeBorder(image, 0, 0, self.padding, self.padding,
                                             cv2.BORDER_REPLICATE) for image in gray]
                         if self.padding else gray)
        raw = self.matcher.compute(*matching_gray)
        if self.padding:
            raw = raw[:, self.padding:self.padding + self.size[0]]
        signed_disparity = raw.astype(np.float32) / 16
        disparity = signed_disparity * self.sign
        valid = (raw > (self.minimum - 1) * 16) & (disparity > 0)
        # A left pixel x corresponds to right pixel x-d, including negative
        # disparities when the physical camera order is reversed.
        xr = np.rint(np.arange(self.size[0])[None, :] - signed_disparity).astype(np.int32)
        valid &= (xr >= 0) & (xr < self.size[0])
        xr = np.clip(xr, 0, self.size[0] - 1)
        y = np.arange(self.size[1])[:, None]
        if self.rectified_masks is not None:
            valid &= self.rectified_masks[0] & self.rectified_masks[1][y, xr]
        if self.reverse_matcher is not None:
            reverse = self.reverse_matcher.compute(matching_gray[1], matching_gray[0])
            if self.padding:
                reverse = reverse[:, self.padding:self.padding + self.size[0]]
            reverse = reverse[y, xr]
            valid &= reverse > (self.reverse_minimum - 1) * 16
            valid &= np.abs(signed_disparity + reverse.astype(np.float32) / 16) <= self.lr_threshold
        if self.calibrated:
            depth = np.full(raw.shape, np.nan, np.float32)
            np.divide(self.fx_baseline, disparity, out=depth, where=valid)
            # Do not paint physically out-of-range mismatches as valid blue
            # surfaces at the far end of the colour scale.
            valid &= np.isfinite(depth) & (depth >= self.low) & (depth <= self.high)
            depth[~valid] = np.nan
            colors = self.renderer.colorize(depth)
        else:
            # Diagnostic disparity has no metric interpretation before stereo
            # calibration. Never apply a guessed focal length/baseline.
            normalized = np.clip(disparity / self.disparities, 0, 1)
            colors = cv2.applyColorMap(cv2.convertScaleAbs(normalized, alpha=255), cv2.COLORMAP_TURBO)
            colors[~valid] = 0
            depth = None
        return colors, depth, float(np.mean(valid))


def fit_image(image, size):
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    scaled = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((height, width, 3), np.uint8)
    x, y = (width - scaled.shape[1]) // 2, (height - scaled.shape[0]) // 2
    output[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
    return output


def label(image, text, y=28, color=(255, 255, 255)):
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def sample_depth(depth, point=(0.5, 0.5), radius=3, allow_nonpositive=False):
    """Median Z in a small neighbourhood; never fill an entirely missing area."""
    if depth is None:
        return None
    x = round(np.clip(point[0], 0, 1) * (depth.shape[1] - 1))
    y = round(np.clip(point[1], 0, 1) * (depth.shape[0] - 1))
    patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    values = patch[np.isfinite(patch) & ((patch > 0) if not allow_nonpositive else True)]
    return float(np.median(values)) if values.size >= 3 else None


class ArucoRobotTracker:
    """Detect matching robot tags in both rectified stereo views and fuse their poses."""

    ALLOWED_IDS = set(range(6))

    def __init__(self, marker_size_m=0.05, glow_size_m=0.16):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("ArUco support is missing; install an OpenCV build with the aruco module")
        self.marker_size = float(marker_size_m)
        self.glow_size = float(glow_size_m)
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, parameters)
        half = self.marker_size / 2
        self.object_corners = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0]
        ], dtype=np.float64)
        glow_half = self.glow_size / 2
        self.glow_corners = np.array([
            [-glow_half, glow_half, 0], [glow_half, glow_half, 0],
            [glow_half, -glow_half, 0], [-glow_half, -glow_half, 0]
        ], dtype=np.float64)

    def _detect_one(self, gray, intrinsic):
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses, accepted_corners, accepted_ids = {}, [], []
        zero_distortion = np.zeros((5, 1), dtype=np.float64)
        if ids is not None:
            for image_corners, raw_id in zip(corners, ids.reshape(-1)):
                marker_id = int(raw_id)
                if marker_id not in self.ALLOWED_IDS:
                    continue
                image_points = image_corners.reshape(4, 2).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    self.object_corners, image_points, intrinsic, zero_distortion,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
                if not np.isfinite(tvec).all() or tvec[2] <= 0:
                    continue
                rotation, _ = cv2.Rodrigues(rvec)
                poses[marker_id] = {
                    "image_points": image_points,
                    "center_pnp": tvec,
                    "rotation_pnp": rotation,
                }
                accepted_corners.append(image_corners.reshape(1, 4, 2).astype(np.float32))
                accepted_ids.append(marker_id)
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if accepted_ids:
            cv2.aruco.drawDetectedMarkers(
                annotated, accepted_corners,
                np.asarray(accepted_ids, dtype=np.int32).reshape(-1, 1))
            for marker_id, image_corners in zip(accepted_ids, accepted_corners):
                x, y = np.rint(image_corners[0].mean(axis=0)).astype(int)
                cv2.putText(annotated, f"ROBOT {marker_id}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return poses, annotated

    def detect_pair(self, left_gray, right_gray, intrinsic, projection_matrices):
        left_poses, left_annotated = self._detect_one(left_gray, intrinsic)
        right_poses, right_annotated = self._detect_one(right_gray, intrinsic)
        p_left, p_right = projection_matrices
        found = []
        for marker_id in sorted(set(left_poses) & set(right_poses)):
            lp, rp = left_poses[marker_id], right_poses[marker_id]
            # Both images are already rectified. Triangulate corresponding
            # corners into the exact rectified_left frame used by projector_model.
            q = cv2.triangulatePoints(p_left, p_right,
                                      lp["image_points"].T, rp["image_points"].T)
            with np.errstate(divide="ignore", invalid="ignore"):
                xyz = (q[:3] / q[3:4]).T
            use_stereo = (xyz.shape == (4, 3) and np.isfinite(xyz).all()
                          and np.all(xyz[:, 2] > 0.05) and np.all(xyz[:, 2] < 8.0))
            if use_stereo:
                centre = xyz.mean(axis=0)
                # Marker local X points from corner 0 to 1; local Y from 0 to 3.
                axis_x = ((xyz[1] - xyz[0]) + (xyz[2] - xyz[3])) * 0.5
                axis_y = ((xyz[3] - xyz[0]) + (xyz[2] - xyz[1])) * 0.5
                nx = np.linalg.norm(axis_x)
                if nx < 1e-8:
                    use_stereo = False
                else:
                    axis_x /= nx
                    axis_y -= axis_x * float(axis_x @ axis_y)
                    ny = np.linalg.norm(axis_y)
                    if ny < 1e-8:
                        use_stereo = False
                    else:
                        axis_y /= ny
            if not use_stereo:
                # Safe fallback: retain the working left-camera PnP path so
                # a noisy/missing stereo correspondence never blanks projection.
                centre = lp["center_pnp"]
                rotation = lp["rotation_pnp"]
                axis_x = rotation[:, 0]
                axis_y = -rotation[:, 1]
            half = self.glow_size * 0.5
            # Axis Y is image-down; these four points form an oriented frame
            # centred exactly on the estimated marker centre.
            glow = np.array([
                centre - axis_x * half - axis_y * half,
                centre + axis_x * half - axis_y * half,
                centre + axis_x * half + axis_y * half,
                centre - axis_x * half + axis_y * half,
            ], dtype=np.float64)
            found.append({"id": marker_id, "center": centre,
                          "camera_corners": glow, "stereo_used": bool(use_stereo)})
        return found, left_annotated, right_annotated

    @staticmethod
    def projected_markers(detections, projector_model, rectangle):
        """Project marker position and direction to the printed tag's top edge."""
        left, top, right, bottom = rectangle
        markers = []
        for detection in detections:
            forward = detection["camera_corners"][:2].mean(axis=0)
            pixels, depth = project_points(projector_model.matrix, [detection["center"], forward])
            if not np.isfinite(pixels).all() or not np.all(depth > 0):
                continue
            x, y = pixels[0]
            dx, dy = pixels[1] - pixels[0]
            if not (left <= x < right and top <= y < bottom) or math.hypot(dx, dy) < 1e-6:
                continue
            markers.append({"id": detection["id"], "x": float(x), "y": float(y),
                            "angle": math.atan2(dy, dx)})
        return markers

    @staticmethod
    def overlay(base, detections, projector_model, rectangle):
        if not detections or projector_model is None:
            return base.copy()
        output = base.copy()
        h, w = output.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        polygons, centres = [], []
        for detection in detections:
            pixels, z = project_points(projector_model.matrix, detection["camera_corners"])
            centre_pixel, centre_z = project_points(
                projector_model.matrix, np.asarray(detection["center"], dtype=np.float64).reshape(1, 3))
            if (not np.isfinite(pixels).all() or not np.all(z > 0)
                    or not np.isfinite(centre_pixel).all() or not np.all(centre_z > 0)):
                continue
            polygon = np.rint(pixels).astype(np.int32)
            centre = np.rint(centre_pixel[0]).astype(int)
            polygons.append(polygon)
            centres.append(centre)
            cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
        if not polygons:
            return output
        for polygon in polygons:
            solid = np.zeros_like(mask)
            cv2.fillPoly(solid, [polygon], 255, lineType=cv2.LINE_AA)
            inner = cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            mask = cv2.max(mask, cv2.subtract(solid, inner))
        glow = cv2.GaussianBlur(mask, (0, 0), 9.0)
        white = np.repeat((glow.astype(np.float32) * 0.78)[:, :, None], 3, axis=2)
        output = np.maximum(output, white.astype(np.uint8))
        for polygon, centre in zip(polygons, centres):
            cv2.polylines(output, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(output, tuple(centre), 4, (255, 255, 255), -1, cv2.LINE_AA)
        left, top, right, bottom = rectangle
        output[:top] = 0
        output[bottom:] = 0
        output[:, :left] = 0
        output[:, right:] = 0
        return output


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=int, default=None, help="Logical left camera; default from calibration")
    parser.add_argument("--right", type=int, default=None, help="Logical right camera; default from calibration")
    parser.add_argument("--calibration", type=Path, default=Path("stereo_calibration/stereo_calibration.npz"))
    parser.add_argument("--preview", action="store_true", help="Allow diagnostic disparity without calibration")
    parser.add_argument("--size", choices=["640x360", "1280x720"], default="640x360")
    parser.add_argument("--method", choices=["bm", "sgbm"], default="sgbm")
    parser.add_argument("--disparities", type=int, default=None,
                        help="Override search length at width 640; default: derive from calibration and near/far")
    parser.add_argument("--lr-check", action=argparse.BooleanOptionalAction, default=True,
                        help="Reject inconsistent left/right matches (additional computation)")
    parser.add_argument("--lr-threshold", type=float, default=1.5, help="Consistency tolerance in pixels at width 640")
    parser.add_argument("--pad-edges", action=argparse.BooleanOptionalAction, default=True,
                        help="Recover available edge correspondences with a padded search (additional computation)")
    parser.add_argument("--max-skew-ms", type=float, default=50, help="Software receipt-time pairing tolerance")
    parser.add_argument("--focus-left", type=float)
    parser.add_argument("--focus-right", type=float)
    parser.add_argument("--near", type=float, default=0.3)
    parser.add_argument("--far", type=float, default=4)
    parser.add_argument("--appearance", choices=STYLES, default="smooth",
                        help="Presentation cleanup: raw, balanced or smooth (S cycles while running)")
    parser.add_argument("--fill", choices=("surface", "conservative"), default="surface",
                        help="Dense estimated surface for projection, or conservative gaps (F toggles)")
    parser.add_argument("--update-mode", choices=("changes", "live"), default="changes",
                        help="Rate-limited updates while the scene changes (default), or every processed frame")
    parser.add_argument("--check-interval", type=float, default=0.5, help="Seconds between depth checks in changes mode")
    parser.add_argument("--update-interval", "--settle-time", dest="update_interval", type=float, default=1.0,
                        help="Minimum seconds between updates in changes mode (settle-time is a legacy alias)")
    parser.add_argument("--change-mm", type=float, default=20, help="Minimum local depth change in millimetres")
    parser.add_argument("--change-area", type=float, default=0.003, help="Minimum connected changed fraction of the image")
    parser.add_argument("--color-near", type=float, help="Near end of fixed colour scale; default: --near")
    parser.add_argument("--color-far", type=float, help="Far end of fixed colour scale; default: --far")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--x", type=int, default=1920)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--diagnostics-fps", type=float, default=5,
                        help="Camera preview refresh rate; computation and projector run independently")
    parser.add_argument("--work-area", type=Path, default=Path("stereo_calibration/work_area.json"),
                        help="Saved projector footprint in rectified camera coordinates; loaded when present")
    parser.add_argument("--calibrate-area", action="store_true",
                        help="Show a white projector field and select its four outer corners (also P while running)")
    parser.add_argument("--projector-inset-x", type=int, default=None,
                        help="Black columns at each end of the projector width; default: saved work area, or 0")
    parser.add_argument("--projector-inset-y", type=int, default=None,
                        help="Black rows above and below the active projector field")
    parser.add_argument("--height-map", action="store_true", help="Colour height above the measured floor plane")
    parser.add_argument("--floor-reference", type=Path, default=Path("stereo_calibration/floor_reference.json"))
    parser.add_argument("--height-max", type=float, default=.5, help="Maximum illuminated height above the floor, metres")
    parser.add_argument("--height-gamma", type=float, default=.5, help="Below 1 increases contrast near the floor")
    parser.add_argument("--height-noise-mm", type=float, default=5., help="Neutral floor band for display")
    parser.add_argument("--projector-model", type=Path, default=Path("stereo_calibration/projector_model.json"))
    parser.add_argument("--projector-3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-m", type=float,
                        help="Known optical-axis distance in metres; select a surface, then press V to check")
    parser.add_argument("--validation-frames", type=int, default=20,
                        help="Fresh stereo pairs per V check (minimum 10)")
    parser.add_argument("--compare", action="store_true", help="Measure both resolutions sequentially and exit")
    parser.add_argument("--seconds", type=float, default=0, help="Duration per mode, 0=until closed (compare: 10s)")
    parser.add_argument("--output", type=Path, default=Path("results/live"))
    parser.add_argument("--aruco", action="store_true", help="Track robot markers 0-5 (DICT_4X4_50) and project a white halo")
    parser.add_argument("--aruco-marker-mm", type=float, default=50.0, help="Physical marker side length in mm")
    parser.add_argument("--aruco-glow-mm", type=float, default=160.0, help="Projected halo square side length in mm")
    parser.add_argument("--lan-preview", action="store_true", help="Serve the colourized depth preview over LAN HTTP")
    parser.add_argument("--lan-preview-port", type=int, default=8765, help="LAN preview HTTP port")
    parser.add_argument("--lan-preview-origin",
                        help="Allow marker reads from this additional web origin (e.g. https://example.com)")
    args = parser.parse_args()
    if args.lan_preview_origin:
        try:
            origin = args.lan_preview_origin
            parsed = urlsplit(origin)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username is not None or origin != f"{parsed.scheme}://{parsed.netloc}"
                    or parsed.port == 0 or any(not 33 <= ord(char) <= 126 for char in origin)):
                raise ValueError
        except ValueError:
            parser.error("Use an HTTP(S) origin without a path for --lan-preview-origin")
    saved_indices = [0, 1]
    if args.calibration.is_file() and (args.left is None or args.right is None):
        with np.load(args.calibration, allow_pickle=False) as saved:
            if "camera_indices" in saved:
                saved_indices = saved["camera_indices"].tolist()
    args.left = saved_indices[0] if args.left is None else args.left
    args.right = saved_indices[1] if args.right is None else args.right
    args.color_near = args.near if args.color_near is None else args.color_near
    args.color_far = args.far if args.color_far is None else args.color_far
    if args.left == args.right or min(args.left, args.right) < 0:
        parser.error("Select two different nonnegative camera indices")
    if args.disparities is not None and (args.disparities < 16 or args.disparities % 16 or args.disparities * 2 >= 1280):
        parser.error("disparities must be a multiple of 16, between 16 and 624")
    if not all(math.isfinite(n) for n in (args.near, args.far, args.seconds, args.max_skew_ms,
                                         args.diagnostics_fps, args.lr_threshold, args.color_near, args.color_far,
                                         args.check_interval, args.update_interval, args.change_mm, args.change_area)):
        parser.error("Numeric settings must be finite")
    if args.far <= args.near or args.near <= 0 or args.seconds < 0 or args.max_skew_ms <= 0:
        parser.error("Need 0 < near < far, seconds >= 0, max-skew-ms > 0")
    if args.width < 2 or args.height < 2:
        parser.error("Output dimensions must be >= 2")
    if args.projector_inset_x is not None and not 0 <= 2 * args.projector_inset_x <= args.width - 2:
        parser.error("projector-inset-x must leave at least two visible columns")
    if args.projector_inset_y is not None and not 0 <= 2 * args.projector_inset_y <= args.height - 2:
        parser.error("projector-inset-y must leave at least two visible rows")
    if (not all(math.isfinite(v) for v in (args.height_max, args.height_gamma, args.height_noise_mm))
            or args.height_gamma <= 0 or args.height_noise_mm < 0 or args.height_max <= args.height_noise_mm / 1000):
        parser.error("Need positive height-gamma and height-max above the nonnegative noise band")
    if not 0 < args.color_near < args.color_far:
        parser.error("Need 0 < color-near < color-far")
    if args.diagnostics_fps <= 0:
        parser.error("diagnostics-fps must be > 0")
    if args.lr_threshold <= 0:
        parser.error("lr-threshold must be > 0")
    if args.check_interval <= 0 or args.update_interval <= 0 or args.change_mm <= 0 or not 0 < args.change_area <= 1:
        parser.error("Need check-interval > 0, update-interval > 0, change-mm > 0 and 0 < change-area <= 1")
    if args.reference_m is not None and (not math.isfinite(args.reference_m) or args.reference_m <= 0):
        parser.error("reference-m must be finite and positive")
    if (not math.isfinite(args.aruco_marker_mm) or not math.isfinite(args.aruco_glow_mm)
            or args.aruco_marker_mm <= 0 or args.aruco_glow_mm <= args.aruco_marker_mm):
        parser.error("ArUco marker/glow sizes must be positive and glow larger than marker")
    if args.validation_frames < 10:
        parser.error("validation-frames must be >= 10")
    if any(value is not None and not math.isfinite(value) for value in (args.focus_left, args.focus_right)):
        parser.error("Focus must be finite")
    if not args.calibration.is_file() and not args.preview:
        parser.error("Stereo calibration missing: run stereo_calibrate.py, or use --preview for diagnostics")
    return args


def run(args):
    dpi_awareness()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = args.calibration if args.calibration.is_file() else None
    calibration_hash = hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration else None
    focus_values = [args.focus_left, args.focus_right]
    if calibration is not None:
        with np.load(calibration, allow_pickle=False) as saved:
            if "camera_indices" in saved and saved["camera_indices"].tolist() != [args.left, args.right]:
                raise ValueError("Use the same camera order as in stereo calibration")
            if "focus_values" in saved:
                for index, value in enumerate(saved["focus_values"]):
                    if focus_values[index] is not None and abs(focus_values[index] - value) > 0.5:
                        raise ValueError("Changed focus requires new stereo calibration")
                    focus_values[index] = float(value)
    engines = {size: StereoEngine(size, args.method, calibration, args.disparities, args.near, args.far,
                                 args.lr_check, args.lr_threshold, args.pad_edges)
               for size in SIZES}
    appearance_mode = args.appearance
    fill_mode = args.fill
    def new_updates():
        return SurfaceUpdates(args.change_mm / 1000, args.update_interval, args.change_area,
                              block=max(1, round(8 * size[0] / 640)), max_gap=max(5, args.check_interval * 4),
                              region=work_area.mask(size[::-1]) if work_area else None)
    appearances = {size: DepthAppearance(appearance_mode, size[0] / 640, fill_mode) for size in SIZES}
    color_renderer = Renderer(args.color_near, args.color_far, bright=True)
    size = tuple(map(int, args.size.split("x")))
    output_size = (args.width, args.height)
    floor_reference = (FloorPlane.load(args.floor_reference, calibration_hash)
                       if args.height_map and not args.calibrate_area else None)
    if floor_reference is not None and calibration is None:
        raise ValueError("Height mode needs stereo calibration")
    height_renderers = ({item: HeightRenderer(floor_reference, engines[item].rectified_intrinsic,
                                             args.height_max, args.height_gamma, args.height_noise_mm / 1000)
                         for item in SIZES} if floor_reference else {})
    projector_model = (ProjectorModel.load(args.projector_model, output_size, calibration_hash)
                       if args.projector_3d and not args.calibrate_area and args.projector_model.is_file() else None)
    aruco_tracker = (ArucoRobotTracker(args.aruco_marker_mm / 1000, args.aruco_glow_mm / 1000)
                     if args.aruco else None)
    if aruco_tracker is not None and projector_model is None:
        raise ValueError("ArUco projector glow needs the calibrated 3D projector model; enable --projector-3d")
    requested_rect = (projector_bounds(output_size, args.projector_inset_x or 0, args.projector_inset_y or 0)
                      if args.projector_inset_x is not None or args.projector_inset_y is not None else None)
    work_area = None
    if args.work_area.is_file():
        try:
            work_area = WorkArea.load(args.work_area, output_size, calibration_hash, requested_rect)
        except (ValueError, KeyError):
            if not args.calibrate_area:
                raise
    if (work_area or args.calibrate_area) and calibration is None:
        raise ValueError("Calibrate stereo cameras before setting the projector work area")
    updates = new_updates()
    field_rect = work_area.projector_rect if work_area else requested_rect or projector_bounds(output_size)
    if aruco_tracker is not None and calibration is None:
        raise ValueError("ArUco tracking needs calibrated stereo cameras")
    field_left, field_top, field_right, field_bottom = field_rect
    field_size = (field_right - field_left, field_bottom - field_top)
    cameras = [Camera(args.left, focus_values[0]), Camera(args.right, focus_values[1])]
    lan_preview = None
    if args.lan_preview:
        from depth_http_preview import DepthPreview
        lan_preview = DepthPreview(port=args.lan_preview_port, calibrated=calibration is not None,
                                   projection_size=output_size, available_rect=field_rect,
                                   marker_origin=args.lan_preview_origin)
    aruco_stop = threading.Event()
    aruco_result_lock = threading.Lock()
    aruco_result = aruco_thread = None

    def track_aruco():
        nonlocal aruco_result
        # The 720p maps keep small tags detectable without a disparity pass.
        engine = engines[CAPTURE_SIZE]
        last_pair = (0, 0)
        while not aruco_stop.is_set():
            tick = time.perf_counter()
            pair = choose_pair(cameras[0].snapshot(), cameras[1].snapshot(), last_pair,
                               args.max_skew_ms / 1000)
            if pair:
                left, right = pair
                last_pair = left.sequence, right.sequence
                stamp = min(left.timestamp, right.timestamp)
                rectified = engine.rectify(left.image, right.image)
                result = aruco_tracker.detect_pair(*rectified, engine.rectified_intrinsic,
                                                   engine.rectified_projections)
                with aruco_result_lock:
                    aruco_result = stamp, result
                markers = aruco_tracker.projected_markers(result[0], projector_model, field_rect)
                lan_preview.update_markers(markers, stamp)
            aruco_stop.wait(max(0, ARUCO_INTERVAL - (time.perf_counter() - tick)))

    previous = (0, 0)
    aruco_detections = []
    aruco_annotated = None
    aruco_right_annotated = None
    projector_base = np.zeros((args.height, args.width, 3), np.uint8)
    periods = []
    measurements = deque(maxlen=900)
    frame_count = 0
    publication_count = 0
    next_probe = 0
    published_depth = published_stamp = None
    shown_stamp = None
    display_result = colors = None
    style_stats = {"display_fraction": 0.0, "interpolated_fraction": 0.0, "held_fraction": 0.0,
                   "completed_fraction": 0.0}
    canonical = np.zeros((720, 1280, 3), np.uint8)
    output = np.zeros((args.height, args.width, 3), np.uint8)
    started = None
    last_frame = time.perf_counter()
    opened = time.perf_counter()
    mode = 0
    completed = False
    last_preview = -1
    duration = args.seconds or (10 if args.compare else 0)
    if args.compare:
        size = SIZES[0]
        updates = new_updates()
    cv2.namedWindow(PROJECTOR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PROJECTOR, *output_size)
    cv2.moveWindow(PROJECTOR, args.x, args.y)
    cv2.imshow(PROJECTOR, np.zeros((args.height, args.width, 3), np.uint8))
    cv2.waitKey(1)
    if not args.windowed:
        cv2.setWindowProperty(PROJECTOR, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow(PREVIEW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW, 1100, 650)
    cv2.moveWindow(PREVIEW, 30, 40)
    def area_center():
        return (tuple((work_area.points.mean(axis=0) / (np.array(work_area.input_size) - 1)).tolist())
                if work_area else (0.5, 0.5))

    sample_point = area_center()
    needs_area_calibration = args.calibrate_area
    pending_check = None
    distance_checks = []
    check_status = (f"V: check against {args.reference_m:g} m" if args.reference_m is not None
                    else "Distance check: start with --reference-m METRES")

    def clicked(event, x, y, flags, param):
        nonlocal sample_point, pending_check
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < 640 and 360 <= y < 720:
            sample_point = (x / 639, (y - 360) / 359)
            pending_check = None

    cv2.setMouseCallback(PREVIEW, clicked)

    def finish_period():
        if started is None or not measurements:
            return
        elapsed = time.perf_counter() - started
        period = {"size": list(size), "method": engines[size].method, "calibrated": calibration is not None,
                  "min_disparity": engines[size].minimum, "num_disparities": engines[size].disparities,
                  "requested_depth_range_m": [args.near, args.far],
                  "effective_search_near_m": engines[size].effective_near,
                  "left_right_check": args.lr_check,
                  "padded_search": args.pad_edges,
                  "appearance": appearance_mode if calibration else "uncalibrated_disparity",
                  "fill": fill_mode,
                  "update_mode": args.update_mode,
                  "check_interval_s": args.check_interval,
                  "update_interval_s": args.update_interval,
                  "change_threshold_m": args.change_mm / 1000,
                  "change_area_fraction": args.change_area,
                  "work_area": str(args.work_area) if work_area else None,
                  "work_area_fraction": float(work_area.mask(size[::-1]).mean()) if work_area else 1.,
                  "work_area_source_points": work_area.points.tolist() if work_area else None,
                  "work_area_input_size": list(work_area.input_size) if work_area else None,
                  "projector_rect": list(field_rect),
                  "height_mode": floor_reference is not None,
                  "height_max_m": args.height_max if floor_reference else None,
                  "height_gamma": args.height_gamma if floor_reference else None,
                  "above_height_range": "black" if floor_reference else None,
                  "floor_plane_m": floor_reference.plane.tolist() if floor_reference else None,
                  "projector_mapping": "3d" if projector_model else "floor_homography" if work_area else "image_fit",
                  "projector_quality": projector_model.quality if projector_model else None,
                  "color_depth_range_m": [args.color_near, args.color_far],
                  "frames": frame_count, "elapsed_s": elapsed, "output_fps": publication_count / elapsed,
                  "depth_checks": frame_count, "depth_checks_per_second": frame_count / elapsed,
                  "publications": publication_count,
                  "published_capture_time": published_stamp,
                  "published_age_s": time.perf_counter() - published_stamp if published_stamp is not None else None,
                  "statistics_samples": len(measurements),
                  "compute_ms": float(np.mean([row[0] for row in measurements])),
                  "appearance_ms": float(np.mean([row[4] for row in measurements])),
                  "display_fraction": float(np.mean([row[5] for row in measurements])),
                  "interpolated_fraction": float(np.mean([row[6] for row in measurements])),
                  "held_fraction": float(np.mean([row[7] for row in measurements])),
                  "completed_fraction": float(np.mean([row[8] for row in measurements])),
                  "receipt_skew_ms": float(np.mean([row[1] for row in measurements])),
                  "receipt_to_check_ms": float(np.mean([row[3] for row in measurements])),
                  "receipt_to_submit_ms": (float(np.mean([row[3] for row in measurements if row[9]]))
                                            if any(row[9] for row in measurements) else None),
                  "valid_fraction": float(np.mean([row[2] for row in measurements])),
                  "camera_fps": [camera.fps() for camera in cameras]}
        periods.append(period)
        print(json.dumps(period), flush=True)

    try:
        print(f"Starting two cameras; update mode: {args.update_mode}. 1/2: resolution; B/G: algorithm; "
              "S: appearance; F: fill; R: rescan; P: projector area; Esc: stop.", flush=True)
        if aruco_tracker is not None:
            print(f"ArUco tracking enabled: DICT_4X4_50, IDs 0-5, marker {args.aruco_marker_mm:g} mm.",
                  flush=True)
            if lan_preview is not None:
                aruco_thread = threading.Thread(target=track_aruco, name="aruco-stream", daemon=True)
                aruco_thread.start()
        if calibration is None:
            print("UNCALIBRATED diagnostic disparity: colours are not measured depth.", flush=True)
        while True:
            if needs_area_calibration:
                needs_area_calibration = False

                def area_image():
                    frames = cameras[0].snapshot()
                    if not frames or time.perf_counter() - frames[-1].timestamp > .5:
                        return None
                    image = cv2.resize(frames[-1].image, SIZES[0], interpolation=cv2.INTER_AREA)
                    return cv2.remap(image, *engines[SIZES[0]].maps[0], interpolation=cv2.INTER_LINEAR)

                selected = calibrate_area(area_image, PROJECTOR, args.work_area, output_size,
                                          calibration_hash, field_rect)
                last_frame = opened = time.perf_counter()
                if selected is not None:
                    finish_period()
                    work_area = selected
                    sample_point = area_center()
                    pending_check = None
                    for appearance in appearances.values():
                        appearance.reset()
                    measurements, started, frame_count = deque(maxlen=900), None, 0
                    publication_count, next_probe = 0, 0
                    updates = new_updates()
                    published_depth = published_stamp = display_result = colors = None
                    style_stats = {key: 0. for key in style_stats}
                    shown_stamp = None
                    canonical = np.zeros((720, 1280, 3), np.uint8)
                    output = np.zeros((args.height, args.width, 3), np.uint8)
                    print(f"Work area saved: {args.work_area}", flush=True)
                elif args.calibrate_area and started is None:
                    break
                cv2.imshow(PROJECTOR, output)
                previous, next_probe, last_preview = (0, 0), 0, -1
            due = args.update_mode == "live" or time.perf_counter() >= next_probe
            pair = (choose_pair(cameras[0].snapshot(), cameras[1].snapshot(), previous,
                                args.max_skew_ms / 1000) if due else None)
            if pair:
                left, right = pair
                previous = (left.sequence, right.sequence)
                now = time.perf_counter()
                if started is None:
                    started = now
                begin = now
                next_probe = now + args.check_interval
                raw_colors, depth, valid = engines[size].compute(left.image, right.image)
                if aruco_thread is not None:
                    with aruco_result_lock:
                        latest = aruco_result
                    if latest is not None and time.perf_counter() - latest[0] < 0.5:
                        aruco_detections, aruco_annotated, aruco_right_annotated = latest[1]
                    else:
                        aruco_detections, aruco_annotated, aruco_right_annotated = [], None, None
                elif aruco_tracker is not None and engines[size].last_rectified is not None:
                    aruco_detections, aruco_annotated, aruco_right_annotated = aruco_tracker.detect_pair(
                        engines[size].last_rectified[0], engines[size].last_rectified[1],
                        engines[size].rectified_intrinsic, engines[size].rectified_projections)
                if work_area and depth is not None:
                    depth = work_area.restrict(depth)
                    raw_colors = raw_colors.copy()
                    raw_colors[~work_area.mask(depth.shape)] = 0
                    valid = float(np.isfinite(depth[work_area.mask(depth.shape)]).mean())
                compute_ms = (time.perf_counter() - begin) * 1000
                appearance_begin = time.perf_counter()
                stamp = min(left.timestamp, right.timestamp)
                accepted = (depth if args.update_mode == "live" else updates.observe(depth, stamp)) if depth is not None else None
                publish = accepted is not None or depth is None
                if publish:
                    if accepted is not None:
                        # Snapshots are independent; do not blend in the old relief.
                        if args.update_mode == "changes":
                            appearances[size].reset()
                        display_result = (work_area.process_display(appearances[size], accepted, stamp,
                                                                   engines[size].last_rectified[0])
                                          if work_area else appearances[size].process(
                                              accepted, stamp, engines[size].last_rectified[0]))
                        colors = (height_renderers[size].render(display_result, accepted) if floor_reference
                                  else display_result.colorize(color_renderer))
                        style_stats = (work_area.statistics(display_result) if work_area
                                       else display_result.statistics())
                        published_depth = accepted.copy()
                    else:
                        colors = raw_colors
                    published_stamp = stamp
                    shown_stamp = stamp
                    publication_count += 1
                    canonical = cv2.resize(colors, CAPTURE_SIZE, interpolation=cv2.INTER_LINEAR)
                    if work_area and projector_model and display_result is not None:
                        output = projector_model.render(colors, display_result.depth,
                                                        engines[size].rectified_intrinsic, field_rect)
                    elif work_area:
                        output = work_area.project(colors)
                    else:
                        output = np.zeros((args.height, args.width, 3), np.uint8)
                        output[field_top:field_bottom, field_left:field_right] = fit_image(canonical, field_size)
                    if work_area:
                        cv2.polylines(canonical, [np.rint(work_area.scaled_points(canonical.shape)).astype(np.int32)],
                                      True, (0, 255, 0), 2)
                    if calibration is None:
                        label(output[field_top:field_bottom, field_left:field_right],
                              "UNCALIBRATED DISPARITY - NOT DEPTH", color=(0, 200, 255))
                    projector_base = output.copy()
                    if aruco_tracker is not None:
                        output = ArucoRobotTracker.overlay(projector_base, aruco_detections,
                                                           projector_model, field_rect)
                    cv2.imshow(PROJECTOR, output)
                    if lan_preview is not None:
                        lan_preview.update(colors)
                appearance_ms = (time.perf_counter() - appearance_begin) * 1000
                if pending_check is not None:
                    value = sample_depth(depth, sample_point)
                    pending_check["attempts"] += 1
                    if value is not None:
                        pending_check["samples"].append(value)
                    check_status = f"Checking: {pending_check['attempts']}/{args.validation_frames}; hold scene still"
                    if pending_check["attempts"] >= args.validation_frames:
                        result = summarize_distance(pending_check["samples"], args.reference_m,
                                                    pending_check["attempts"])
                        result.update({"selected_normalized_xy": list(sample_point), "size": list(size),
                                       "method": engines[size].method, "calibration_sha256": calibration_hash,
                                       "samples_m": pending_check["samples"], "unix_time": time.time()})
                        distance_checks.append(result)
                        check_status = f"Distance check: {'PASS' if result['passed_this_point'] else 'FAIL'}; see JSON report"
                        (args.output / f"distance_check_{time.time_ns()}.json").write_text(
                            json.dumps(result, indent=2), encoding="utf-8")
                        print(json.dumps({"distance_check": result}), flush=True)
                        pending_check = None
                if time.perf_counter() - last_preview >= 1 / args.diagnostics_fps:
                    left_preview = (aruco_annotated if aruco_annotated is not None else left.image)
                    tiles = []
                    right_preview = (aruco_right_annotated if aruco_right_annotated is not None
                                     else right.image)
                    for image in (left_preview, right_preview, canonical):
                        # Keep diagnostic tiles consistently three-channel BGR.
                        if image.ndim == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif image.ndim == 3 and image.shape[2] == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif image.ndim == 3 and image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                        elif image.ndim != 3 or image.shape[2] != 3:
                            raise ValueError(f"Unsupported preview image shape: {image.shape}")
                        tiles.append(cv2.resize(image, (640, 360)))
                    info = np.zeros_like(tiles[0])
                    fps = frame_count / max(0.001, time.perf_counter() - started)
                    selected_depth = sample_depth(depth, sample_point)
                    shown_depth = sample_depth(published_depth, sample_point)
                    readout = f"Live Z: {selected_depth:.3f} m" if selected_depth is not None else "Live Z: no measurement"
                    readout += f"; shown Z: {shown_depth:.3f} m" if shown_depth is not None else "; shown Z: --"
                    height_text = "Distance colour scale"
                    if floor_reference and depth is not None:
                        heights = floor_reference.heights(depth, engines[size].rectified_intrinsic)
                        value = sample_depth(heights, sample_point, allow_nonpositive=True)
                        height_text = (f"Height: {max(0, value):.3f}m; scale 0..{args.height_max:g}m"
                                       if value is not None else "Height: no measurement")
                    height_text += "; projector " + ("3D" if projector_model else "floor plane")
                    cv2.drawMarker(tiles[2], (round(sample_point[0] * 639), round(sample_point[1] * 359)),
                                   (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
                    age = time.perf_counter() - shown_stamp if shown_stamp is not None else 0
                    for i, text in enumerate((f"{size[0]}x{size[1]} {engines[size].method.upper()} checks {fps:.1f}/s; updates {publication_count}",
                                              f"Cameras {cameras[0].fps():.0f}/{cameras[1].fps():.0f} fps; saved map age {age:.1f}s",
                                              f"Stereo: {compute_ms:.1f} ms; finish: {appearance_ms:.1f} ms",
                                              f"Measured {valid:.0%}; shown {style_stats['display_fraction']:.0%}; filled {style_stats['interpolated_fraction']:.1%}; hold {style_stats['held_fraction']:.1%}",
                                              updates.status if args.update_mode == "changes" and calibration else "Live updates",
                                              readout,
                                              f"S: {appearance_mode}; F: {fill_mode}; estimated {style_stats['completed_fraction']:.0%}",
                                              check_status,
                                              height_text,
                                              "Stereo model loaded" if calibration else "NOT CALIBRATED - diagnostic only",
                                              "1/2=size  B/G=method  R=rescan  P=area  Esc=exit")):
                        label(info, text, 24 + 31 * i)
                    cv2.imshow(PREVIEW, np.vstack((np.hstack(tiles[:2]), np.hstack((tiles[2], info)))))
                    last_preview = time.perf_counter()
                key = cv2.waitKey(1) & 0xFF
                submit_time = time.perf_counter()
                measurements.append((compute_ms, abs(left.timestamp - right.timestamp) * 1000, valid,
                                     (submit_time - min(left.timestamp, right.timestamp)) * 1000,
                                     appearance_ms, style_stats["display_fraction"], style_stats["interpolated_fraction"],
                                     style_stats["held_fraction"], style_stats["completed_fraction"], publish))
                frame_count += 1
                last_frame = submit_time
                if duration and submit_time - started >= duration:
                    finish_period()
                    suffix = f"{size[0]}x{size[1]}_{engines[size].method}"
                    original_colors = color_renderer.colorize(depth) if depth is not None else raw_colors
                    for name, image in (("left", left.image), ("right", right.image), ("disparity", original_colors),
                                        ("display", colors), ("projector", output)):
                        if image is not None:
                            (args.output / f"{suffix}_{name}.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
                    for side, image in zip(("left", "right"), engines[size].last_rectified):
                        (args.output / f"{suffix}_rectified_{side}.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
                    if depth is not None:
                        np.save(args.output / f"{suffix}_depth_m.npy", depth)
                        if floor_reference:
                            np.save(args.output / f"{suffix}_height_m.npy",
                                    floor_reference.heights(depth, engines[size].rectified_intrinsic))
                            (args.output / f"{suffix}_height_active_mask.png").write_bytes(cv2.imencode(
                                ".png", height_renderers[size].allowed(depth).astype(np.uint8) * 255)[1].tobytes())
                        if work_area:
                            (args.output / f"{suffix}_work_area_mask.png").write_bytes(
                                cv2.imencode(".png", work_area.mask(depth.shape).astype(np.uint8) * 255)[1].tobytes())
                        (args.output / f"{suffix}_measured_mask.png").write_bytes(
                            cv2.imencode(".png", (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255)[1].tobytes())
                    if display_result is not None:
                        np.save(args.output / f"{suffix}_published_depth_m.npy", published_depth)
                        np.save(args.output / f"{suffix}_display_depth_m.npy", display_result.depth)
                        if floor_reference:
                            np.save(args.output / f"{suffix}_display_height_m.npy",
                                    floor_reference.heights(display_result.depth, engines[size].rectified_intrinsic))
                        for name, mask in (("published_measured_mask", display_result.measured),
                                           ("interpolated_mask", display_result.interpolated),
                                           ("held_mask", display_result.held if display_result.held is not None
                                            else np.zeros(depth.shape, bool)),
                                           ("completed_mask", display_result.completed if display_result.completed is not None
                                            else np.zeros(depth.shape, bool))):
                            (args.output / f"{suffix}_{name}.png").write_bytes(
                                cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes())
                    if args.compare and mode == 0:
                        pending_check = None
                        for appearance in appearances.values():
                            appearance.reset()
                        mode, size = 1, SIZES[1]
                        measurements, started, frame_count = deque(maxlen=900), None, 0
                        publication_count, next_probe = 0, 0
                        updates = new_updates()
                        published_depth = published_stamp = display_result = colors = None
                    else:
                        completed = True
                        break
            else:
                key = cv2.waitKey(5) & 0xFF
                if time.perf_counter() - last_frame > max(2, args.check_interval + 1) and shown_stamp is None:
                    status = np.zeros((args.height, args.width, 3), np.uint8)
                    label(status[field_top:field_bottom, field_left:field_right],
                          "Waiting for two fresh camera frames")
                    cv2.imshow(PROJECTOR, status)
                if time.perf_counter() - opened > 20 and started is None:
                    raise OSError("No paired frames. Check camera indices, other apps and receipt skew tolerance")
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")) and calibration is not None:
                needs_area_calibration = True
            if any(cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 for window in (PROJECTOR, PREVIEW)):
                break
            if key in (ord("v"), ord("V")) and calibration is not None and args.reference_m is not None:
                pending_check = {"samples": [], "attempts": 0}
            if not args.compare and key in (ord("s"), ord("S"), ord("f"), ord("F"), ord("r"), ord("R")):
                finish_period()
                if key in (ord("f"), ord("F")):
                    fill_mode = "conservative" if fill_mode == "surface" else "surface"
                elif key in (ord("s"), ord("S")):
                    modes = list(STYLES)
                    appearance_mode = modes[(modes.index(appearance_mode) + 1) % len(modes)]
                appearances = {item: DepthAppearance(appearance_mode, item[0] / 640, fill_mode) for item in SIZES}
                measurements, started, frame_count = deque(maxlen=900), None, 0
                publication_count, next_probe = 0, 0
                updates = new_updates()
                published_depth = published_stamp = display_result = colors = None
                last_preview = -1
            if not args.compare and key in (ord("1"), ord("2"), ord("b"), ord("g")):
                pending_check = None
                finish_period()
                if key in (ord("1"), ord("2")):
                    size = SIZES[key - ord("1")]
                else:
                    method = "bm" if key == ord("b") else "sgbm"
                    engines = {item: StereoEngine(item, method, calibration, args.disparities, args.near, args.far,
                                                 args.lr_check, args.lr_threshold, args.pad_edges)
                               for item in SIZES}
                for appearance in appearances.values():
                    appearance.reset()
                measurements, started, frame_count = deque(maxlen=900), None, 0
                publication_count, next_probe = 0, 0
                updates = new_updates()
                published_depth = published_stamp = display_result = colors = None
    finally:
        if not completed:
            finish_period()
        if aruco_thread is not None:
            aruco_stop.set()
            aruco_thread.join()
        if lan_preview is not None:
            lan_preview.close()
        for camera in cameras:
            camera.close()
        cv2.destroyAllWindows()
        report = {"calibrated": calibration is not None, "capture_size": list(CAPTURE_SIZE),
                  "calibration_sha256": calibration_hash, "distance_checks": distance_checks,
                  "camera_settings": [camera.info for camera in cameras], "periods": periods,
                  "notes": "Output FPS counts publications, depth_checks counts stereo probes. "
                  "In changes mode the projector holds a static map and updates at a limited rate during changes and motion. "
                  "Motion does not need to stop; published depth is the latest measured frame without temporal trails. "
                  "A saved work area restricts changes, display reconstruction and exports to the projector footprint. "
                  "Its homography aligns a flat floor, not elevated 3D relief. "
                  "Depth checks still run periodically; expensive presentation runs only on publication. "
                  "Timestamps are software receipt times, not hardware exposure synchronization. "
                  "Long interactive periods retain the most recent 900 metric samples; depth_checks counts all processed pairs. "
                  "Uncalibrated disparity cannot establish depth accuracy or compare depth quality."}
        report["notes"] += (" Presentation smoothing and hole interpolation affect only display images/display_depth_m.npy. "
                            "Small enclosed gaps may briefly retain earlier depth with a static camera image; "
                            "held pixels remain inferred and their lifetime is not refreshed by holding. "
                            "Surface completion estimates unobserved regions for presentation only; "
                            "completed_mask marks these points. It is not a full-field accuracy claim. "
                            "Live Z, reference checks and depth_m.npy always use original stereo measurements.")
        (args.output / "live_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        run(arguments())
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
