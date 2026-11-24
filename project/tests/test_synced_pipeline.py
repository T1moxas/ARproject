"""Exercise real stereo, 3D projection, tracking, hotkeys and HTTP together."""

from collections import deque
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from threading import Thread
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import cv2
import numpy as np

from depth_http_preview import DepthPreview
import stereo_live as live
from surface_geometry import FloorPlane, ProjectorModel
from work_area import WorkArea


class SyncedPipelineTests(unittest.TestCase):
    def test_high_resolution_markers_and_both_depth_outputs_share_the_native_runtime(self):
        native_size = (1920, 1080)
        focal, baseline, floor_depth = 900., .12, 1.5
        disparity = round(1.5 * focal * baseline / floor_depth)
        noise = np.random.default_rng(360).integers(0, 255, native_size[::-1], dtype=np.uint8)
        left = cv2.GaussianBlur(noise, (3, 3), 0)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        left[520:605, 960:1045] = 245
        left[540:585, 980:1025] = cv2.aruco.generateImageMarker(dictionary, 2, 45)
        right = np.zeros_like(left)
        right[:, :-disparity] = left[:, disparity:]
        frames = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (left, right)]
        presentations, services, errors = [], [], []
        keys = deque()

        class Camera:
            def __init__(self, index):
                self.index, self.info, self.closed = index, {"size": list(native_size)}, False

            def snapshot(self):
                sequence = math.floor(time.perf_counter() * 30)
                stamp = sequence / 30 - self.index * .002
                return [live.Frame(sequence, stamp, frames[self.index])]

            def fps(self):
                return 30.

            def status(self):
                return {"index": self.index, "ready": True, "error": "", "fps": self.fps()}

            def close(self):
                self.closed = True

        cameras = [Camera(0), Camera(1)]

        def preview(**kwargs):
            service = DepthPreview(host="127.0.0.1", **kwargs)
            services.append(service)
            return service

        def wait_key(delay):
            time.sleep(min(.005, delay / 1000))
            return keys.popleft() if keys else -1

        def show(name, image):
            if name == live.PROJECTOR:
                presentations.append(image.copy())

        def until(poll, timeout=10):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if errors:
                    raise errors[0]
                result = poll()
                if result:
                    return result
                time.sleep(.04)
            self.fail("Timed out waiting for the combined projection pipeline")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            folder = Path(directory)
            stereo, floor, projector, area = (folder / name for name in ("stereo.npz", "floor.json",
                                                                         "projector.json", "area.json"))
            intrinsic = np.array([[focal, 0, 640.], [0, focal, 360.], [0, 0, 1.]])
            np.savez(stereo, K1=intrinsic, K2=intrinsic, D1=np.zeros(5), D2=np.zeros(5),
                     R=np.eye(3), T=[-baseline, 0, 0], capture_size=[1280, 720],
                     camera_indices=[0, 1], focus_values=[0., 0.])
            fingerprint = hashlib.sha256(stereo.read_bytes()).hexdigest()
            FloorPlane([0, 0, -1, floor_depth], fingerprint).save(floor)
            output_intrinsic = intrinsic.copy()
            output_intrinsic[1, 2] += 40
            ProjectorModel(np.column_stack((output_intrinsic, np.zeros(3))), (1280, 800),
                           fingerprint).save(projector)
            rectangle = (100, 80, 1180, 720)
            WorkArea([[60, 28], [578, 28], [578, 332], [60, 332]], (640, 360), (1280, 800),
                     fingerprint, rectangle).save(area)
            argv = ["stereo_live.py", "--height-map", "--aruco", "--no-aruco-project-halo",
                    "--windowed", "--lan-preview", "--lan-preview-port", "0", "--seconds", "15",
                    "--output", str(folder / "output"), "--calibration", str(stereo),
                    "--floor-reference", str(floor), "--projector-model", str(projector),
                    "--work-area", str(area)]
            stack.enter_context(patch.object(sys, "argv", argv))
            args = live.arguments()
            profile = {"size": list(native_size), "verified": True, "reason": "Test stereo pair",
                       "common_high_modes": [list(native_size)]}
            stack.enter_context(patch.object(live, "open_stereo_cameras", return_value=(cameras, profile)))
            stack.enter_context(patch("depth_http_preview.DepthPreview", preview))
            for name in ("namedWindow", "resizeWindow", "moveWindow", "setWindowProperty",
                         "setMouseCallback", "destroyAllWindows"):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, "waitKey", wait_key))
            stack.enter_context(patch.object(cv2, "imshow", show))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))

            def run():
                try:
                    live.run(args)
                except Exception as error:
                    errors.append(error)

            thread = Thread(target=run)
            thread.start()
            try:
                service = until(lambda: services[0] if services else None)
                base = f"http://127.0.0.1:{service.port}/api/v1/"

                def get(path):
                    with urlopen(base + path, timeout=3) as response:
                        return response.headers, response.read()

                def state(path="state"):
                    return json.loads(get(path)[1])

                def after(snapshot_id=0):
                    body = state()["snapshot"]
                    return body if body and body["id"] > snapshot_id else None

                def visible_marker():
                    body = state("markers")
                    return body if any(mark["visible"] for mark in body["markers"]) else None

                first = until(after)
                marker_frame = until(visible_marker)
                marker, = marker_frame["markers"]
                self.assertEqual(marker["id"], 2)
                self.assertEqual(marker_frame["tracking"]["capture_size"], list(native_size))
                self.assertEqual(len(marker["position_m"]), 3)
                png_headers, png = get("depth.png")
                jpg_headers, jpg = get("preview.jpg")
                self.assertEqual(png_headers["X-Snapshot-Id"], str(first["id"]))
                self.assertEqual(jpg_headers["X-Snapshot-Id"], str(first["id"]))
                layer = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
                self.assertEqual(layer.shape, (300, 480, 4))
                self.assertTrue(layer[:, :, 3].any())
                self.assertFalse(layer[0, :, 3].any())
                self.assertEqual(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape, (800, 1280, 3))
                self.assertFalse(presentations[-1][:, :100].any())
                captured = state()["metrics"]["capture_time_unix_ms"]
                keys.append(ord("s"))
                second = until(lambda: after(first["id"]))
                self.assertEqual(state("projection")["settings"]["appearance"], "raw")
                keys.append(ord("f"))
                until(lambda: after(second["id"]))
                self.assertEqual(state("projection")["settings"]["fill"], "conservative")
                self.assertAlmostEqual(state()["metrics"]["capture_time_unix_ms"], captured, delta=2)
                self.assertGreater(state("markers")["frame_id"], marker_frame["frame_id"])
            finally:
                keys.append(27)
                thread.join(timeout=18)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
        self.assertTrue(all(camera.closed for camera in cameras))
