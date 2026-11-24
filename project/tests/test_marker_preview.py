import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from depth_http_preview import DepthPreview
import stereo_live
from stereo_live import ArucoRobotTracker
from surface_geometry import ProjectorModel


class MarkerProjectionTests(unittest.TestCase):
    def test_heading_follows_the_printed_top_edge_in_projector_coordinates(self):
        model = ProjectorModel([[100, 0, 20, 0], [0, 100, 30, 0], [0, 0, 1, 0]],
                               (1280, 800), "")
        square = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0],
                           [0.05, 0.05, 0], [-0.05, 0.05, 0]])
        rotated = square @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        center = np.array([0.1, 0.2, 1])
        outside = np.array([-0.6, 0.2, 1])
        detections = [
            {"id": 0, "center": center, "camera_corners": square + center},
            {"id": 1, "center": center, "camera_corners": rotated + center},
            {"id": 2, "center": outside, "camera_corners": square + outside},
        ]
        result = ArucoRobotTracker.projected_markers(detections, model, (10, 10, 100, 100))
        self.assertEqual([marker["id"] for marker in result], [0, 1])
        self.assertEqual((result[0]["x"], result[0]["y"]), (30, 50))
        self.assertAlmostEqual(result[0]["angle"], -math.pi / 2)
        self.assertAlmostEqual(result[1]["angle"], 0)


class MarkerPreviewTests(unittest.TestCase):
    def setUp(self):
        self.preview = DepthPreview(host="127.0.0.1", port=0, projection_size=(1280, 800),
                                    available_rect=(268, 40, 1012, 760),
                                    marker_origin="https://example.com")
        self.addCleanup(self.preview.close)
        self.base = f"http://127.0.0.1:{self.preview.port}"

    def fetch(self, path, **kwargs):
        return urlopen(Request(self.base + path, **kwargs), timeout=3)

    def test_marker_state_traces_capture_and_excludes_stale_poses(self):
        endpoint = "/api/v1/markers"
        with self.fetch(endpoint) as response:
            state = json.load(response)
        self.assertEqual(state["frame_id"], 0)
        self.assertIsNone(state["age_s"])
        self.assertEqual(state["markers"], [])
        markers = [{"id": 0, "x": 550, "y": 380, "angle": math.pi / 4}]
        self.preview.update_markers(markers, time.perf_counter())
        with self.fetch(endpoint) as response:
            state = json.load(response)
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["projection"], {
            "width": 1280, "height": 800, "available_rect": [268, 40, 1012, 760],
        })
        self.assertEqual(state["frame_id"], 1)
        self.assertEqual(state["markers"], markers)
        self.preview.update_markers(markers, time.perf_counter() - 1)
        with self.fetch(endpoint) as response:
            state = json.load(response)
        self.assertEqual(state["frame_id"], 2)
        self.assertGreaterEqual(state["age_s"], 1)
        self.assertEqual(state["markers"], [])

    def test_camera_health_distinguishes_ready_pairs_from_missing_camera(self):
        cameras = [{"index": 1, "ready": True, "fps": 30.0, "error": ""},
                   {"index": 0, "ready": True, "fps": 29.8, "error": ""}]
        markers = [{"id": 0, "x": 550, "y": 380, "angle": math.pi / 4}]
        self.preview.update_markers(markers, time.perf_counter(), pair_skew_ms=8.0)
        with patch.object(self.preview, "_marker_camera_status", lambda: cameras), \
                patch.object(self.preview, "_max_pair_skew_ms", 50):
            with self.fetch("/api/v1/markers") as response:
                state = json.load(response)
            self.assertEqual(state["tracking"], {"cameras": cameras, "pair_skew_ms": 8.0, "max_skew_ms": 50})
            self.assertEqual(state["markers"], markers)
            cameras[1] = {"index": 0, "ready": False, "fps": 0.0, "error": "Camera 0: cannot open"}
            with self.fetch("/api/v1/markers") as response:
                state = json.load(response)
            self.assertEqual(state["tracking"]["cameras"], cameras)
            self.assertIsNone(state["tracking"]["pair_skew_ms"])
            self.assertEqual(state["markers"], [])

    def test_additional_origin_is_limited_to_marker_reads(self):
        headers = {"Origin": "https://example.com"}
        with self.fetch("/api/v1/markers", method="OPTIONS", headers=headers) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], headers["Origin"])
            self.assertEqual(response.headers["Access-Control-Allow-Methods"], "GET, OPTIONS")
        with self.fetch("/api/v1/state", headers=headers) as response:
            self.assertIsNone(response.headers["Access-Control-Allow-Origin"])

    def test_swarm_assets_are_served_from_the_static_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "dist"
            build.mkdir()
            (build / "index.html").write_text("<html>AR</html>", encoding="utf-8")
            (build / "main.mjs").write_text("export {};", encoding="utf-8")
            (build.parent / "private.txt").write_text("private", encoding="utf-8")
            with patch("depth_http_preview.SWARM_BUILD", build):
                with self.fetch("/swarm/") as response:
                    self.assertEqual(response.read(), b"<html>AR</html>")
                with self.fetch("/swarm/main.mjs?v=1") as response:
                    self.assertEqual(response.headers.get_content_type(), "text/javascript")
                with self.assertRaises(HTTPError) as blocked:
                    self.fetch("/swarm/%2e%2e%2fprivate.txt")
                self.assertEqual(blocked.exception.code, 404)
                blocked.exception.close()


class SwarmOnlyTests(unittest.TestCase):
    def test_startup_uses_calibration_and_reports_markers_without_depth_or_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calibration = root / "stereo.npz"
            model = root / "projector.json"
            output_size = (1280, 800)
            intrinsic = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
            np.savez(calibration, K1=intrinsic, K2=intrinsic, D1=np.zeros(5), D2=np.zeros(5),
                     R=np.eye(3), T=[-0.12, 0, 0], capture_size=stereo_live.CAPTURE_SIZE,
                     camera_indices=[1, 0], focus_values=[0., 0.])
            ProjectorModel(np.column_stack((intrinsic, np.zeros(3))), output_size,
                           hashlib.sha256(calibration.read_bytes()).hexdigest()).save(model)
            argv = ["stereo_live.py", "--swarm-only", "--calibration", str(calibration),
                    "--projector-model", str(model), "--work-area", str(root / "area.json"),
                    "--height-map", "--floor-reference", str(root / "floor.json"),
                    "--output", str(root / "output"), "--seconds", "1", "--lan-preview-port", "0"]
            with patch.object(sys, "argv", argv):
                args = stereo_live.arguments()

            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            tag = cv2.aruco.generateImageMarker(dictionary, 2, 50)
            cameras = []
            previews = []
            started = threading.Event()

            class FakeCamera:
                def __init__(self, index, focus):
                    self.index, self.focus = index, focus
                    self.closed = False
                    self.image = np.full((720, 1280, 3), 255, np.uint8)
                    x = 635 if index == 1 else 517
                    self.image[335:385, x:x + 50] = tag[:, :, None]
                    cameras.append(self)

                def snapshot(self):
                    sequence = int(time.perf_counter() * 30)
                    return [stereo_live.Frame(sequence, sequence / 30 - self.index * .004, self.image)]

                def status(self):
                    return {"index": self.index, "ready": True, "fps": 30.0, "error": ""}

                def close(self):
                    self.closed = True

            def start_preview(**kwargs):
                preview = DepthPreview(host="127.0.0.1", **kwargs)
                previews.append(preview)
                started.set()
                return preview

            with patch.object(stereo_live, "Camera", FakeCamera), \
                    patch("stereo_capture._enumerate_with_deadline", return_value=({}, None)), \
                    patch("depth_http_preview.DepthPreview", start_preview), \
                    patch.object(stereo_live.StereoEngine, "_make_matcher", side_effect=AssertionError), \
                    patch.object(cv2, "namedWindow", side_effect=AssertionError):
                error = []

                def run():
                    try:
                        stereo_live.run(args)
                    except Exception as exc:
                        error.append(exc)
                        started.set()

                service = threading.Thread(target=run, daemon=True)
                service.start()
                try:
                    self.assertTrue(started.wait(3))
                    self.assertFalse(error)
                    endpoint = f"http://127.0.0.1:{previews[0].port}/api/v1/markers"
                    limit = time.perf_counter() + .9
                    while True:
                        with urlopen(endpoint, timeout=1) as response:
                            state = json.load(response)
                        if state["markers"] or time.perf_counter() >= limit:
                            break
                        time.sleep(.02)
                finally:
                    service.join(timeout=3)

            self.assertFalse(service.is_alive())
            self.assertFalse(error)
            self.assertEqual([(cam.index, cam.focus) for cam in cameras], [(1, 0.0), (0, 0.0)])
            self.assertTrue(all(cam.closed for cam in cameras))
            self.assertFalse(args.output.exists())
            self.assertEqual([camera["index"] for camera in state["tracking"]["cameras"]], [1, 0])
            self.assertLessEqual(state["tracking"]["pair_skew_ms"], state["tracking"]["max_skew_ms"])
            self.assertEqual([marker["id"] for marker in state["markers"]], [2])
            self.assertAlmostEqual(state["markers"][0]["x"], 659.5, delta=2)
            self.assertAlmostEqual(state["markers"][0]["y"], 359.5, delta=2)
