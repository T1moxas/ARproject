import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from depth_http_preview import DepthPreview
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
