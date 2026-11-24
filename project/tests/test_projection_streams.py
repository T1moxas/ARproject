import json
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from depth_http_preview import DepthPreview, ProjectionConflict

try:
    from websockets.sync.client import connect
except ImportError:
    connect = None


class ProjectionPreviewTests(unittest.TestCase):
    def start_preview(self, **kwargs):
        preview = DepthPreview(host="127.0.0.1", port=0, **kwargs)
        self.addCleanup(preview.close)
        self.base = f"http://127.0.0.1:{preview.port}"
        return preview

    def fetch(self, path, **kwargs):
        return urlopen(Request(self.base + path, **kwargs), timeout=3)

    def wait_snapshot(self, minimum):
        end = time.perf_counter() + 3
        while time.perf_counter() < end:
            with self.fetch("/api/v1/state") as response:
                snapshot = json.load(response)["snapshot"]
            if snapshot and snapshot["id"] >= minimum:
                return snapshot
            time.sleep(.01)
        self.fail(f"Snapshot {minimum} was not encoded")

    def image(self, path):
        with self.fetch(path) as response:
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            return image, response.headers

    def test_legacy_physical_and_web_images_share_one_publication(self):
        preview = self.start_preview(projection_size=(48, 32), data_origin="https://example.test")
        with self.assertRaises(HTTPError) as waiting:
            self.fetch("/api/v1/depth.png")
        self.assertEqual(waiting.exception.code, 503)
        waiting.exception.close()

        projector = np.full((32, 48, 3), 90, np.uint8)
        web = np.zeros((12, 18, 4), np.uint8)
        web[2:10, 4:12] = [5, 14, 72, 103]
        legacy = np.full((9, 17, 3), 180, np.uint8)
        preview.update_projection(projector, web, legacy=legacy, metrics={"measured_fraction": .75})
        projector[:] = 0
        snapshot = self.wait_snapshot(1)
        self.assertTrue(snapshot["depth_ready"])
        physical_image, physical_headers = self.image("/api/v1/preview.jpg")
        depth_image, depth_headers = self.image("/api/v1/depth.png")
        legacy_image, _ = self.image("/frame.jpg")
        np.testing.assert_allclose(physical_image, 90, atol=1)
        np.testing.assert_array_equal(depth_image, web)
        np.testing.assert_allclose(legacy_image, legacy, atol=1)
        self.assertEqual(physical_image.shape, (32, 48, 3))
        self.assertEqual(legacy_image.shape, (9, 17, 3))
        for headers in (physical_headers, depth_headers):
            self.assertEqual(headers["X-Snapshot-Id"], str(snapshot["id"]))
            self.assertEqual(headers["X-Projection-Revision"], snapshot["projection_revision"])
        with self.fetch("/api/v1/state", headers={"Origin": "https://example.test"}) as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "https://example.test")
            self.assertEqual(json.load(response)["metrics"]["measured_fraction"], .75)

        preview.update_projection(np.full((32, 48, 3), 117, np.uint8), None, legacy=legacy)
        self.assertFalse(self.wait_snapshot(2)["depth_ready"])
        physical_image, _ = self.image("/api/v1/preview.jpg")
        np.testing.assert_allclose(physical_image, 117, atol=1)

    def test_busy_encoder_replaces_pending_frames(self):
        preview = self.start_preview()
        physical = np.zeros((8, 8, 3), np.uint8)
        web = np.zeros((4, 4, 4), np.uint8)
        encoding = threading.Event()
        release = threading.Event()
        actual = preview._encode

        def wait_for_release(*args):
            if not encoding.is_set():
                encoding.set()
                release.wait(3)
            return actual(*args)

        with patch.object(preview, "_encode", wait_for_release):
            try:
                preview.update_projection(physical, web)
                self.assertTrue(encoding.wait(2))
                preview.update_projection(physical, web)
                web[:] = [1, 2, 3, 4]
                preview.update_projection(physical, web)
            finally:
                release.set()
            self.wait_snapshot(3)
        image, headers = self.image("/api/v1/depth.png")
        self.assertEqual(headers["X-Snapshot-Id"], "3")
        np.testing.assert_array_equal(image, web)
        self.assertEqual(preview.update_count, 2)

    def test_projection_setting_callbacks_and_browser_origin(self):
        settings = {"mode": "balanced", "revision": 1}

        def apply(request):
            if request.get("expected_revision") != settings["revision"]:
                raise ProjectionConflict("revision changed")
            settings.update(mode=request["mode"], revision=2)
            return dict(settings)

        self.start_preview(projection_state=settings.copy, on_projection_settings=apply)
        path = "/api/v1/projection"
        with self.fetch(path) as response:
            self.assertEqual(json.load(response), settings)
        payload = json.dumps({"mode": "raw", "expected_revision": 1}).encode()
        for origin, code in (("https://elsewhere.test", 403), (self.base, 200), (self.base, 409)):
            headers = {"Origin": origin, "Content-Type": "application/json"}
            if code == 200:
                with self.fetch(path, data=payload, headers=headers) as response:
                    self.assertEqual(json.load(response), {"mode": "raw", "revision": 2})
            else:
                with self.assertRaises(HTTPError) as rejected:
                    self.fetch(path, data=payload, headers=headers)
                self.assertEqual(rejected.exception.code, code)
                rejected.exception.close()

    def test_legacy_settings_revision_tracks_the_rendered_frame(self):
        applied = []
        preview = self.start_preview(projector_ready=True, on_settings=applied.append)
        rendered_revision, settings = preview.settings_snapshot()
        settings["roi"]["vertices"][0][0] = .125
        self.assertNotEqual(settings, preview.settings_snapshot()[1])
        settings.update(expected_settings_revision=rendered_revision,
                        projection_revision=preview.projection_revision, palette_gamma=.8)
        request = {"data": json.dumps(settings).encode(), "headers": {"Content-Type": "application/json"}}
        with self.fetch("/api/v1/settings", **request) as response:
            accepted = json.load(response)
        with self.assertRaises(HTTPError) as stale:
            self.fetch("/api/v1/settings", **request)
        self.assertEqual(stale.exception.code, 409)
        stale.exception.close()
        current_revision, current_settings = preview.settings_snapshot()
        self.assertEqual(current_revision, accepted["settings_revision"])
        self.assertEqual(applied, [current_settings])

        projector = np.zeros((8, 8, 3), np.uint8)
        web = np.zeros((4, 4, 4), np.uint8)
        preview.update_projection(projector, web, settings_revision=rendered_revision)
        self.assertEqual(self.wait_snapshot(1)["settings_revision"], rendered_revision)
        for endpoint in ("/api/v1/preview.jpg", "/api/v1/depth.png"):
            with self.fetch(endpoint) as response:
                self.assertEqual(response.headers["X-Settings-Revision"], str(rendered_revision))
        preview.update_projection(projector, web, settings_revision=current_revision)
        self.assertEqual(self.wait_snapshot(2)["settings_revision"], current_revision)

    def test_extended_marker_retains_metadata_during_camera_loss_and_expiry(self):
        cameras = [{"index": 1, "ready": True, "fps": 30}, {"index": 0, "ready": True, "fps": 30}]
        preview = self.start_preview(marker_camera_status=lambda: cameras)
        marker = {"id": 2, "x": 30, "y": 22, "angle": 0, "visible": True, "last_seen_unix_ms": 500}
        stamp = time.perf_counter() - 1
        preview.update_markers([marker], stamp, pair_skew_ms=4, capture_size=(1920, 1080))
        with self.fetch("/api/v1/markers") as response:
            state = json.load(response)
        self.assertEqual(state["markers"], [marker])
        self.assertEqual(state["tracking"]["capture_size"], [1920, 1080])
        self.assertEqual(state["tracking"]["pair_skew_ms"], 4)
        cameras[0]["ready"] = False
        with self.fetch("/api/v1/markers") as response:
            state = json.load(response)
        self.assertEqual(state["markers"], [{**marker, "visible": False}])
        cameras[0]["ready"] = True
        preview.update_markers([marker], stamp - 2, pair_skew_ms=4)
        with self.fetch("/api/v1/markers") as response:
            state = json.load(response)
        self.assertEqual(state["markers"], [{**marker, "visible": False}])
        self.assertIsNone(state["tracking"]["pair_skew_ms"])

    @unittest.skipIf(connect is None, "WebSocket dependency unavailable")
    def test_websocket_channels_advance_independently(self):
        preview = self.start_preview(stream_port=0)
        with self.fetch("/api/v1/state") as response:
            streams = json.load(response)["streams"]
        base = f"ws://127.0.0.1:{streams['port']}"
        with connect(base + streams["depth"], origin=self.base, close_timeout=1) as depth, \
                connect(base + streams["markers"], origin=self.base, close_timeout=1) as markers:
            self.assertEqual(json.loads(depth.recv(timeout=2))["snapshot_id"], 0)
            self.assertEqual(json.loads(markers.recv(timeout=2))["frame_id"], 0)
            marker = {"id": 0, "x": 5, "y": 5, "angle": 0, "visible": True}
            preview.update_markers([marker], time.perf_counter())
            self.assertEqual(json.loads(markers.recv(timeout=2))["markers"], [marker])
            with self.assertRaises(TimeoutError):
                depth.recv(timeout=.1)
            preview.update_projection(np.full((8, 8, 3), 1, np.uint8), np.full((4, 4, 4), 2, np.uint8))
            update = json.loads(depth.recv(timeout=2))
            self.assertEqual(update["snapshot_id"], 1)
            self.assertEqual(update["projection_revision"], preview.projection_revision)
