from types import SimpleNamespace
import math
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from aruco_tracking import ArucoTrackHistory
from stereo_capture import common_modes, open_stereo_cameras
from stereo_live import ArucoRobotTracker
from surface_geometry import ProjectorModel


class CaptureNegotiationTests(unittest.TestCase):
    size = (1280, 720)
    mjpg = int.from_bytes(b"MJPG", "little")

    def mode(self, size, crop=None, fps=30):
        return size, crop, fps, self.mjpg

    def test_uses_only_common_calibration_compatible_modes(self):
        capabilities = {
            1: [self.mode(self.size), self.mode((3840, 2160)), self.mode((1920, 1080)),
                self.mode((1600, 900), crop=(12, 0, 1600, 900)), self.mode((4096, 3072))],
            0: [self.mode(self.size), self.mode((1920, 1080), fps=20),
                self.mode((1600, 900)), self.mode((4096, 3072))],
        }
        self.assertEqual(common_modes(capabilities, [1, 0], self.size),
                         [((1920, 1080), (self.mjpg, self.mjpg))])

    def test_failed_larger_pair_is_closed_and_second_pair_is_reused(self):
        capabilities = {
            i: [self.mode(size) for size in (self.size, (1920, 1080), (2560, 1440))]
            for i in (1, 0)
        }
        instances = []

        class Camera:
            def __init__(self, index, focus, **options):
                self.index = index
                self.size = options.get("capture_size")
                self.stop = threading.Event()
                self.thread = SimpleNamespace(is_alive=lambda: False)
                self.closed = False
                instances.append(self)

            def snapshot(self):
                if self.index == 0 and self.size == (2560, 1440):
                    return []
                stamp = time.perf_counter()
                frame = SimpleNamespace(image=SimpleNamespace(shape=self.size[::-1] + (3,)),
                                        timestamp=stamp)
                return [frame, frame]

            def close(self):
                self.closed = True

        with patch("stereo_capture.sys.platform", "win32"), \
                patch("stereo_capture._enumerate_with_deadline", return_value=(capabilities, None)):
            cameras, profile = open_stereo_cameras(Camera, (1, 0), (0, 0), self.size, probe_timeout=.035)
        self.assertEqual(profile["size"], [1920, 1080])
        self.assertTrue(profile["verified"])
        self.assertTrue(all(camera.closed for camera in instances[:2]))
        self.assertEqual(cameras, instances[2:])
        self.assertFalse(any(camera.closed for camera in cameras))


class StereoArucoTests(unittest.TestCase):
    def test_detects_all_six_ids_in_1080p_and_rejects_single_camera(self):
        tracker = ArucoRobotTracker()
        intrinsic = np.array([[1350., 0., 960.], [0., 1350., 540.], [0., 0., 1.]])
        projections = (np.column_stack((intrinsic, [0., 0., 0.])),
                       np.column_stack((intrinsic, [-162., 0., 0.])))
        left, right = np.full((1080, 1920), 255, np.uint8), np.full((1080, 1920), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for marker_id in range(6):
            top, origin = 250 + 490 * (marker_id // 3), 440 + 480 * (marker_id % 3)
            image = np.rot90(cv2.aruco.generateImageMarker(dictionary, marker_id, 80), k=marker_id % 4)
            left[top:top + 80, origin:origin + 80] = image
            right[top:top + 80, origin - 192:origin - 112] = image
        found, _, _ = tracker.detect_pair(left, right, intrinsic, projections)
        self.assertEqual([marker["id"] for marker in found], list(range(6)))
        self.assertTrue(all(marker["quality"]["reprojection_rms_px"] < .2 for marker in found))
        for marker in found:
            self.assertAlmostEqual(marker["center"][2], 162 / 192, delta=.002)
            self.assertAlmostEqual(np.linalg.det(marker["rotation"]), 1.0)
        projection = ProjectorModel(np.column_stack(([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]],
                                                     [0., 0., 0.])), (1280, 800), "")
        previews = tracker.projected_markers(found, projection, (0, 0, 1280, 800))
        self.assertAlmostEqual(previews[0]["x"], (440 + 39.5) * (2 / 3), delta=.2)
        self.assertEqual(previews[0]["coordinate_space"], "rectified_left")
        lost, _, _ = tracker.detect_pair(left, np.full_like(right, 255), intrinsic, projections)
        self.assertEqual(lost, [])

    def test_stereo_size_and_vertical_residual_reject_inconsistent_pairs(self):
        tracker = ArucoRobotTracker()
        intrinsic = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
        projections = (np.column_stack((intrinsic, [0., 0., 0.])),
                       np.column_stack((intrinsic, [-108., 0., 0.])))
        a = np.array([[620., 320.], [680., 320.], [680., 380.], [620., 380.]])
        self.assertIsNone(tracker._stereo_pose(0, a, a - [72., 0.], projections, 1))
        self.assertIsNone(tracker._stereo_pose(0, a, a - [144., 4.], projections, 1))
        self.assertIsNotNone(tracker._stereo_pose(0, a, a - [144., 0.], projections, 1))

    def test_history_flags_one_missed_frame_and_filters_shortest_turn(self):
        tracker = ArucoRobotTracker()
        stamp = time.perf_counter()
        history = ArucoTrackHistory()

        def detection(angle, offset=0):
            rvec = np.array([0., 0., math.radians(angle)])
            rotation, _ = cv2.Rodrigues(rvec)
            return {"id": 2, "center": np.array([offset, 0., 1.]), "rotation": rotation,
                    "local_glow": tracker.glow_corners}

        first, = history.update([detection(177)], stamp)
        missing, = history.update([], stamp + 1)
        self.assertFalse(missing["visible"])
        self.assertEqual(missing["last_seen_unix_ms"], first["last_seen_unix_ms"])
        latest, = history.update([detection(-177, offset=.1)], stamp + 2)
        self.assertTrue(latest["visible"])
        self.assertAlmostEqual(latest["last_seen_unix_ms"] - first["last_seen_unix_ms"], 2000, delta=1)
        self.assertGreater(latest["center"][0], .09)
        angle = math.degrees(math.atan2(latest["rotation"][1, 0], latest["rotation"][0, 0]))
        self.assertGreater(abs(angle), 176)
        self.assertEqual(history.update([], stamp + 5), [])
