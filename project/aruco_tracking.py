"""Filter stereo poses and distinguish each observation from its last known pose."""

import math
import time

import cv2
import numpy as np


class ArucoTrackHistory:
    """Keep a bounded history keyed by marker ID, with monotonic sample timing."""

    def __init__(self, hold_seconds=2.6, smoothing_seconds=.7):
        self.hold_seconds = hold_seconds
        self.smoothing_seconds = smoothing_seconds
        self.wall_offset = time.time() - time.perf_counter()
        self.tracks = {}

    def update(self, detections, sample_stamp):
        active = {}
        for detection in detections:
            marker_id = detection["id"]
            previous = self.tracks.get(marker_id)
            centre = detection["center"]
            rotation = detection["rotation"]
            if previous is not None and 0 < sample_stamp - previous["stamp"] <= self.hold_seconds:
                fraction = 1 - math.exp(-(sample_stamp - previous["stamp"]) / self.smoothing_seconds)
                centre = previous["center"] + fraction * (centre - previous["center"])
                before = previous["rotation"]
                # Interpolate the shortest rotation in SO(3), including across ±π.
                turn, _ = cv2.Rodrigues(before.T @ rotation)
                adjustment, _ = cv2.Rodrigues(turn * fraction)
                rotation = before @ adjustment
            corners = detection["local_glow"] @ rotation.T + centre
            active[marker_id] = {**detection, "center": centre, "rotation": rotation,
                                 "camera_corners": corners, "stamp": sample_stamp,
                                 "visible": True,
                                 "last_seen_unix_ms": round((self.wall_offset + sample_stamp) * 1000)}
        for marker_id, previous in self.tracks.items():
            if marker_id not in active and sample_stamp - previous["stamp"] <= self.hold_seconds:
                active[marker_id] = {**previous, "visible": False}
        self.tracks = active
        return [active[marker_id] for marker_id in sorted(active)]
