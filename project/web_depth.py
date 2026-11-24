"""Muted surface layer in the projector coordinate system of the robots."""

import cv2
import numpy as np


def surface_layer(display, height_renderer, far, near):
    depth = display.depth
    valid = np.isfinite(depth) & (depth > 0)
    if height_renderer is not None:
        heights = height_renderer.floor.heights(depth, height_renderer.intrinsic)
        noise = height_renderer.noise
        minimum = height_renderer.minimum
        allowed = valid & (heights >= (minimum if minimum else -noise))
        obstacles = allowed & (heights > height_renderer.maximum + 1e-6)
        baseline = max(minimum, noise)
        relief = (heights - baseline) / (height_renderer.maximum - baseline)
    else:
        allowed = valid
        obstacles = np.zeros(depth.shape, bool)
        relief = (far - depth) / (far - near)

    bands = np.zeros(depth.shape, np.uint8)
    bands[allowed] = np.floor(np.clip(relief[allowed], 0, 1) * 10).astype(np.uint8)
    layer = np.zeros((*depth.shape, 4), np.uint8)
    # BGRA. Only the measured obstacles have a neutral gray fill.
    layer[allowed, :3] = np.stack((72 + bands[allowed] * 5, 60 + bands[allowed] * 4,
                                   47 + bands[allowed] * 4), axis=1)
    layer[allowed, 3] = np.where(display.measured[allowed], 165, 105)
    layer[obstacles] = (121, 115, 110, 197)
    return layer


def compact_size(width, height):
    scale = min(1, 480 / width, 320 / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def compact_layer(projected):
    height, width = projected.shape[:2]
    size = compact_size(width, height)
    if size == (width, height):
        return projected
    # Nearest keeps valid and unknown cells distinct at obstacle boundaries.
    return cv2.resize(projected, size, interpolation=cv2.INTER_NEAREST)
