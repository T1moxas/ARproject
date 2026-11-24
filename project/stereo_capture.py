"""Discover DirectShow modes and keep a single negotiated stream per camera."""

from __future__ import annotations

import ctypes as ct
import queue
import struct
import sys
import threading
import time
import uuid


class Guid(ct.Structure):
    _fields_ = [("data1", ct.c_uint32), ("data2", ct.c_uint16), ("data3", ct.c_uint16),
                ("data4", ct.c_ubyte * 8)]


class MediaType(ct.Structure):
    _fields_ = [("major", Guid), ("subtype", Guid),
                ("fixed_samples", ct.c_int32), ("temporal_compression", ct.c_int32),
                ("sample_size", ct.c_uint32), ("format_type", Guid), ("unknown", ct.c_void_p),
                ("format_size", ct.c_uint32), ("format_data", ct.c_void_p)]


def guid(value):
    return Guid.from_buffer_copy(uuid.UUID(value).bytes_le)


def _method(interface, slot, *types):
    table = ct.cast(interface, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *types)(table[slot])


def _release(interface):
    if interface:
        _method(interface, 2)(interface)


def _co_create(ole32, cls, iid):
    result = ct.c_void_p()
    code = ole32.CoCreateInstance(ct.byref(guid(cls)), None, 1,
                                   ct.byref(guid(iid)), ct.byref(result))
    if code < 0:
        raise OSError(f"DirectShow CoCreateInstance: 0x{code & 0xffffffff:08x}")
    return result


def _video_mode(media):
    format_type = bytes(media.format_type)
    bitmap_offsets = {
        bytes(guid("05589f80-c356-11ce-bf01-00aa0055595a")): 48,  # VIDEOINFOHEADER
        bytes(guid("f72a76a0-eb0a-11d0-ace4-0000c0cc16ba")): 72,  # VIDEOINFOHEADER2
    }
    offset = bitmap_offsets.get(format_type)
    if not media.format_data or offset is None or media.format_size < offset + 40:
        return None
    header = ct.string_at(media.format_data, offset + 40)
    width, height = struct.unpack_from("<ii", header, offset + 4)
    height = abs(height)
    if width <= 0 or height <= 0:
        return None
    source = struct.unpack_from("<iiii", header)
    # An empty source rectangle means the whole raster. Explicit crops that
    # differ from the calibration mode are not usable without new intrinsics.
    crop = None if source in ((0, 0, 0, 0), (0, 0, width, height)) else source
    interval, = struct.unpack_from("<q", header, 40)
    return (width, height), crop, (10_000_000 / interval if interval > 0 else 30.0), media.subtype.data1


def _pin_modes(ole32, pin):
    config = ct.c_void_p()
    iid = guid("c6e13340-30ac-11d0-a18c-00a0c9118956")  # IAMStreamConfig
    query = _method(pin, 0, ct.POINTER(Guid), ct.POINTER(ct.c_void_p))
    if query(pin, ct.byref(iid), ct.byref(config)) < 0:
        return []
    modes = []
    try:
        count, size = ct.c_int(), ct.c_int()
        capabilities = _method(config, 5, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))
        if capabilities(config, ct.byref(count), ct.byref(size)) < 0 or not 0 < size.value <= 4096:
            return modes
        get_caps = _method(config, 6, ct.c_int, ct.POINTER(ct.POINTER(MediaType)), ct.c_void_p)
        buffer = ct.create_string_buffer(size.value)
        for index in range(count.value):
            media = ct.POINTER(MediaType)()
            if get_caps(config, index, ct.byref(media), buffer) < 0 or not media:
                continue
            try:
                mode = _video_mode(media.contents)
                if mode is not None:
                    modes.append(mode)
            finally:
                ole32.CoTaskMemFree(media.contents.format_data)
                _release(media.contents.unknown)
                ole32.CoTaskMemFree(media)
        return modes
    finally:
        _release(config)


def _device_modes(ole32, moniker):
    bind = _method(moniker, 8, ct.c_void_p, ct.c_void_p, ct.POINTER(Guid), ct.POINTER(ct.c_void_p))
    iid = guid("56a86895-0ad4-11ce-b03a-0020af0ba770")  # IBaseFilter
    source = ct.c_void_p()
    if bind(moniker, None, None, ct.byref(iid), ct.byref(source)) < 0:
        return []
    modes = []
    pins = ct.c_void_p()
    try:
        if _method(source, 10, ct.POINTER(ct.c_void_p))(source, ct.byref(pins)) < 0:
            return modes
        next_pin = _method(pins, 3, ct.c_uint32, ct.POINTER(ct.c_void_p), ct.c_void_p)
        while True:
            pin = ct.c_void_p()
            if next_pin(pins, 1, ct.byref(pin), None) != 0:
                break
            try:
                modes.extend(_pin_modes(ole32, pin))
            finally:
                _release(pin)
        return modes
    finally:
        _release(pins)
        _release(source)


def _directshow_modes(indices):
    # OpenCV CAP_DSHOW enumerates the same video-input device category in this
    # order. The eventual mode is checked against actual simultaneous frames.
    ole32 = ct.windll.ole32
    ole32.CoInitializeEx.argtypes = [ct.c_void_p, ct.c_uint32]
    ole32.CoCreateInstance.argtypes = [ct.POINTER(Guid), ct.c_void_p, ct.c_uint32,
                                       ct.POINTER(Guid), ct.POINTER(ct.c_void_p)]
    ole32.CoTaskMemFree.argtypes = [ct.c_void_p]
    ole32.CoTaskMemFree.restype = None
    if ole32.CoInitializeEx(None, 2) < 0:
        raise OSError("Could not initialize DirectShow COM")
    enumerator = monikers = None
    try:
        enumerator = _co_create(ole32, "62be5d10-60eb-11d0-bd3b-00a0c911ce86",
                                "29840822-5b84-11d0-bd3b-00a0c911ce86")
        category = guid("860bb310-5d01-11d0-bd3b-00a0c911ce86")
        monikers = ct.c_void_p()
        create = _method(enumerator, 3, ct.POINTER(Guid), ct.POINTER(ct.c_void_p), ct.c_uint32)
        if create(enumerator, ct.byref(category), ct.byref(monikers), 0) != 0 or not monikers:
            return {}
        next_device = _method(monikers, 3, ct.c_uint32, ct.POINTER(ct.c_void_p), ct.c_void_p)
        result = {}
        for index in range(max(indices) + 1):
            device = ct.c_void_p()
            if next_device(monikers, 1, ct.byref(device), None) != 0:
                break
            try:
                if index in indices:
                    result[index] = _device_modes(ole32, device)
            finally:
                _release(device)
        return result
    finally:
        _release(monikers)
        _release(enumerator)
        ole32.CoUninitialize()


def _enumerate_with_deadline(indices, timeout):
    result = queue.Queue(maxsize=1)

    def read_modes():
        try:
            result.put((_directshow_modes(indices), None))
        except (OSError, ValueError, AttributeError) as error:
            result.put(({}, str(error)))

    # Some camera drivers block when queried before the device is ready.
    threading.Thread(target=read_modes, daemon=True, name="capture-capabilities").start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        return {}, "DirectShow capability query timed out"


def common_modes(advertised, indices, base_size):
    """Common advertised rasters preserving aspect ratio and known source crop."""
    eligible = []
    base_width, base_height = base_size
    mjpg = int.from_bytes(b"MJPG", "little")
    for index in indices:
        modes = advertised.get(index, ())
        baseline_crops = {crop for size, crop, _fps, _codec in modes if size == base_size}
        rates = {}
        for size, crop, fps, codec in modes:
            width, height = size
            if (crop in baseline_crops and width > base_width
                    and width * base_height == height * base_width and fps >= 1):
                # Compressed modes reserve less bandwidth on shared USB.
                mode = (codec == mjpg, min(float(fps), 30.0), codec)
                rates[size] = max(rates.get(size, mode), mode)
        eligible.append(rates)
    if len(eligible) != 2:
        return []
    return [(size, (eligible[0][size][2], eligible[1][size][2])) for size in
            sorted(eligible[0].keys() & eligible[1].keys(),
                   key=lambda mode: mode[0] * mode[1], reverse=True)]


def _capture_ready(cameras, size, max_skew_s):
    snapshots = [camera.snapshot() for camera in cameras]
    return all(len(frames) >= 2 for frames in snapshots) and any(
        left.image.shape[:2] == size[::-1] and right.image.shape[:2] == size[::-1]
        and abs(left.timestamp - right.timestamp) <= max_skew_s
        and time.perf_counter() - min(left.timestamp, right.timestamp) < .5
        for left in snapshots[0] for right in snapshots[1])


def open_stereo_cameras(camera_type, indices, focus_values, base_size, *, max_skew_s=.05,
                        capability_timeout=2.5, probe_timeout=2.5):
    """Open a verified common high-resolution pair, or report the baseline fallback.

    Capability checks happen once before depth/ArUco start. A selected pair stays
    open and supplies both consumers; no resolution switches during sampling.
    """
    profile = {"size": list(base_size), "verified": False, "common_high_modes": []}

    def baseline(reason):
        profile["reason"] = reason
        return [camera_type(index, focus) for index, focus in zip(indices, focus_values)], profile

    if sys.platform != "win32":
        return baseline("Capability enumeration requires Windows DirectShow")
    advertised, error = _enumerate_with_deadline(indices, capability_timeout)
    if error:
        return baseline(error)
    if any(not advertised.get(index) for index in indices):
        return baseline("Mode capabilities unavailable for one or both cameras")
    common = common_modes(advertised, indices, base_size)
    profile["common_high_modes"] = [list(size) for size, _codecs in common]
    if not common:
        return baseline("No common larger calibrated-aspect DirectShow mode")
    overall_deadline = time.perf_counter() + 3 * probe_timeout
    for size, codecs in common:
        if time.perf_counter() >= overall_deadline:
            break
        cameras = []
        keep = False
        try:
            for index, focus, codec in zip(indices, focus_values, codecs):
                cameras.append(camera_type(index, focus, capture_size=size, fourcc=codec))
            deadline = min(overall_deadline, time.perf_counter() + probe_timeout)
            while time.perf_counter() < deadline:
                if _capture_ready(cameras, size, max_skew_s):
                    profile.update(size=list(size), verified=True, reason="Highest verified common mode")
                    keep = True
                    return cameras, profile
                time.sleep(.025)
        finally:
            if not keep:
                for camera in cameras:
                    camera.stop.set()
                for camera in cameras:
                    camera.close()
        if any(camera.thread.is_alive() for camera in cameras):
            return baseline("A driver did not release a failed high-resolution probe")
    return baseline("No advertised high-resolution mode delivered a synchronized pair during startup")
