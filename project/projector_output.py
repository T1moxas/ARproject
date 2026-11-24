"""Present the calibrated raster on an enumerated Windows display."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock
import ctypes
from ctypes import wintypes
import sys

import cv2

_CDS_TEST = 2
_CDS_FULLSCREEN = 4


class _DisplayDevice(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128),
    ]


class _DisplayMode(ctypes.Structure):
    # DEVMODEW with the display branch of the printer/display union.
    _fields_ = [
        ("name", wintypes.WCHAR * 32), ("spec_version", wintypes.WORD),
        ("driver_version", wintypes.WORD), ("size", wintypes.WORD),
        ("driver_extra", wintypes.WORD), ("fields", wintypes.DWORD),
        ("position", wintypes.POINT), ("orientation", wintypes.DWORD),
        ("fixed_output", wintypes.DWORD), ("color", wintypes.SHORT),
        ("duplex", wintypes.SHORT), ("y_resolution", wintypes.SHORT),
        ("tt_option", wintypes.SHORT), ("collate", wintypes.SHORT),
        ("form_name", wintypes.WCHAR * 32), ("log_pixels", wintypes.WORD),
        ("bits", wintypes.DWORD), ("width", wintypes.DWORD), ("height", wintypes.DWORD),
        ("flags", wintypes.DWORD), ("frequency", wintypes.DWORD),
        ("icm_method", wintypes.DWORD), ("icm_intent", wintypes.DWORD),
        ("media_type", wintypes.DWORD), ("dither_type", wintypes.DWORD),
        ("reserved1", wintypes.DWORD), ("reserved2", wintypes.DWORD),
        ("panning_width", wintypes.DWORD), ("panning_height", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class Display:
    id: str
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool
    modes: tuple


class WindowsDisplays:
    def __init__(self):
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.EnumDisplayDevicesW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_DisplayDevice), wintypes.DWORD]
        self.user32.EnumDisplayDevicesW.restype = wintypes.BOOL
        self.user32.EnumDisplaySettingsExW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_DisplayMode), wintypes.DWORD]
        self.user32.EnumDisplaySettingsExW.restype = wintypes.BOOL
        self.user32.ChangeDisplaySettingsExW.argtypes = [
            wintypes.LPCWSTR, ctypes.POINTER(_DisplayMode), wintypes.HWND, wintypes.DWORD, wintypes.LPVOID]
        self.user32.ChangeDisplaySettingsExW.restype = wintypes.LONG

    def mode(self, display, index=0xFFFFFFFF):
        mode = _DisplayMode()
        mode.size = ctypes.sizeof(mode)
        return mode if self.user32.EnumDisplaySettingsExW(display, index, ctypes.byref(mode), 0) else None

    def _modes(self, display):
        index = 0
        while (mode := self.mode(display, index)) is not None:
            index += 1
            if mode.bits >= 32 and mode.orientation <= 3 and not mode.flags & 2:
                yield mode

    def displays(self):
        result = []
        index = 0
        while True:
            adapter = _DisplayDevice()
            adapter.cb = ctypes.sizeof(adapter)
            if not self.user32.EnumDisplayDevicesW(None, index, ctypes.byref(adapter), 0):
                break
            index += 1
            if not adapter.StateFlags & 1 or adapter.StateFlags & 8:
                continue
            identifier = adapter.DeviceName
            current = self.mode(identifier)
            if current is None:
                continue
            monitor = _DisplayDevice()
            monitor.cb = ctypes.sizeof(monitor)
            self.user32.EnumDisplayDevicesW(identifier, 0, ctypes.byref(monitor), 0)
            name = f"{monitor.DeviceString or adapter.DeviceString} ({identifier.rsplit(chr(92), 1)[-1]})"
            modes = {(current.width, current.height)}
            for mode in self._modes(identifier):
                size = mode.width, mode.height
                if size not in modes and mode.orientation == current.orientation and self.test_mode(identifier, mode):
                    modes.add(size)
            result.append(Display(identifier, name, current.position.x, current.position.y,
                                  current.width, current.height, bool(adapter.StateFlags & 4),
                                  tuple(sorted(modes, key=lambda size: (size[0] * size[1], size)))))
        return result

    def test_mode(self, display, mode):
        return self.user32.ChangeDisplaySettingsExW(display, ctypes.byref(mode), None, _CDS_TEST, None) == 0

    def select_mode(self, display, size):
        current = self.mode(display)
        if current is None:
            raise ValueError("Дисплей отключён")
        candidates = [mode for mode in self._modes(display)
                      if (mode.width, mode.height) == size and mode.orientation == current.orientation]
        candidates.sort(key=lambda mode: (mode.frequency == current.frequency, mode.frequency), reverse=True)
        for mode in candidates:
            if self.test_mode(display, mode):
                if self.user32.ChangeDisplaySettingsExW(display, ctypes.byref(mode), None, _CDS_FULLSCREEN, None) == 0:
                    return current
        raise ValueError("Windows не приняла этот режим дисплея")

    def restore(self, display, previous):
        if self.user32.ChangeDisplaySettingsExW(display, ctypes.byref(previous), None, 0, None) != 0:
            raise RuntimeError(f"Не удалось восстановить режим {display}")


def compatible(size, reference):
    return size[0] * reference[1] == size[1] * reference[0]


def frame_at_size(frame, size):
    height, width = frame.shape[:2]
    if (width, height) == size:
        return frame
    if not compatible(size, (width, height)):
        raise ValueError("Projector raster aspect ratio differs from calibration")
    method = cv2.INTER_AREA if size[0] < width else cv2.INTER_LINEAR
    return cv2.resize(frame, size, interpolation=method)


class ProjectorOutput:
    def __init__(self, window, reference, x, y, fullscreen, display_id=None, size=None, backend=None):
        self.window = window
        self.reference = reference
        self.default_xy = (x, y)
        self.backend = backend if backend is not None else WindowsDisplays() if sys.platform == "win32" else None
        self.displays = self.backend.displays() if self.backend else []
        self.changed_mode = None
        self.settings = None
        self.presentation_error = None
        display = next((item for item in self.displays if item.x <= x < item.x + item.width
                        and item.y <= y < item.y + item.height), None)
        if display_id is not None:
            display = next((item for item in self.displays if item.id == display_id
                            or item.id.endswith(f"DISPLAY{display_id}")), None)
            if display is None:
                raise ValueError(f"Display {display_id} is not connected")
        elif display is None:
            display = next((item for item in self.displays if not item.primary), None)
            if display is None and self.displays:
                display = self.displays[0]
        identifier = display.id if display else None
        active = (display.width, display.height) if display else None
        if size is None:
            if fullscreen and active and compatible(active, reference):
                size = active
            else:
                size = reference
                if fullscreen:
                    fullscreen = False
                    self.presentation_error = ("Выберите полноэкранный режим с пропорциями калибровки"
                                               if display and any(compatible(mode, reference) for mode in display.modes)
                                               else "Windows не подтвердила полноэкранный режим с пропорциями калибровки")
        elif display is None or size not in display.modes:
            raise ValueError("Requested resolution is not advertised by the selected display")
        self.initial = {"display": identifier, "width": size[0], "height": size[1], "fullscreen": fullscreen}
        self.validate(self.initial)

    def caller_displays(self):
        return [{"id": item.id, "name": item.name, "width": item.width, "height": item.height,
                 "primary": item.primary,
                 "modes": [{"width": width, "height": height} for width, height in item.modes
                           if compatible((width, height), self.reference)]} for item in self.displays]

    def validate(self, settings):
        identifier = settings["display"]
        size = (settings["width"], settings["height"])
        fullscreen = settings["fullscreen"]
        if any(type(value) is not int or value < 2 for value in size) or type(fullscreen) is not bool:
            raise ValueError("Некорректное разрешение или признак полноэкранного режима")
        display = next((item for item in self.displays if item.id == identifier), None)
        if display is None and (identifier is not None or self.displays):
            raise ValueError("Выбранный дисплей недоступен")
        if not compatible(size, self.reference):
            raise ValueError("Пропорции разрешения должны совпадать с калибровкой")
        if (fullscreen or size != self.reference) and (display is None or size not in display.modes):
            raise ValueError("Дисплей не поддерживает это разрешение")
        return display

    def configure(self, settings):
        display = self.validate(settings)
        old = self.settings
        requested = {key: settings[key] for key in self.initial}
        if requested == old:
            return
        try:
            self._configure(requested, display)
        except (ValueError, RuntimeError, cv2.error) as error:
            if old is not None:
                self._configure(old, self.validate(old))
            raise ValueError(str(error)) from error
        self.settings = requested
        if old is not None:
            self.presentation_error = None

    def _configure(self, requested, display):
        if self.settings is not None:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        self.restore_mode()
        width, height = requested["width"], requested["height"]
        if display is None:
            x, y = self.default_xy
        else:
            current = self.backend.mode(display.id)
            if current is None:
                raise ValueError("Дисплей отключён")
            if requested["fullscreen"] and (width, height) != (current.width, current.height):
                original = self.backend.select_mode(display.id, (width, height))
                self.changed_mode = display.id, original
                current = self.backend.mode(display.id)
                if current is None or (current.width, current.height) != (width, height):
                    raise ValueError("Дисплей не переключился в запрошенный режим")
            x, y = current.position.x, current.position.y
            if not requested["fullscreen"]:
                scale = min(1, max(1, current.width - 48) / width, max(1, current.height - 80) / height)
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
                x, y = x + 15, y + 15
        cv2.resizeWindow(self.window, width, height)
        cv2.moveWindow(self.window, x, y)
        cv2.waitKey(1)
        if requested["fullscreen"]:
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def restore_mode(self):
        if self.changed_mode is not None:
            self.backend.restore(*self.changed_mode)
            self.changed_mode = None

    def present(self, reference_frame):
        size = self.settings["width"], self.settings["height"]
        output = frame_at_size(reference_frame, size)
        cv2.imshow(self.window, output)
        return output


class ProjectionControl:
    """Drive native window and appearance changes on the display thread."""

    def __init__(self, output, appearance, fill, appearances, noise_mm):
        self.output = output
        self._lock = Lock()
        self._queue = Queue(maxsize=1)
        self._closed = False
        self._state = {
            "controllable": True, "revision": 0,
            "settings": dict(output.initial, appearance=appearance, fill=fill),
            "modes": {"appearance": list(appearances), "fill": ["conservative", "surface"]},
            "displays": output.caller_displays(),
            "calibration": {"width": output.reference[0], "height": output.reference[1], "noise_mm": noise_mm},
            "presentation_error": output.presentation_error,
        }

    def state(self):
        with self._lock:
            return deepcopy(self._state)

    def request(self, payload):
        future = Future()
        with self._lock:
            if self._closed:
                raise TimeoutError("Служба проекции остановлена")
            try:
                self._queue.put_nowait((payload, future))
            except Full as error:
                raise TimeoutError("Применяются предыдущие настройки") from error
        try:
            return future.result(timeout=10)
        except TimeoutError:
            future.cancel()
            raise

    def service(self, apply):
        try:
            payload, future = self._queue.get_nowait()
        except Empty:
            return
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(self.change(payload, apply))
        except Exception as error:
            future.set_exception(error)

    def change(self, payload, apply):
        from depth_http_preview import ProjectionConflict

        current = self.state()
        if not isinstance(payload, dict) or type(payload.get("expected_revision")) is not int:
            raise ValueError("Укажите ревизию настроек")
        if payload["expected_revision"] != current["revision"]:
            raise ProjectionConflict("Настройки изменились. Обновите состояние проектора")
        settings = {key: payload.get(key) for key in current["settings"]}
        if any(settings[key] not in values for key, values in current["modes"].items()):
            raise ValueError("Неизвестный режим проекции или фильтр")
        if settings == current["settings"]:
            return current
        self.output.validate(settings)
        apply(settings, current["settings"])
        current["settings"] = settings
        current["revision"] += 1
        current["presentation_error"] = self.output.presentation_error
        with self._lock:
            self._state = current
        return deepcopy(current)

    def cycle(self, key, apply):
        current = self.state()
        settings = current["settings"]
        choices = current["modes"][key]
        settings[key] = choices[(choices.index(settings[key]) + 1) % len(choices)]
        self.change(dict(settings, expected_revision=current["revision"]), apply)

    def close(self):
        with self._lock:
            self._closed = True
            try:
                _, future = self._queue.get_nowait()
            except Empty:
                return
            if future.set_running_or_notify_cancel():
                future.set_exception(TimeoutError("Служба проекции остановлена"))
