"""Latest-only depth and marker publication with the legacy LAN API."""
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import mimetypes
import threading
import socket
import json
import time
import cv2
import numpy as np

SITE_ORIGIN = "https://tfhguk.github.io"
SWARM_BUILD = Path(__file__).resolve().parents[1] / "web" / "dist"
SWARM_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
}
DATA_READ_ROUTES = {"/api/v1/state", "/api/v1/markers", "/api/v1/depth.png", "/api/v1/preview.jpg"}


class ProjectionConflict(ValueError):
    """The projection settings changed before the requested update."""


@dataclass(frozen=True)
class _ProjectionFrame:
    physical: np.ndarray
    web: np.ndarray | None
    legacy: np.ndarray | None
    snapshot_id: int
    submitted: float
    settings_revision: int
    projection_revision: str
    metrics: dict


def _default_settings(rect, width, height, height_max=0.5, gamma=0.5, interval=1.0):
    l, t, r, b = rect
    return {
        "schema_version": 1,
        "height_range_m": [0.0, float(height_max)],
        "above_range": "black",
        "palette_gamma": float(gamma),
        "update_interval_s": float(interval),
        "roi": {
            "coordinate_space": "projector_normalized",
            "vertices": [[l / width, t / height], [r / width, t / height],
                         [r / width, b / height], [l / width, b / height]],
        },
    }


class DepthPreview:
    def __init__(self, host="0.0.0.0", port=8765, calibrated=False,
                 projection_size=(1280, 800), available_rect=None,
                 projector_ready=False, stage_hint=None, initial_settings=None,
                 on_settings=None, on_action=None, marker_origin=None,
                 marker_camera_status=None, max_pair_skew_ms=None,
                 stream_port=None, projection_state=None, on_projection_settings=None,
                 data_origin=None, marker_capture_profile=None):
        self._lock = threading.RLock()
        self._settings_write_lock = threading.Lock()
        self._frame_job = threading.Condition(self._lock)
        self._pending_frame = None
        self._closing = False
        self._sequence = 0
        self._jpeg = None
        self._legacy_jpeg = None
        self._depth_png = None
        self._metrics = {}
        self._snapshot_projection_revision = None
        self._marker_frame_id = 0
        self._marker_stamp = None
        self._marker_time_ms = None
        self._markers = []
        self._marker_pair_skew_ms = None
        self._marker_capture_size = None
        self._marker_capture_profile = marker_capture_profile
        self._extended_markers = marker_capture_profile is not None
        self._marker_camera_status = marker_camera_status
        self._max_pair_skew_ms = max_pair_skew_ms
        self._streams = None
        self._stream_error = None
        self.calibrated = bool(calibrated)
        self.projector_ready = bool(projector_ready)
        self.stage_hint = stage_hint
        self.width, self.height = map(int, projection_size)
        self.available_rect = tuple(available_rect or (0, 0, self.width, self.height))
        self.projection_revision = "local-" + str(time.time_ns())
        self.settings_revision = 0
        self.snapshot_id = 0
        self.snapshot_time = 0.0
        self.snapshot_settings_revision = 0
        self.update_count = 0
        self.settings = initial_settings or _default_settings(
            self.available_rect, self.width, self.height)
        self.on_settings = on_settings
        self.on_action = on_action
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _route(self):
                return self.path.partition("?")[0]

            def _cors(self):
                origin = self.headers.get("Origin")
                route = self._route()
                markers = route == "/api/v1/markers"
                data_read = route in DATA_READ_ROUTES
                if origin and route != "/api/v1/projection" and (
                    origin == SITE_ORIGIN or (markers and origin == marker_origin)
                    or (data_read and origin == data_origin)
                ):
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, OPTIONS" if data_read else "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Expose-Headers",
                                 "X-Snapshot-Id, X-Projection-Revision, X-Settings-Revision")

            def _send(self, code, body, content_type="application/json", extra=None):
                try:
                    self.send_response(code)
                    self._cors()
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    if extra:
                        for key, value in extra.items():
                            self.send_header(key, str(value))
                    self.end_headers()
                    self.wfile.write(body)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    # Browser closed/reloaded. This is a normal disconnect.
                    pass

            def _json(self, code, data):
                self._send(code, json.dumps(data, separators=(",", ":")).encode("utf-8"))

            def _projection_error(self, status, message):
                self._json(status, {"error": {"message": message, "field": "projection"}})

            def _request_body(self, *, allow_empty=False):
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0 and allow_empty:
                    return {}
                if not 0 < length <= 64 * 1024:
                    raise ValueError("Некорректный размер запроса")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("Ожидается JSON-объект")
                return request

            def _serve_frame(self, depth):
                with owner._lock:
                    body = owner._depth_png if depth else owner._jpeg
                    headers = {
                        "X-Snapshot-Id": owner.snapshot_id,
                        "X-Projection-Revision": owner._snapshot_projection_revision,
                        "X-Settings-Revision": owner.snapshot_settings_revision,
                    }
                if body is None:
                    self._json(503, {"error": {"message": "Кадр ещё не готов"}})
                else:
                    self._send(200, body, "image/png" if depth else "image/jpeg", headers)

            def do_OPTIONS(self):
                try:
                    self.send_response(204)
                    self._cors()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except OSError:
                    pass

            def do_GET(self):
                try:
                    route = self._route()
                    if route == "/swarm":
                        self._send(308, b"", "text/plain", {"Location": "/swarm/"})
                    elif route.startswith("/swarm/"):
                        self._swarm_asset(route)
                    elif route == "/api/v1/markers":
                        self._json(200, owner._marker_state())
                    elif route == "/":
                        self._send(200, owner.PAGE, "text/html; charset=utf-8")
                    elif self.path.startswith("/status"):
                        with owner._lock:
                            ready = owner._legacy_jpeg is not None
                            data = {"calibrated": owner.calibrated, "frame_ready": ready}
                        self._json(200, data)
                    elif self.path.startswith("/frame.jpg"):
                        with owner._lock:
                            body = owner._legacy_jpeg
                        if body is None:
                            self._send(503, b"Frame not ready", "text/plain")
                        else:
                            self._send(200, body, "image/jpeg")
                    elif route == "/api/v1/state":
                        with owner._lock:
                            data = owner._state_locked()
                        self._json(200, data)
                    elif route in ("/api/v1/preview.jpg", "/api/v1/depth.png"):
                        self._serve_frame(route == "/api/v1/depth.png")
                    elif route == "/api/v1/projection":
                        if projection_state is None:
                            self._projection_error(501, "Настройки проектора недоступны в этом режиме")
                        else:
                            self._json(200, projection_state())
                    elif self.path.startswith("/api/v1/actions/"):
                        self._json(404, {"error":{"message":"Action endpoint not found","field":"action"}})
                    else:
                        self._send(404, b"Not found", "text/plain")
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    pass

            def _swarm_asset(self, route):
                try:
                    filename = unquote(route.removeprefix("/swarm/"), errors="strict") or "index.html"
                    relative = Path(filename)
                    if relative.anchor or ".." in relative.parts:
                        raise FileNotFoundError
                    root = SWARM_BUILD.resolve()
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise FileNotFoundError
                    body = path.read_bytes()
                except (ValueError, OSError, RuntimeError):
                    self._send(404, b"Not found", "text/plain")
                    return
                content_type = SWARM_MIME_TYPES.get(path.suffix) or mimetypes.guess_type(path)[0]
                self._send(200, body, content_type or "application/octet-stream")

            def _update_projection_settings(self):
                try:
                    # Read the bounded body before rejecting the request. Closing
                    # a Windows socket with unread bytes can discard the reply.
                    request = self._request_body()
                    request_host = self.headers.get("Host")
                    if not request_host or self.headers.get("Origin") != f"http://{request_host.lower()}":
                        self._projection_error(403, "Измените настройки со страницы локальной службы AR")
                        return
                    if on_projection_settings is None:
                        self._projection_error(501, "Настройки проектора недоступны в этом режиме")
                        return
                    if self.headers.get_content_type() != "application/json":
                        self._projection_error(415, "Ожидается application/json")
                        return
                    result = on_projection_settings(request)
                    self._json(200, result)
                except ProjectionConflict as error:
                    self._projection_error(409, str(error))
                except (ValueError, TypeError, KeyError) as error:
                    self._projection_error(400, str(error))
                except (TimeoutError, OSError) as error:
                    self._projection_error(503, str(error))
                except Exception as error:
                    self._projection_error(500, str(error))

            def do_POST(self):
                if self._route() == "/api/v1/projection":
                    self._update_projection_settings()
                    return
                if self.path.startswith("/api/v1/actions/"):
                    action = self.path.rsplit("/", 1)[-1]
                    if action == "calibrate-floor":
                        self._json(501, {"error":{"message":"Калибровку пола запускайте в локальном окне AR, не через веб-сайт","field":"action"}})
                        return
                    if action != "rescan":
                        self._json(404, {"error":{"message":"Unknown action","field":"action"}})
                        return
                    try:
                        request = self._request_body(allow_empty=True)
                        if request.get("projection_revision") != owner.projection_revision:
                            self._json(409, {"error":{"message":"Калибровка изменилась","field":"projection_revision"}})
                            return
                        if owner.on_action is None:
                            self._json(501, {"error":{"message":"Повторное сканирование недоступно"}})
                            return
                        owner.on_action("rescan")
                        self._json(200, {"request_id":request.get("request_id"), "accepted":True,
                                         "action_id":"rescan-"+str(time.time_ns())})
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                        pass
                    except (ValueError, TypeError) as error:
                        self._json(400, {"error":{"message":str(error)}})
                    except Exception as error:
                        self._json(500, {"error":{"message":str(error)}})
                    return
                if self.path != "/api/v1/settings":
                    self._json(404, {"error":{"message":"Endpoint not implemented","field":"action"}})
                    return
                try:
                    request = self._request_body()
                    with owner._settings_write_lock:
                        with owner._lock:
                            expected = owner.settings_revision
                            if request.get("projection_revision") != owner.projection_revision:
                                self._json(409, {"error":{"message":"Калибровка проектора изменилась","field":"projection_revision"}})
                                return
                            if request.get("expected_settings_revision") != expected:
                                self._json(409, {"error":{"message":"Настройки изменились; обновите их перед применением","field":"expected_settings_revision"}})
                                return
                            if not owner.projector_ready:
                                self._json(409, {"error":{"message":"3D-калибровка проектора ещё не подтверждена","field":"projection"}})
                                return
                            validated = owner._validate_settings(request)
                            callback = owner.on_settings
                        if callback is None:
                            self._json(501, {"error":{"message":"Изменение диапазона и зоны недоступно в этом режиме"}})
                            return
                        callback(validated)
                        with owner._lock:
                            owner.settings = validated
                            owner.settings_revision += 1
                            # Defaults may already be committed while the projector
                            # is still presenting a frame from the preceding revision.
                            reply = {"request_id": request.get("request_id"),
                                     "settings_revision": owner.settings_revision,
                                     "snapshot_id": owner.snapshot_id}
                    self._json(200, reply)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    pass
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    self._json(400, {"error":{"message":str(error),"field":"settings"}})
                except Exception as error:
                    self._json(500, {"error":{"message":"Не удалось применить настройки: "+str(error)}})

            def log_message(self, *_):
                pass

        # The simple standalone viewer page remains available at /.
        self.PAGE = ('''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth map preview</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:1rem}img{display:block;max-width:100%;height:auto;background:#222}#status{margin:.5rem 0;color:#aaa}</style>
<h2>Карта глубин / выход проектора</h2><div id="status">Подключение…</div><img id="map" alt="Карта глубин">
<script>const img=document.querySelector('#map'),s=document.querySelector('#status');let busy=false;async function tick(){if(busy)return;busy=true;try{const r=await fetch('/status',{cache:'no-store'});const j=await r.json();s.textContent=j.calibrated?'Поток активен · метрическая глубина':'Поток активен · диагностическая disparity';if(j.frame_ready)img.src='/frame.jpg?t='+Date.now()}catch(e){s.textContent='Нет связи'}finally{busy=false}}setInterval(tick,500);tick();</script></html>''').encode("utf-8")

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        addresses = self._lan_ipv4_addresses()
        if stream_port is not None:
            try:
                from preview_streams import PreviewStreams
                hosts = {"localhost", "127.0.0.1", socket.gethostname().lower(), *addresses}
                if host != "0.0.0.0":
                    hosts.add(host)
                origins = {None, SITE_ORIGIN, *(f"http://{name}:{self.port}" for name in hosts)}
                origins.update(value for value in (data_origin, marker_origin) if value)
                self._streams = PreviewStreams(host, stream_port, origins, self._depth_state, self._marker_state)
                print(f"Depth and marker WebSocket: ws://localhost:{self._streams.port}/ws/", flush=True)
            except (ImportError, OSError) as error:
                self._stream_error = f"WebSocket: {error}"
                print(f"{self._stream_error}. Browser polling remains available.", flush=True)
        self._encoder = threading.Thread(target=self._encode_projector_frames, name="ar-frame-encoder", daemon=True)
        self._encoder.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"Depth preview local: http://localhost:{self.port}/", flush=True)
        print(f"AR Swarm local: http://localhost:{self.port}/swarm/", flush=True)
        for ip in addresses:
            print(f"Depth preview LAN: http://{ip}:{self.port}/", flush=True)
        if not addresses:
            print(f"Depth preview LAN: use this PC IPv4 address at port {self.port}", flush=True)

    def _state_locked(self):
        stage = self.stage_hint or ("stereo_required" if not self.calibrated else
                 "floor_required" if not self.projector_ready else "ready")
        projection = {
            "width": self.width, "height": self.height,
            "available_rect": list(self.available_rect),
            "calibration_3d_accepted": self.projector_ready,
        }
        snapshot = None
        if self._jpeg is not None:
            snapshot = {
                "id": self.snapshot_id,
                "age_s": max(0.0, time.time() - self.snapshot_time),
                "projection_revision": self._snapshot_projection_revision,
                "settings_revision": self.snapshot_settings_revision,
                "depth_ready": self._depth_png is not None,
            }
        return {
            "schema_version": 1, "stage": stage,
            "projection_revision": self.projection_revision,
            "projection": projection,
            "settings_revision": self.settings_revision,
            "settings": self.settings,
            "snapshot": snapshot,
            "streams": None if self._streams is None else {
                "port": self._streams.port, "depth": "/ws/depth", "markers": "/ws/markers",
            },
            "metrics": {"check_hz": 0.0, "update_count": self.update_count,
                        "measured_fraction": 0.0, "filled_fraction": 0.0, **self._metrics},
            "errors": [self._stream_error] if self._stream_error else [],
        }

    def _depth_state(self):
        with self._lock:
            return {
                "snapshot_id": self.snapshot_id if self._depth_png is not None else 0,
                "projection_revision": self._snapshot_projection_revision or self.projection_revision,
            }

    def settings_snapshot(self):
        """Read the committed legacy configuration and its revision atomically."""
        with self._lock:
            return self.settings_revision, deepcopy(self.settings)

    @staticmethod
    def _invisible_markers(markers):
        return [{**marker, "visible": False} for marker in markers if "visible" in marker]

    def _marker_state(self):
        with self._lock:
            age = None if self._marker_stamp is None else max(0.0, time.perf_counter() - self._marker_stamp)
            max_age = 2.5 if self._extended_markers else 0.75
            fresh = age is not None and age <= max_age
            skew_ms = self._marker_pair_skew_ms if fresh else None
            capture_size = self._marker_capture_size
            extended = self._extended_markers
            state = {
                "schema_version": 1,
                "frame_id": self._marker_frame_id,
                "age_s": age,
                "captured_at_unix_ms": self._marker_time_ms,
                "projection": {"width": self.width, "height": self.height,
                               "available_rect": list(self.available_rect)},
                "markers": self._markers if fresh else self._invisible_markers(self._markers),
            }
        if self._marker_camera_status is not None:
            cameras = self._marker_camera_status()
            if not all(camera["ready"] for camera in cameras):
                state["markers"] = self._invisible_markers(state["markers"])
                skew_ms = None
            state["tracking"] = {
                "cameras": cameras,
                "pair_skew_ms": skew_ms,
                "max_skew_ms": self._max_pair_skew_ms,
            }
            if capture_size is not None:
                state["tracking"]["capture_size"] = list(capture_size)
            if extended:
                state["tracking"]["sample_interval_s"] = 1.0
            if self._marker_capture_profile is not None:
                state["tracking"]["capture_profile"] = self._marker_capture_profile
        return state

    def update_markers(self, markers, stamp, *, pair_skew_ms=None, capture_size=None):
        captured = None
        if stamp is not None:
            age = max(0.0, time.perf_counter() - stamp)
            captured = round((time.time() - age) * 1000)
        with self._lock:
            self._marker_frame_id += 1
            self._marker_stamp = stamp
            self._marker_time_ms = captured
            self._markers = [dict(marker) for marker in markers]
            self._marker_pair_skew_ms = pair_skew_ms
            if capture_size is not None:
                self._marker_capture_size = tuple(capture_size)
            self._extended_markers |= capture_size is not None or any("visible" in marker for marker in markers)

    def _validate_settings(self, request):
        if request.get("schema_version") != 1:
            raise ValueError("Unsupported settings schema")
        h = request.get("height_range_m")
        if not isinstance(h, list) or len(h) != 2:
            raise ValueError("height_range_m must have two values")
        lo, hi = map(float, h)
        gamma = float(request.get("palette_gamma"))
        interval = float(request.get("update_interval_s"))
        if not (0 <= lo < hi <= 0.5 and 0.1 <= gamma <= 3 and 0.5 <= interval <= 10):
            raise ValueError("Settings out of supported range")
        if request.get("above_range") != "black":
            raise ValueError("above_range must be black")
        roi = request.get("roi") or {}
        if roi.get("coordinate_space") != "projector_normalized":
            raise ValueError("ROI coordinate space is invalid")
        vertices = roi.get("vertices")
        if not isinstance(vertices, list) or not 3 <= len(vertices) <= 32:
            raise ValueError("ROI needs 3 to 32 vertices")
        l, t, r, b = self.available_rect
        for point in vertices:
            if (not isinstance(point, list) or len(point) != 2 or
                    not all(np.isfinite(float(v)) for v in point) or
                    not (0 <= float(point[0]) <= 1 and 0 <= float(point[1]) <= 1) or
                    not (l <= float(point[0]) * self.width <= r and
                         t <= float(point[1]) * self.height <= b)):
                raise ValueError("ROI vertices must stay inside the available projector area")
        return {
            "schema_version": 1, "height_range_m": [lo, hi],
            "above_range": "black", "palette_gamma": gamma,
            "update_interval_s": interval,
            "roi": {"coordinate_space":"projector_normalized", "vertices":vertices},
        }

    @staticmethod
    def _lan_ipv4_addresses():
        addresses = set()
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
                ip = item[4][0]
                if not ip.startswith("127."):
                    addresses.add(ip)
        except OSError:
            pass
        return sorted(addresses)

    @staticmethod
    def _encode(ext, image, parameters):
        try:
            ok, data = cv2.imencode(ext, image, parameters)
            return data.tobytes() if ok else None
        except cv2.error:
            return None

    def _publish(self, jpeg, depth, legacy, snapshot_id, submitted, settings_revision,
                 projection_revision, metrics):
        with self._lock:
            if self._closing or self.snapshot_id >= snapshot_id:
                return
            self._jpeg = jpeg
            self._depth_png = depth
            self._legacy_jpeg = legacy
            self._snapshot_projection_revision = projection_revision
            self.snapshot_id = snapshot_id
            self.snapshot_time = submitted
            self.snapshot_settings_revision = settings_revision
            self._metrics = metrics
            self.update_count += 1

    def _encode_projector_frames(self):
        while True:
            with self._frame_job:
                self._frame_job.wait_for(lambda: self._closing or self._pending_frame is not None)
                if self._closing:
                    return
                job, self._pending_frame = self._pending_frame, None
            jpeg = self._encode(".jpg", job.physical, [cv2.IMWRITE_JPEG_QUALITY, 90])
            depth = (self._encode(".png", job.web, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                     if job.web is not None else None)
            if jpeg is None or (depth is None and job.web is not None):
                continue
            legacy = (self._encode(".jpg", job.legacy, [cv2.IMWRITE_JPEG_QUALITY, 82])
                      if job.legacy is not None else jpeg)
            if legacy is not None:
                self._publish(jpeg, depth, legacy, job.snapshot_id, job.submitted,
                              job.settings_revision, job.projection_revision, job.metrics)

    def update_projection(self, physical_bgr, web_bgra, *, legacy=None, metrics=None, settings_revision=None):
        """Queue one matched projector/website snapshot and discard any pending predecessor."""
        if physical_bgr is None:
            return
        physical = physical_bgr.copy()
        web = web_bgra.copy() if web_bgra is not None else None
        legacy_copy = legacy.copy() if legacy is not None else None
        with self._frame_job:
            if self._closing:
                return
            revision = self.settings_revision if settings_revision is None else settings_revision
            if type(revision) is not int or not 0 <= revision <= self.settings_revision:
                raise ValueError("Unknown settings revision for rendered frame")
            self._sequence += 1
            self._pending_frame = _ProjectionFrame(
                physical, web, legacy_copy, self._sequence, time.time(), revision,
                self.projection_revision, dict(metrics or {}),
            )
            self._frame_job.notify()

    def update(self, image):
        """Publish an existing legacy camera-space preview."""
        if image is None:
            return
        with self._lock:
            self._sequence += 1
            snapshot_id = self._sequence
            settings_revision = self.settings_revision
            projection_revision = self.projection_revision
        submitted = time.time()
        jpeg = self._encode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if jpeg is not None:
            self._publish(jpeg, None, jpeg, snapshot_id, submitted, settings_revision,
                          projection_revision, {})

    def close(self):
        with self._frame_job:
            if self._closing:
                return
            self._closing = True
            self._pending_frame = None
            self._frame_job.notify_all()
        if self._streams is not None:
            self._streams.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self._encoder.join(timeout=2)
