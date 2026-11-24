"""Push the latest marker state and depth snapshot ID on independent connections."""

import asyncio
from concurrent.futures import Future
from http import HTTPStatus
import json
import threading

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class PreviewStreams:
    def __init__(self, host, port, origins, depth_state, marker_state):
        self._states = {"/ws/depth": depth_state, "/ws/markers": marker_state}
        self._client_count = 0
        self._started = Future()
        self._thread = threading.Thread(target=self._run, args=(host, port, origins),
                                        name="ar-websocket", daemon=True)
        self._thread.start()
        self.port = self._started.result()

    def _run(self, host, port, origins):
        try:
            asyncio.run(self._serve(host, port, origins))
        except Exception as error:
            if not self._started.done():
                self._started.set_exception(error)
            else:
                print(f"WebSocket preview stopped: {error}", flush=True)

    async def _serve(self, host, port, origins):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handle, host, port, origins=origins,
                         process_request=self._check_route, compression=None,
                         open_timeout=3, close_timeout=1, max_size=1024,
                         max_queue=1, write_limit=8192) as server:
            self._started.set_result(server.sockets[0].getsockname()[1])
            await self._stop.wait()

    def _check_route(self, connection, request):
        if request.path.partition("?")[0] not in self._states:
            return connection.respond(HTTPStatus.NOT_FOUND, "Unknown AR stream\n")
        return None

    async def _handle(self, connection):
        if self._client_count >= 16:
            await connection.close(1013, "Too many preview connections")
            return
        self._client_count += 1
        snapshot = self._states[connection.request.path.partition("?")[0]]
        previous = None
        last_send = -float("inf")
        try:
            while not self._stop.is_set():
                current = snapshot()
                # Relative age changes on every read. A heartbeat refreshes it
                # while avoiding a stream of mostly identical messages.
                key = {name: value for name, value in current.items() if name != "age_s"}
                now = self._loop.time()
                if key != previous or now - last_send >= 2:
                    await connection.send(json.dumps(current, separators=(",", ":")))
                    previous, last_send = key, now
                try:
                    await asyncio.wait_for(connection.recv(), timeout=0.20)
                except asyncio.TimeoutError:
                    continue
                await connection.close(1008, "Preview streams are read-only")
                break
        except ConnectionClosed:
            pass
        finally:
            self._client_count -= 1

    def close(self):
        if self._thread.is_alive():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                return
            self._thread.join(timeout=3)
