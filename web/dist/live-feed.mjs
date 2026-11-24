export async function request(url, read, { signal, timeout = 2400, ...options } = {}) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  const timer = setTimeout(abort, timeout);
  try {
    const response = await fetch(url, { ...options, cache: 'no-store', signal: controller.signal });
    return await read(response);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

async function readJson(response) {
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body?.error?.message || body?.message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

export const requestJson = (url, options) => request(url, readJson, options);

export function startLiveFeeds(origin, { markers, markerError, depth, depthError }) {
  const lifetime = new AbortController();
  const { signal } = lifetime;
  const url = (path) => new URL(path, origin);
  let streamConfig;
  let depthTarget;
  let depthActive;
  let depthSnapshot;
  let depthGeneration = 0;
  let projectionRevision;
  let depthSocketSeen = -Infinity;
  let markerSocketSeen = -Infinity;
  const sockets = new Map();

  function poll(action, interval) {
    let timer;
    const tick = async () => {
      await action();
      if (!signal.aborted) timer = setTimeout(tick, interval);
    };
    signal.addEventListener('abort', () => clearTimeout(timer), { once: true });
    tick();
  }

  async function fetchMarkers() {
    const start = performance.now();
    if (start - markerSocketSeen < 2100) return;
    try {
      const frame = await requestJson(url('/api/v1/markers'), { signal, timeout: 1400 });
      if (!signal.aborted && markerSocketSeen < start) markers(frame, (performance.now() - start) / 2);
    } catch {
      if (!signal.aborted && markerSocketSeen < start) markerError();
    }
  }

  async function fetchDepth() {
    if (depthActive) return;
    while (depthTarget && !signal.aborted) {
      depthActive = depthTarget;
      depthTarget = undefined;
      try {
        const result = await request(url('/api/v1/depth.png'), async (response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const id = response.headers.get('X-Snapshot-Id');
          const revision = response.headers.get('X-Projection-Revision');
          return {
            blob: await response.blob(),
            id: id === null ? depthActive.id : Number(id),
            revision: revision ?? depthActive.revision,
          };
        }, { signal });
        const bitmap = await createImageBitmap(result.blob);
        if (signal.aborted || depthActive.generation !== depthGeneration
            || (projectionRevision && result.revision !== projectionRevision)) {
          bitmap.close();
        } else {
          depthSnapshot = { id: result.id, revision: result.revision };
          depth(bitmap, result.id);
          if (depthTarget?.revision === result.revision && depthTarget.id <= result.id) {
            depthTarget = undefined;
          }
        }
      } catch {
        if (!signal.aborted) depthError();
      } finally {
        depthActive = undefined;
      }
    }
  }

  function clearDepth(revision) {
    projectionRevision = revision;
    if (!depthSnapshot && !depthTarget && depthActive?.generation !== depthGeneration) return;
    depthGeneration += 1;
    depthSnapshot = undefined;
    depthTarget = undefined;
    depth(null);
  }

  function wantDepth(id, revision) {
    if (signal.aborted || !Number.isSafeInteger(id) || id < 0 || !revision) return;
    if (id === 0) {
      clearDepth(revision);
      return;
    }
    if (projectionRevision && projectionRevision !== revision) {
      clearDepth(revision);
    }
    projectionRevision = revision;
    if (depthSnapshot?.revision === revision && depthSnapshot.id >= id) return;
    if ([depthActive, depthTarget].some((entry) => entry?.generation === depthGeneration
      && entry.revision === revision && entry.id >= id)) return;
    depthTarget = { id, revision, generation: depthGeneration };
    fetchDepth();
  }

  function subscribe(name, receive) {
    if (sockets.has(name) || !streamConfig) return;
    const endpoint = url(streamConfig[name]);
    endpoint.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:';
    endpoint.port = streamConfig.port;
    let socket;
    try {
      socket = new WebSocket(endpoint);
    } catch {
      return;
    }
    sockets.set(name, socket);
    socket.addEventListener('message', ({ data }) => {
      if (signal.aborted || sockets.get(name) !== socket) return;
      try {
        receive(JSON.parse(data));
      } catch {
        socket.close();
      }
    });
    socket.addEventListener('close', () => {
      if (sockets.get(name) !== socket) return;
      sockets.delete(name);
      if (name === 'markers') markerSocketSeen = -Infinity;
    });
  }

  function updateStreams(config) {
    const valid = Number.isInteger(config?.port) && config.port > 0 && config.port <= 65535
      && config.depth === '/ws/depth' && config.markers === '/ws/markers';
    const next = valid ? config : null;
    if (next?.port !== streamConfig?.port) {
      for (const socket of sockets.values()) socket.close();
      sockets.clear();
      markerSocketSeen = -Infinity;
    }
    streamConfig = next;
    subscribe('markers', (frame) => {
      markers(frame, 0);
      markerSocketSeen = performance.now();
    });
    subscribe('depth', ({ snapshot_id, projection_revision }) => {
      depthSocketSeen = performance.now();
      wantDepth(snapshot_id, projection_revision);
    });
  }

  async function fetchState() {
    const start = performance.now();
    try {
      const state = await requestJson(url('/api/v1/state'), { signal });
      if (signal.aborted) return;
      updateStreams(state.streams);
      if (depthSocketSeen >= start) return;
      if (state.snapshot?.depth_ready) {
        wantDepth(state.snapshot.id, state.snapshot.projection_revision ?? state.projection_revision);
      } else {
        clearDepth(state.projection_revision);
      }
    } catch {
      if (!signal.aborted) depthError();
    }
  }

  poll(fetchMarkers, 380);
  poll(fetchState, 850);
  return () => {
    lifetime.abort();
    for (const socket of sockets.values()) socket.close();
    sockets.clear();
  };
}
