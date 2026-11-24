import { Arena, PALETTE } from './arena.mjs';
import { startLiveFeeds } from './live-feed.mjs';
import { projectorSettings } from './projector-settings.mjs';

const byId = (id) => document.getElementById(id);
const canvas = byId('arena');
const viewport = byId('viewport');
const arena = new Arena(canvas);
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
const demoField = { width: 1280, height: 800, rect: [268, 40, 1012, 760] };
const routes = [
  { x: .39, y: .45, rx: .13, ry: .16, phase: .70, rate: .95 },
  { x: .75, y: .25, rx: .12, ry: .08, phase: -1.25, rate: -1.05 },
  { x: .66, y: .71, rx: .17, ry: .10, phase: 1.95, rate: .80 },
  { x: .20, y: .23, rx: .10, ry: .07, phase: 2.50, rate: .85 },
  { x: .22, y: .77, rx: .10, ry: .09, phase: -1.05, rate: -1.10 },
  { x: .80, y: .51, rx: .09, ry: .10, phase: -2.60, rate: .73 },
];

const controls = ['vectors', 'rings', 'trails'];
const effects = Object.fromEntries(controls.map((name) => [name, byId(name).checked]));
const keys = new Set();
const movementKeys = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyQ', 'KeyE']);
const maxFrameAge = 2500;
const timeFormat = new Intl.DateTimeFormat('ru-RU', { timeStyle: 'medium' });
const setProjectorOrigin = projectorSettings();
const dialog = byId('connection');
const address = byId('camera-address');
const servicePage = location.pathname.startsWith('/swarm/');
address.value = servicePage ? location.origin : 'http://127.0.0.1:8765';
const cameraTelemetry = byId('camera-telemetry');
const cameraChannels = [...cameraTelemetry.querySelectorAll('.camera-channel')].map((row) => ({
  row, index: row.querySelector('.channel-index'), signal: row.querySelector('.channel-signal'),
}));
const pairSkew = byId('pair-skew');
const captureSize = byId('capture-size');
const fpsFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });

let selected = 0;
let mode = 'demo';
let field = demoField;
let markers = [];
let liveTargets = [];
let liveConnection;
let liveFrameId;
let liveExpiry = 0;
let depthImage;
let paused = motionPreference.matches;
let projection = false;
let needsRender = true;
let time = 0;
let demoTime = 0;
let drag;
let projectionControlsTimeout;
let lastTime;
let lastReadout = 0;
const manual = new Map();

const markerRows = PALETTE.map(({ color }, id) => {
  const button = document.createElement('button');
  button.className = 'marker-row';
  button.type = 'button';
  button.dataset.marker = id;
  button.style.setProperty('--marker', color);
  const name = `R–0${id}`;
  button.innerHTML = `<span class="marker-indicator" aria-hidden="true"><i></i></span>
    <span class="marker-name">${name}</span><span class="readout marker-angle">—</span>
    <span class="marker-direction" aria-hidden="true">→</span>`;
  button.addEventListener('click', () => select(id));
  byId('marker-list').append(button);
  return { button, heading: button.querySelector('.marker-angle'), arrow: button.querySelector('.marker-direction') };
});

function markDirty() {
  needsRender = true;
}

function select(id) {
  selected = id;
  markerRows.forEach(({ button }, index) => button.setAttribute('aria-pressed', String(index === id)));
  markDirty();
}

function setStatus(state, label) {
  if (byId('status-label').textContent !== label) byId('status-label').textContent = label;
  byId('source-status').dataset.state = state;
}

function showCameraTelemetry(tracking, fresh) {
  cameraTelemetry.hidden = !tracking;
  if (!tracking) return;
  tracking.cameras.forEach((camera, index) => {
    const channel = cameraChannels[index];
    const signal = camera.ready ? `${fpsFormat.format(camera.fps)} к/с` : 'Нет кадров';
    channel.row.dataset.ready = String(camera.ready);
    channel.row.title = typeof camera.error === 'string' ? camera.error : '';
    channel.index.textContent = `#${camera.index}`;
    if (channel.signal.textContent !== signal) channel.signal.textContent = signal;
  });
  pairSkew.hidden = !fresh || !Number.isFinite(tracking.pair_skew_ms) || tracking.pair_skew_ms < 0;
  if (!pairSkew.hidden) {
    const skew = Math.round(tracking.pair_skew_ms);
    pairSkew.textContent = `Разница ${skew} мс`;
    pairSkew.title = Number.isFinite(tracking.max_skew_ms)
      ? `Разница времени получения кадров. Допуск ${tracking.max_skew_ms} мс`
      : 'Разница времени получения кадров';
  }
  const size = tracking.capture_size;
  captureSize.hidden = !Array.isArray(size) || size.length !== 2
    || !size.every((value) => Number.isSafeInteger(value) && value > 0);
  if (!captureSize.hidden) captureSize.textContent = `ArUco ${size.join(' × ')}`;
}

function showCameraStatus(tracking, age, fresh) {
  if (tracking) {
    const [left, right] = tracking.cameras;
    if (!left.ready || !right.ready) {
      const side = !left.ready && !right.ready ? 'с камер' : !left.ready ? 'слева' : 'справа';
      setStatus('wait', `Нет сигнала ${side}`);
      return;
    }
    if (!fresh) {
      setStatus('wait', 'Нет синхронной стереопары');
      return;
    }
  }
  if (!fresh) {
    setStatus('wait', age === null ? 'Ожидание кадров ArUco' : 'Нет свежих кадров');
  } else if (!liveTargets.some((marker) => marker.visible)) {
    setStatus('wait', 'В поле нет меток ArUco');
  } else {
    setStatus('live', 'Камеры онлайн');
  }
}

function setField(next) {
  if (field.width === next.width && field.height === next.height && field.rect.every((v, i) => v === next.rect[i])) return;
  field = next;
  byId('resolution').innerHTML = `${field.width} × ${field.height} <small>PX</small>`;
  arena.reset();
  markDirty();
}

function wrapAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function clampToField(x, y) {
  const [l, t, r, b] = field.rect;
  return { x: Math.min(r, Math.max(l, x)), y: Math.min(b, Math.max(t, y)) };
}

function routeMarker(route, id, elapsed) {
  const [l, t, r, b] = demoField.rect;
  const phase = elapsed * route.rate + route.phase;
  const rx = route.rx * (r - l);
  const ry = route.ry * (b - t);
  const tx = Math.cos(phase);
  const ty = Math.sin(phase);
  return {
    id, x: l + route.x * (r - l) + rx * tx, y: t + route.y * (b - t) + ry * ty,
    angle: Math.atan2(ry * tx * route.rate, -rx * ty * route.rate),
  };
}

function demoMarkers(dt) {
  const xAxis = Number(keys.has('ArrowRight')) - Number(keys.has('ArrowLeft'));
  const yAxis = Number(keys.has('ArrowDown')) - Number(keys.has('ArrowUp'));
  const yaw = Number(keys.has('KeyE')) - Number(keys.has('KeyQ'));
  const distance = 200 * dt / (Math.hypot(xAxis, yAxis) || 1);
  if (selected !== null && (xAxis || yAxis || yaw) && dt > 0 && !drag) {
    const item = manual.get(selected) ?? { ...markers.find((marker) => marker.id === selected) };
    Object.assign(item, clampToField(item.x + distance * xAxis, item.y + distance * yAxis));
    delete item.target;
    if (xAxis || yAxis) item.angle = Math.atan2(yAxis, xAxis);
    item.angle = wrapAngle(item.angle + yaw * Math.PI * dt);
    manual.set(selected, item);
  }
  return routes.map((route, id) => {
    const item = manual.get(id);
    if (!item) return routeMarker(route, id, demoTime);
    if (item.target) {
      const dx = item.target.x - item.x;
      const dy = item.target.y - item.y;
      if (Math.hypot(dx, dy) > .8) {
        const blend = paused ? 1 : 1 - Math.exp(-dt / .047);
        item.x += dx * blend;
        item.y += dy * blend;
        item.angle = wrapAngle(item.angle + wrapAngle(Math.atan2(dy, dx) - item.angle) * blend);
      } else {
        Object.assign(item, item.target);
        delete item.target;
      }
    }
    return item;
  });
}

function resetDemo() {
  demoTime = 0;
  keys.clear();
  manual.clear();
  endDrag();
  arena.reset();
  // Give the first visible frame the same short history as continuous playback.
  const steps = paused ? 0 : 16;
  for (let step = -steps; step < 0; step++) {
    const history = routes.map((route, id) => routeMarker(route, id, step * .045));
    time += .045;
    draw(history, .045);
  }
  markers = demoMarkers(0);
  updateReadouts();
  markDirty();
}

function liveMarkers(dt, now) {
  if (liveExpiry && now > liveExpiry) {
    liveExpiry = 0;
    liveTargets = [];
    pairSkew.hidden = true;
    showCameraStatus(liveConnection?.tracking, maxFrameAge, false);
  }
  const blend = 1 - Math.exp(-dt / .095);
  return liveTargets.filter((target) => now < target.expiresAt).map((target) => {
    const previous = markers.find((marker) => marker.id === target.id);
    if (!previous || !target.visible || !previous.visible
        || Math.hypot(target.x - previous.x, target.y - previous.y) > (field.rect[2] - field.rect[0]) * .30) {
      return { ...target };
    }
    previous.x += (target.x - previous.x) * blend;
    previous.y += (target.y - previous.y) * blend;
    previous.angle = wrapAngle(previous.angle + wrapAngle(target.angle - previous.angle) * blend);
    previous.lastSeen = target.lastSeen;
    previous.quality = target.quality;
    return previous;
  });
}

function parseFrame(frame) {
  const projection = frame?.projection;
  const rect = projection?.available_rect;
  if (frame?.schema_version !== 1 || !Number.isSafeInteger(frame.frame_id) || frame.frame_id < 0
      || !Array.isArray(frame.markers) || !Number.isSafeInteger(projection?.width)
      || !Number.isSafeInteger(projection.height) || !Array.isArray(rect) || rect.length !== 4
      || !rect.every(Number.isFinite)
      || !(0 <= rect[0] && rect[0] < rect[2] && rect[2] <= projection.width
         && 0 <= rect[1] && rect[1] < rect[3] && rect[3] <= projection.height)
      || (frame.age_s !== null && !(Number.isFinite(frame.age_s) && frame.age_s >= 0))) {
    throw new Error('Некорректный формат кадра');
  }
  const validMarkers = new Map();
  for (const item of frame.markers) {
    if (!item || !Number.isInteger(item.id) || item.id < 0 || item.id >= markerRows.length
        || !Number.isFinite(item.angle)
        || !(item.x >= rect[0] && item.x <= rect[2] && item.y >= rect[1] && item.y <= rect[3])
        || !Number.isFinite(item.x) || !Number.isFinite(item.y)) continue;
    validMarkers.set(item.id, {
      id: item.id, x: item.x, y: item.y, angle: wrapAngle(item.angle),
      visible: item.visible !== false,
      lastSeen: item.last_seen_unix_ms >= 0 && item.last_seen_unix_ms <= 8.64e15
        && Number.isFinite(item.last_seen_unix_ms) ? item.last_seen_unix_ms : null,
      quality: item.quality,
    });
  }
  const tracking = frame.tracking;
  const validTracking = Array.isArray(tracking?.cameras) && tracking.cameras.length === cameraChannels.length
    && tracking.cameras.every((camera) => Number.isSafeInteger(camera?.index) && camera.index >= 0
      && typeof camera.ready === 'boolean' && Number.isFinite(camera.fps) && camera.fps >= 0);
  return {
    id: frame.frame_id, age: frame.age_s,
    capturedAt: Number.isFinite(frame.captured_at_unix_ms) ? frame.captured_at_unix_ms : null,
    field: { width: projection.width, height: projection.height, rect },
    markers: [...validMarkers.values()],
    tracking: validTracking ? tracking : null,
  };
}

function stopCamera() {
  if (!liveConnection) return;
  liveConnection.stop();
  liveConnection = undefined;
  showDepth(null);
}

function recordPaths(connection, positions, timestamp, newFrame) {
  const byMarker = new Map(positions.filter(({ visible }) => visible).map((marker) => [marker.id, marker]));
  connection.paths.forEach((path, id) => {
    const marker = byMarker.get(id);
    if (!marker) connection.active.delete(id);
    if (!marker || !newFrame) return;
    const previous = path.at(-1);
    const join = connection.active.has(id) && previous && timestamp - previous.time < 2500;
    // Keep sample gaps visible and bound the retained path even at higher legacy rates.
    path.push({ x: marker.x, y: marker.y, time: timestamp, join });
    if (path.length > 40) path.shift();
    connection.active.add(id);
  });
}

function receiveCameraFrame(connection, value, delay) {
  const frame = parseFrame(value);
  const fieldBefore = field;
  setField(frame.field);
  const restarted = liveFrameId !== undefined && frame.id < liveFrameId;
  if (fieldBefore !== field || restarted) {
    arena.reset();
    markers = [];
    liveFrameId = undefined;
    connection.paths.forEach((path) => { path.length = 0; });
    connection.active.clear();
  }
  const now = performance.now();
  const age = frame.age === null ? Infinity : frame.age * 1000 + delay;
  connection.tracking = frame.tracking;
  const paired = !frame.tracking || (frame.tracking.pair_skew_ms !== null
    && frame.tracking.cameras.every((camera) => camera.ready));
  const fresh = age < maxFrameAge && paired;
  liveTargets = age < maxFrameAge ? frame.markers.map((marker) => {
    const sampleAge = frame.capturedAt === null || marker.lastSeen === null
      ? age : age + Math.max(0, frame.capturedAt - marker.lastSeen);
    return { ...marker, visible: fresh && marker.visible, expiresAt: now + maxFrameAge - sampleAge };
  }).filter((marker) => marker.expiresAt > now) : [];
  liveExpiry = age < maxFrameAge ? now + maxFrameAge - age : 0;
  recordPaths(connection, liveTargets, now - age, liveFrameId !== frame.id);
  showCameraTelemetry(frame.tracking, fresh);
  showCameraStatus(frame.tracking, frame.age, fresh);
  liveFrameId = frame.id;
}

function cameraError(connection) {
  liveTargets = [];
  liveExpiry = 0;
  connection.tracking = null;
  connection.active.clear();
  showCameraTelemetry(null, false);
  setStatus('offline', 'Нет связи со службой AR');
}

function showDepth(bitmap, snapshotId) {
  depthImage?.close();
  depthImage = bitmap;
  const label = byId('depth-status');
  label.textContent = bitmap ? `Карта · кадр ${snapshotId}` : 'Ожидание карты';
  label.title = bitmap ? `Получен в ${timeFormat.format(Date.now())}` : '';
  markDirty();
}

function syncMode() {
  const demo = mode === 'demo';
  document.body.dataset.mode = mode;
  byId('demo-source').setAttribute('aria-pressed', String(demo));
  byId('camera-source').setAttribute('aria-pressed', String(!demo));
  byId('pause').disabled = !demo;
  byId('reset').disabled = !demo;
  byId('depth-status').hidden = demo;
  byId('project-label').textContent = demo ? 'На проектор' : 'Эффекты на экран';
  byId('trails-label').textContent = demo ? 'Шлейф частиц' : 'Следы движения';
  showCameraTelemetry(null, false);
  byId('field-hint').innerHTML = demo
    ? 'Перетащите маркер или используйте <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd>. Поворот <kbd>Q</kbd> <kbd>E</kbd>.'
    : 'Выберите маркер на поле или в списке, чтобы выделить его.';
}

function useCamera(origin) {
  stopCamera();
  endDrag();
  keys.clear();
  mode = 'live';
  markers = [];
  liveTargets = [];
  liveFrameId = undefined;
  liveExpiry = 0;
  arena.reset();
  syncMode();
  showDepth(null);
  setProjectorOrigin(origin);
  setStatus('wait', 'Подключение камер');
  const connection = { paths: PALETTE.map(() => []), active: new Set() };
  liveConnection = connection;
  connection.stop = startLiveFeeds(origin, {
    markers: (value, delay) => receiveCameraFrame(connection, value, delay),
    markerError: () => cameraError(connection),
    depth: showDepth,
    depthError: () => {
      byId('depth-status').textContent = depthImage ? 'Последняя карта' : 'Карта недоступна';
    },
  });
}

function setPaused(next) {
  paused = next;
  byId('pause-label').textContent = paused ? 'Продолжить' : 'Пауза';
  byId('pause').firstElementChild.className = `icon ${paused ? 'icon-play' : 'icon-pause'}`;
  setStatus(paused ? 'wait' : 'demo', paused ? 'Симуляция на паузе' : 'Симуляция активна');
  markDirty();
}

function updateReadouts() {
  const active = markers.filter((marker) => marker.visible !== false).length;
  byId('marker-count').textContent = `${active} / ${markerRows.length}`;
  markerRows.forEach(({ button, heading, arrow }, id) => {
    const marker = markers.find((item) => item.id === id);
    const degrees = marker ? (Math.round(marker.angle * 180 / Math.PI) + 360) % 360 : null;
    const stale = marker?.visible === false;
    button.dataset.present = String(Boolean(marker && !stale));
    button.dataset.stale = String(stale);
    heading.textContent = stale ? 'Потерян' : degrees === null ? '—' : `${String(degrees).padStart(3, '0')}°`;
    arrow.style.rotate = marker ? `${degrees}deg` : '';
  });
  const detail = byId('marker-detail');
  const marker = markers.find(({ id }) => id === selected);
  const deviation = marker?.quality?.reprojection_rms_px ?? marker?.quality?.reprojection_error_px;
  detail.hidden = mode !== 'live' || !marker?.lastSeen;
  if (!detail.hidden) {
    const accuracy = Number.isFinite(deviation) ? ` · RMS ${deviation.toFixed(2)} px` : '';
    detail.textContent = `Обнаружен в ${timeFormat.format(marker.lastSeen)}${accuracy}`;
  }
}

function endDrag() {
  if (!drag) return;
  const pointerId = drag.pointer;
  drag = undefined;
  delete canvas.dataset.dragging;
  if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
}

function draw(items, dt) {
  arena.render({
    field, markers: items, time, dt, effects, selected, projection,
    live: mode === 'live', depthImage, paths: liveConnection?.paths, wallTime: performance.now(),
    quality: motionPreference.matches ? .55 : 1,
  });
}

function frame(now) {
  const elapsed = lastTime === undefined ? 0 : Math.min(.05, (now - lastTime) / 1000);
  lastTime = now;
  const dt = mode === 'demo' && paused ? 0 : elapsed;
  time += dt;
  if (mode === 'demo') {
    demoTime += dt;
    if (dt || needsRender) markers = demoMarkers(dt);
  } else {
    markers = liveMarkers(dt, now);
  }
  if (dt || needsRender) {
    draw(markers, dt);
    needsRender = false;
  }
  if (now - lastReadout > 100) {
    lastReadout = now;
    updateReadouts();
  }
  requestAnimationFrame(frame);
}

function setProjection(active) {
  projection = active;
  document.body.dataset.projection = String(active);
  byId('leave-projection').hidden = !active;
  delete viewport.dataset.controls;
  keys.clear();
  endDrag();
  markDirty();
  if (!active) byId('project').focus();
}

function exitProjection() {
  if (document.fullscreenElement === viewport) {
    document.exitFullscreen().catch(() => setProjection(false));
  } else {
    setProjection(false);
  }
}

controls.forEach((name) => byId(name).addEventListener('change', (event) => {
  effects[name] = event.currentTarget.checked;
  markDirty();
}));

byId('pause').addEventListener('click', () => setPaused(!paused));
byId('reset').addEventListener('click', resetDemo);
byId('demo-source').addEventListener('click', () => {
  if (mode === 'demo') return;
  stopCamera();
  mode = 'demo';
  setProjectorOrigin(null);
  syncMode();
  setField(demoField);
  setPaused(motionPreference.matches);
  resetDemo();
});
byId('camera-source').addEventListener('click', () => {
  byId('address-error').hidden = true;
  dialog.showModal();
  address.select();
});
byId('cancel-connect').addEventListener('click', () => dialog.close());
byId('connection-form').addEventListener('submit', (event) => {
  event.preventDefault();
  let url;
  try {
    url = new URL(address.value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
        || url.pathname !== '/' || url.search || url.hash) throw new Error();
  } catch {
    byId('address-error').hidden = false;
    address.focus();
    return;
  }
  dialog.close();
  useCamera(url.origin);
});

canvas.addEventListener('pointerdown', (event) => {
  if (projection || event.button !== 0 || drag) return;
  const id = arena.pick(event.clientX, event.clientY);
  if (id === null) return;
  select(id);
  canvas.focus({ preventScroll: true });
  if (mode !== 'demo') return;
  const item = markers.find((marker) => marker.id === id);
  const point = arena.point(event.clientX, event.clientY);
  if (!item || !point) return;
  drag = {
    pointer: event.pointerId, id, dx: item.x - point.x, dy: item.y - point.y,
    startX: event.clientX, startY: event.clientY, started: false,
  };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', (event) => {
  if (drag?.pointer !== event.pointerId) return;
  const point = arena.point(event.clientX, event.clientY);
  if (!point) return;
  if (!drag.started) {
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 3) return;
    drag.started = true;
    manual.set(drag.id, { ...markers.find((marker) => marker.id === drag.id) });
    canvas.dataset.dragging = 'true';
  }
  manual.get(drag.id).target = clampToField(point.x + drag.dx, point.y + drag.dy);
  markDirty();
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(type, (event) => { if (drag?.pointer === event.pointerId) endDrag(); });
}

canvas.addEventListener('keydown', (event) => {
  if (mode !== 'demo' || projection || paused || !movementKeys.has(event.code)) return;
  keys.add(event.code);
  event.preventDefault();
});
window.addEventListener('keyup', (event) => keys.delete(event.code));
window.addEventListener('blur', () => { keys.clear(); endDrag(); });
canvas.addEventListener('blur', () => keys.clear());

byId('project').addEventListener('click', () => {
  setProjection(true);
  byId('project').blur();
  viewport.requestFullscreen?.().catch(() => {});
});
byId('leave-projection').addEventListener('click', exitProjection);
document.addEventListener('fullscreenchange', () => {
  if (projection && document.fullscreenElement !== viewport) setProjection(false);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && projection) exitProjection();
});
viewport.addEventListener('pointermove', showProjectionControls);
viewport.addEventListener('pointerdown', showProjectionControls);

function showProjectionControls() {
  if (!projection) return;
  viewport.dataset.controls = 'true';
  clearTimeout(projectionControlsTimeout);
  projectionControlsTimeout = setTimeout(() => { delete viewport.dataset.controls; }, 1800);
}

new ResizeObserver(markDirty).observe(viewport);
select(0);
if (servicePage && new URLSearchParams(location.search).get('live') === '1') {
  useCamera(location.origin);
} else {
  setPaused(paused);
  resetDemo();
}
requestAnimationFrame(frame);
