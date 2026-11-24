import { Arena, PALETTE } from './arena.mjs';

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
const maxFrameAge = 750;
const dialog = byId('connection');
const address = byId('camera-address');
address.value = location.pathname.startsWith('/swarm/') ? location.origin : 'http://127.0.0.1:8765';

let selected = 0;
let mode = 'demo';
let field = demoField;
let markers = [];
let liveTargets = [];
let liveConnection;
let liveFrameId;
let liveExpiry = 0;
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
    setStatus('wait', 'Нет свежих кадров');
  }
  const blend = 1 - Math.exp(-dt / .052);
  return liveTargets.map((target) => {
    const previous = markers.find((marker) => marker.id === target.id);
    if (!previous || Math.hypot(target.x - previous.x, target.y - previous.y) > (field.rect[2] - field.rect[0]) * .30) {
      return { ...target };
    }
    previous.x += (target.x - previous.x) * blend;
    previous.y += (target.y - previous.y) * blend;
    previous.angle = wrapAngle(previous.angle + wrapAngle(target.angle - previous.angle) * blend);
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
    validMarkers.set(item.id, { id: item.id, x: item.x, y: item.y, angle: wrapAngle(item.angle) });
  }
  return {
    id: frame.frame_id, age: frame.age_s,
    field: { width: projection.width, height: projection.height, rect },
    markers: [...validMarkers.values()],
  };
}

function stopCamera() {
  if (!liveConnection) return;
  clearTimeout(liveConnection.timer);
  liveConnection.request?.abort();
  liveConnection = undefined;
}

async function fetchCamera(connection) {
  const start = performance.now();
  const request = new AbortController();
  const timeout = setTimeout(() => request.abort(), 1400);
  connection.request = request;
  try {
    const response = await fetch(connection.url, { signal: request.signal, cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const frame = parseFrame(await response.json());
    if (liveConnection !== connection) return;
    const fieldBefore = field;
    setField(frame.field);
    const restarted = liveFrameId !== undefined && frame.id < liveFrameId;
    if (restarted) arena.reset();
    if (fieldBefore !== field || restarted) {
      markers = [];
      liveFrameId = undefined;
    }
    const age = frame.age === null ? Infinity : frame.age * 1000 + (performance.now() - start) / 2;
    if (age < maxFrameAge) {
      if (liveFrameId !== frame.id || !liveExpiry) {
        liveTargets = frame.markers;
        liveExpiry = performance.now() + maxFrameAge - age;
      }
      setStatus('live', 'Камеры онлайн');
    } else {
      liveTargets = [];
      liveExpiry = 0;
      setStatus('wait', frame.age === null ? 'Ожидание кадров ArUco' : 'Нет свежих кадров');
    }
    liveFrameId = frame.id;
  } catch {
    if (liveConnection !== connection) return;
    liveTargets = [];
    liveExpiry = 0;
    setStatus('offline', 'Нет связи со службой AR');
  } finally {
    clearTimeout(timeout);
    if (liveConnection === connection) {
      const interval = liveExpiry ? 68 : 400;
      connection.timer = setTimeout(() => fetchCamera(connection), Math.max(0, start + interval - performance.now()));
    }
  }
}

function syncMode() {
  const demo = mode === 'demo';
  document.body.dataset.mode = mode;
  byId('demo-source').setAttribute('aria-pressed', String(demo));
  byId('camera-source').setAttribute('aria-pressed', String(!demo));
  byId('pause').disabled = !demo;
  byId('reset').disabled = !demo;
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
  setStatus('wait', 'Подключение камер');
  liveConnection = { url: new URL('/api/v1/markers', origin) };
  fetchCamera(liveConnection);
}

function setPaused(next) {
  paused = next;
  byId('pause-label').textContent = paused ? 'Продолжить' : 'Пауза';
  byId('pause').firstElementChild.className = `icon ${paused ? 'icon-play' : 'icon-pause'}`;
  setStatus(paused ? 'wait' : 'demo', paused ? 'Симуляция на паузе' : 'Симуляция активна');
  markDirty();
}

function updateReadouts() {
  byId('marker-count').textContent = `${markers.length} / ${markerRows.length}`;
  markerRows.forEach(({ button, heading, arrow }, id) => {
    const marker = markers.find((item) => item.id === id);
    const degrees = marker ? (Math.round(marker.angle * 180 / Math.PI) + 360) % 360 : null;
    button.dataset.present = String(Boolean(marker));
    heading.textContent = degrees === null ? '—' : `${String(degrees).padStart(3, '0')}°`;
    arrow.style.rotate = marker ? `${degrees}deg` : '';
  });
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
setPaused(paused);
resetDemo();
requestAnimationFrame(frame);
