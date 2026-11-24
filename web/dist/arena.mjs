const TAU = Math.PI * 2;
const MAX_PARTICLES = 480;
const TRAIL_POINTS = 64;
const TRAIL_SECONDS = 1.75;
const TRAIL_BANDS = 7;
const PATH_MS = 7200;
const PATH_COLOR = '#54eec7';
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export const PALETTE = [
  { color: '#59edff', rgb: '89,237,255' },
  { color: '#ff9d56', rgb: '255,157,86' },
  { color: '#be99ff', rgb: '190,153,255' },
  { color: '#79ffb0', rgb: '121,255,176' },
  { color: '#ff83ba', rgb: '255,131,186' },
  { color: '#92a9ff', rgb: '146,169,255' },
];

const tint = (id, alpha) => `rgba(${PALETTE[id].rgb},${alpha})`;

function layer(canvas, width, height) {
  const result = canvas.ownerDocument.createElement('canvas');
  result.width = width;
  result.height = height;
  return result;
}

function flare(canvas, id) {
  const sprite = layer(canvas, 64, 64);
  const ctx = sprite.getContext('2d');
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, tint(id, 0.85));
  gradient.addColorStop(0.24, tint(id, 0.48));
  gradient.addColorStop(0.55, tint(id, 0.12));
  gradient.addColorStop(1, tint(id, 0));
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  return sprite;
}

function cutCorner(ctx, x, y, length, dx, dy) {
  const bevel = length * 0.38;
  ctx.moveTo(x, y + dy * length);
  ctx.lineTo(x, y + dy * bevel);
  ctx.lineTo(x + dx * bevel, y);
  ctx.lineTo(x + dx * length, y);
}

function grid(ctx, left, top, right, bottom, step) {
  ctx.beginPath();
  for (let x = left; x < right; x += step) {
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
  }
  for (let y = top; y < bottom; y += step) {
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
  }
  ctx.stroke();
}

function drawTerrain(ctx, left, top, width, height) {
  // The contour families are cached with the grid. Each has a different
  // profile and axis, with darker ink on the lower-facing slopes.
  const hills = [
    [0.2, 0.22, 0.24, 0.64, -0.45, 3.18],
    [0.83, 0.23, 0.23, 0.83, 0.17, 0.6],
    [0.14, 0.81, 0.23, 0.81, 0.48, 0.86],
    [0.69, 0.78, 0.29, 0.67, -0.58, 1.37],
  ];
  const unit = Math.min(width, height);

  for (const [x, y, range, flatten, rotation, phase] of hills) {
    ctx.save();
    ctx.translate(left + width * x, top + height * y);
    ctx.rotate(rotation);
    ctx.scale(1, flatten);
    for (let index = 0; index < 11; index += 1) {
      const radius = (1.08 - index * 0.086) * unit * range;
      ctx.beginPath();
      for (let segment = 0; segment <= 96; segment += 1) {
        const angle = segment / 96 * TAU;
        const fold = 1 + Math.sin(angle * 3 + phase + index * 0.13) * 0.16
          + Math.cos(angle * 5 - phase) * 0.08 + Math.sin(angle * 2 + phase) * 0.21;
        const px = Math.cos(angle) * radius * fold + index * 1.8;
        const py = Math.sin(angle) * radius * fold - index * 1.65;
        if (segment === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = index % 3 === 0 ? '#4b529317' : '#383f7909';
      ctx.fill();
      ctx.strokeStyle = index % 3 === 0 ? '#7882c64d' : '#6877b22b';
      ctx.lineWidth = index % 3 === 0 ? 1.4 : 1;
      ctx.stroke();
    }
    ctx.restore();
  }
}

function drawField(ctx, layout, rect) {
  const { width, height, scale, offsetX, offsetY, live } = layout;
  const [left, top, right, bottom] = rect;
  const fieldWidth = right - left;
  const fieldHeight = bottom - top;
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;
  const unit = Math.min(fieldWidth, fieldHeight);
  const spacing = unit / 20;
  const shade = ctx.createRadialGradient(width * 0.52, height * 0.52, 0, width * 0.52,
    height * 0.52, Math.hypot(width, height) * 0.57);
  shade.addColorStop(0, '#171e37');
  shade.addColorStop(0.55, '#101831');
  shade.addColorStop(1, '#080d19');
  ctx.fillStyle = shade;
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.lineWidth = 0.7 / scale;
  ctx.strokeStyle = '#475f9519';
  grid(ctx, left - spacing * 12, top - spacing * 4, right + spacing * 12, bottom + spacing * 4, spacing);

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, fieldWidth, fieldHeight);
  ctx.clip();
  const ground = ctx.createLinearGradient(left, top, right, bottom);
  ground.addColorStop(0, '#131f38');
  ground.addColorStop(0.58, '#121a30');
  ground.addColorStop(1, '#1b1935');
  ctx.fillStyle = ground;
  ctx.fillRect(left, top, fieldWidth, fieldHeight);
  ctx.strokeStyle = '#779bd922';
  grid(ctx, left, top, right, bottom, spacing);
  ctx.strokeStyle = '#7e9bda22';
  grid(ctx, left, top, right, bottom, spacing * 5);
  if (!live) drawTerrain(ctx, left, top, fieldWidth, fieldHeight);

  // Long reference axes and a faded calibration target anchor the map.
  const targetRadius = unit * 0.20;
  ctx.strokeStyle = '#819dcc35';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 8]);
  ctx.beginPath();
  ctx.moveTo(centerX, top);
  ctx.lineTo(centerX, bottom);
  ctx.moveTo(left, centerY);
  ctx.lineTo(right, centerY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#7796cf20';
  for (const ratio of [0.34, 0.62, 1]) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, targetRadius * ratio, 0, TAU);
    ctx.stroke();
  }
  ctx.beginPath();
  for (let i = 0; i < 48; i += 1) {
    const angle = i * TAU / 48;
    const inner = targetRadius - (i % 4 ? 4 : 9);
    const x = Math.cos(angle);
    const y = Math.sin(angle);
    ctx.moveTo(centerX + x * inner, centerY + y * inner);
    ctx.lineTo(centerX + x * targetRadius, centerY + y * targetRadius);
  }
  ctx.stroke();
  ctx.restore();
  drawFieldBorder(ctx, layout, rect);
  ctx.restore();
}

function drawFieldBorder(ctx, layout, rect) {
  const [left, top, right, bottom] = rect;
  const scale = layout.scale;
  const step = Math.min(right - left, bottom - top) / 20;
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;
  ctx.strokeStyle = '#7c9ed254';
  ctx.lineWidth = 1 / scale;
  ctx.strokeRect(left, top, right - left, bottom - top);
  ctx.beginPath();
  for (let x = left + step; x < right - step * 0.5; x += step) {
    const size = Math.abs(x - centerX) < step * 0.55 ? 9 : 4;
    ctx.moveTo(x, top);
    ctx.lineTo(x, top - size / scale);
    ctx.moveTo(x, bottom);
    ctx.lineTo(x, bottom + size / scale);
  }
  for (let y = top + step; y < bottom - step * 0.5; y += step) {
    const size = Math.abs(y - centerY) < step * 0.55 ? 9 : 4;
    ctx.moveTo(left, y);
    ctx.lineTo(left - size / scale, y);
    ctx.moveTo(right, y);
    ctx.lineTo(right + size / scale, y);
  }
  ctx.stroke();
  ctx.beginPath();
  cutCorner(ctx, left, top, 18 / scale, 1, 1);
  cutCorner(ctx, right, top, 18 / scale, -1, 1);
  cutCorner(ctx, right, bottom, 18 / scale, -1, -1);
  cutCorner(ctx, left, bottom, 18 / scale, 1, -1);
  ctx.strokeStyle = '#a4c3edc9';
  ctx.lineWidth = 1.5 / scale;
  ctx.stroke();
}

function newTrack() {
  return {
    previousAngle: null, lastTurn: -Infinity, turnAngle: 0, turnSince: 0, direction: 1,
    beforeX: 0, beforeY: 0, carry: 0, head: 0, count: 0,
    x: new Float32Array(TRAIL_POINTS),
    y: new Float32Array(TRAIL_POINTS),
    time: new Float64Array(TRAIL_POINTS),
  };
}

function trailPoint(track, x, y, time) {
  const write = track.head;
  track.x[write] = x;
  track.y[write] = y;
  track.time[write] = time;
  track.head = (write + 1) % TRAIL_POINTS;
  track.count = Math.min(track.count + 1, TRAIL_POINTS);
}

function roundBox(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

export class Arena {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.backdrop = layer(canvas, 1, 1);
    this.flares = PALETTE.map((_, id) => flare(canvas, id));
    this.tracks = PALETTE.map(newTrack);
    this.particles = Array.from({ length: MAX_PARTICLES }, () => ({}));
    this.particleCount = 0;
    this.visibleMarkers = [];
    this.layout = null;
  }

  reset() {
    for (const track of this.tracks) {
      track.previousAngle = null;
      track.lastTurn = -Infinity;
      track.turnAngle = 0;
      track.turnSince = 0;
      track.carry = 0;
      track.head = 0;
      track.count = 0;
    }
    this.particleCount = 0;
    this.visibleMarkers = [];
  }

  pick(clientX, clientY) {
    if (!this.layout) return null;
    const point = this.mapPoint(clientX, clientY);
    let best = null;
    let nearest = (27 / this.layout.scale) ** 2;
    for (const marker of this.visibleMarkers) {
      const distance = (marker.x - point.x) ** 2 + (marker.y - point.y) ** 2;
      if (distance <= nearest) {
        nearest = distance;
        best = marker.id;
      }
    }
    return best;
  }

  mapPoint(clientX, clientY) {
    const { left, top, width, height } = this.canvas.getBoundingClientRect();
    const layout = this.layout;
    return {
      x: ((clientX - left) * layout.width / width - layout.offsetX) / layout.scale,
      y: ((clientY - top) * layout.height / height - layout.offsetY) / layout.scale,
    };
  }

  point(clientX, clientY) {
    if (!this.layout) return null;
    const point = this.mapPoint(clientX, clientY);
    const [left, top, right, bottom] = this.layout.rect;
    return { x: clamp(point.x, left, right), y: clamp(point.y, top, bottom) };
  }

  resize(field, projection, quality, live) {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width <= 1 || height <= 1) return null;
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 1 + quality);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    const { rect } = field;
    const current = this.layout;
    if (current?.field === field && current.width === width && current.height === height
        && current.dpr === dpr && current.projection === projection && current.live === live) return current;

    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const [left, top, right, bottom] = projection ? [0, 0, field.width, field.height] : rect;
    const fieldWidth = right - left;
    const fieldHeight = bottom - top;
    const padding = projection ? 0 : Math.min(36, width * 0.062, height * 0.073);
    const scale = Math.min((width - padding * 2) / fieldWidth, (height - padding * 2) / fieldHeight);
    const offsetX = (width - fieldWidth * scale) / 2 - left * scale;
    const offsetY = (height - fieldHeight * scale) / 2 - top * scale + (projection ? 0 : padding * 0.14);
    const unit = Math.min(rect[2] - rect[0], rect[3] - rect[1]);
    const fxScale = projection ? unit / 720 : clamp(0.7 / scale, unit / 720, unit / 495);
    this.layout = { field, projection, live, rect, width, height, dpr, offsetX, offsetY, scale, fxScale };

    if (!projection) {
      this.backdrop.width = pixelWidth;
      this.backdrop.height = pixelHeight;
      const ctx = this.backdrop.getContext('2d', { alpha: false });
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawField(ctx, this.layout, rect);
    }
    return this.layout;
  }

  drawMeasuredPaths(ctx, paths, wallTime, scale) {
    ctx.save();
    ctx.lineWidth = 1.65 / scale;
    ctx.lineCap = 'round';
    ctx.strokeStyle = PATH_COLOR;
    for (const path of paths) {
      for (let i = 1; i < path.length; i += 1) {
        const from = path[i - 1];
        const to = path[i];
        const life = 1 - (wallTime - to.time) / PATH_MS;
        const distance = Math.hypot(to.x - from.x, to.y - from.y);
        if (life <= 0 || !to.join || distance < 1) continue;
        ctx.globalAlpha = Math.min(.78, life * .78);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        if (i === path.length - 1 && distance * scale > 15) {
          const angle = Math.atan2(to.y - from.y, to.x - from.x);
          const x = (from.x + to.x) / 2;
          const y = (from.y + to.y) / 2;
          const wing = 5 / scale;
          ctx.moveTo(x - Math.cos(angle - .65) * wing, y - Math.sin(angle - .65) * wing);
          ctx.lineTo(x, y);
          ctx.lineTo(x - Math.cos(angle + .65) * wing, y - Math.sin(angle + .65) * wing);
        }
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  emit(id, x, y, angle, time, size) {
    if (this.particleCount === MAX_PARTICLES) return;
    const particle = this.particles[this.particleCount++];
    const spread = (Math.random() - 0.5) * 2.5;
    const drift = size * (8 + Math.random() * 30);
    particle.x = x - Math.sin(angle) * spread * size * 6;
    particle.y = y + Math.cos(angle) * spread * size * 6;
    particle.vx = Math.cos(angle + spread) * drift;
    particle.vy = Math.sin(angle + spread) * drift;
    particle.id = id;
    particle.time = time;
    particle.life = 0.65 + Math.random() * 1.05;
    particle.size = size * (1.05 + Math.random() * 2.6);
  }

  updateTracks(markers, time, dt, effects, layout, quality) {
    if (dt <= 0) return;
    let seen = 0;
    const size = layout.fxScale;
    for (const marker of markers) {
      const { id, x, y, angle } = marker;
      const track = this.tracks[id];
      seen |= 1 << id;
      const previousAngle = track.previousAngle;
      track.previousAngle = angle;
      if (previousAngle === null) {
        track.count = 0;
        track.beforeX = x;
        track.beforeY = y;
        track.turnSince = time;
        continue;
      }

      const deltaAngle = angle - previousAngle;
      const wrapped = Math.atan2(Math.sin(deltaAngle), Math.cos(deltaAngle));
      if (Math.abs(wrapped) >= 0.006) {
        track.turnAngle += wrapped;
        track.turnSince = time;
      }
      if (Math.abs(track.turnAngle) >= 0.20) {
        if (effects.rings && time - track.lastTurn > 0.39) {
          track.lastTurn = time;
          track.direction = Math.sign(track.turnAngle);
          track.turnAngle = 0;
        }
      }
      if (time - track.turnSince > 0.17 || !effects.rings) track.turnAngle = 0;
      const dx = x - track.beforeX;
      const dy = y - track.beforeY;
      const distance = Math.hypot(dx, dy);
      if (distance > 110 * size || !effects.trails) {
        track.count = 0;
        track.carry = 0;
        track.beforeX = x;
        track.beforeY = y;
        continue;
      }
      if (distance < 3.5 * size) continue;

      if (track.count === 0 || time - track.time[(track.head + TRAIL_POINTS - 1) % TRAIL_POINTS] > 0.22) {
        track.count = 0;
        trailPoint(track, track.beforeX, track.beforeY, time);
      }
      trailPoint(track, x, y, time);
      const backwards = Math.atan2(-dy, -dx);
      const spacing = size * (8 - 3 * quality);
      for (let step = spacing - track.carry; step <= distance; step += spacing) {
        const ratio = step / distance;
        this.emit(id, track.beforeX + dx * ratio, track.beforeY + dy * ratio, backwards, time, size);
        if (quality > 0.65) this.emit(id, track.beforeX + dx * ratio, track.beforeY + dy * ratio, backwards, time, size);
      }
      track.carry = (track.carry + distance) % spacing;
      track.beforeX = x;
      track.beforeY = y;
    }
    for (let id = 0; id < this.tracks.length; id += 1) {
      const track = this.tracks[id];
      if (!(seen & (1 << id))) {
        track.previousAngle = null;
        track.carry = 0;
        track.turnAngle = 0;
        track.lastTurn = -Infinity;
      }
    }
  }

  drawTrails(ctx, time, size, enabled) {
    for (let i = 0; i < this.particleCount;) {
      const particle = this.particles[i];
      const age = time - particle.time;
      if (!enabled || age >= particle.life) {
        const last = --this.particleCount;
        this.particles[i] = this.particles[last];
        this.particles[last] = particle;
        continue;
      }
      const progress = age / particle.life;
      const alpha = (1 - progress) ** 1.5;
      const drift = age * (1 - progress * 0.39);
      const x = particle.x + particle.vx * drift;
      const y = particle.y + particle.vy * drift;
      const radius = particle.size * (1 - progress * 0.6);
      ctx.globalAlpha = alpha * 0.9;
      ctx.drawImage(this.flares[particle.id], x - radius * 5, y - radius * 5, radius * 10, radius * 10);
      ctx.fillStyle = PALETTE[particle.id].color;
      ctx.globalAlpha = alpha;
      ctx.fillRect(x - radius * 0.38, y - radius * 0.38, radius * 0.76, radius * 0.76);
      i += 1;
    }
    ctx.globalAlpha = 1;
    if (!enabled) return;
    ctx.lineCap = 'round';
    for (let id = 0; id < this.tracks.length; id += 1) {
      const trail = this.tracks[id];
      const end = trail.head + TRAIL_POINTS - 1;
      let from = end + 1 - trail.count;
      while (from < end) {
        const index = from % TRAIL_POINTS;
        const age = time - trail.time[index];
        if (age >= TRAIL_SECONDS) {
          from += 1;
          continue;
        }
        const band = Math.max(0, Math.floor(age / TRAIL_SECONDS * TRAIL_BANDS));
        ctx.beginPath();
        ctx.moveTo(trail.x[index], trail.y[index]);
        do {
          from += 1;
          const to = from % TRAIL_POINTS;
          ctx.lineTo(trail.x[to], trail.y[to]);
        } while (from < end && time - trail.time[from % TRAIL_POINTS] >= band * TRAIL_SECONDS / TRAIL_BANDS);
        const alpha = (1 - (band + 0.5) / TRAIL_BANDS) ** 1.7;
        ctx.strokeStyle = tint(id, alpha * 0.21);
        ctx.lineWidth = size * (4 + alpha * 13);
        ctx.stroke();
        ctx.lineWidth = size * (0.8 + alpha * 1.7);
        ctx.strokeStyle = tint(id, alpha * 0.58);
        ctx.stroke();
      }
    }
    ctx.lineCap = 'butt';
  }

  drawVector(ctx, marker, time) {
    const { id, angle } = marker;
    ctx.save();
    ctx.rotate(angle);
    const gradient = ctx.createLinearGradient(10, 0, 94, 0);
    gradient.addColorStop(0, tint(id, 0.06));
    gradient.addColorStop(0.66, tint(id, 0.28));
    gradient.addColorStop(1, tint(id, 0.72));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(14, -14);
    ctx.lineTo(95, 0);
    ctx.lineTo(14, 14);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = tint(id, 0.45);
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(15, -14);
    ctx.lineTo(95, 0);
    ctx.lineTo(15, 14);
    ctx.moveTo(27, 0);
    ctx.lineTo(89, 0);
    ctx.stroke();
    const pulse = (time * 36 + id * 9) % 25;
    for (let i = 0; i < 2; i += 1) {
      const x = 32 + pulse + i * 25;
      ctx.strokeStyle = tint(id, 0.76 * (1 - (x - 32) / 70));
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(x, -5);
      ctx.lineTo(x + 6, 0);
      ctx.lineTo(x, 5);
      ctx.stroke();
    }
    ctx.strokeStyle = PALETTE[id].color;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(81, -8);
    ctx.lineTo(96, 0);
    ctx.lineTo(81, 8);
    ctx.stroke();
    ctx.restore();
  }

  drawRings(ctx, marker, time, selected) {
    const { id, angle } = marker;
    const track = this.tracks[id];
    const wave = (time - track.lastTurn) / 0.88;
    const energy = wave < 1 ? (1 - wave) ** 1.6 : 0;
    const color = PALETTE[id].color;
    ctx.strokeStyle = tint(id, 0.38 + energy * 0.45);
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(0, 0, 26, 0, TAU);
    ctx.stroke();
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = tint(id, selected ? 0.50 : 0.31);
    ctx.beginPath();
    ctx.arc(0, 0, 39, 0, TAU);
    ctx.stroke();

    const spin = angle * 0.6 + time * 0.34 * track.direction;
    ctx.beginPath();
    for (let i = 0; i < 24; i += 1) {
      const theta = i / 24 * TAU + spin;
      const dx = Math.cos(theta);
      const dy = Math.sin(theta);
      const inner = i % 3 === 0 ? 29 : 33;
      ctx.moveTo(dx * inner, dy * inner);
      ctx.lineTo(dx * 36, dy * 36);
    }
    ctx.strokeStyle = tint(id, 0.52 + energy * 0.3);
    ctx.stroke();

    ctx.lineWidth = 2.3 + energy * 0.9;
    ctx.lineCap = 'round';
    ctx.strokeStyle = tint(id, 0.72 + energy * 0.26);
    for (let i = 0; i < 3; i += 1) {
      const start = spin + i * TAU / 3;
      ctx.beginPath();
      ctx.arc(0, 0, 41.5, start, start + 0.48 + energy * 0.42);
      ctx.stroke();
    }
    ctx.lineCap = 'butt';

    if (energy > 0) {
      for (let pulse = 0; pulse < 2; pulse += 1) {
        const radius = 32 + wave * 41 + pulse * 7;
        ctx.strokeStyle = tint(id, energy * (pulse ? 0.36 : 0.85));
        ctx.lineWidth = pulse ? 0.9 : 2.5;
        ctx.beginPath();
        ctx.arc(0, 0, radius, 0, TAU);
        ctx.stroke();
      }
      ctx.fillStyle = color;
      ctx.globalAlpha = energy;
      const theta = angle + wave * 3 * track.direction;
      const orbit = 42 + wave * 25;
      const x = Math.cos(theta) * orbit;
      const y = Math.sin(theta) * orbit;
      ctx.beginPath();
      ctx.arc(x, y, 2.7, 0, TAU);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  drawMarker(ctx, marker, time, effects, selected, size, projection) {
    const { id, x, y, angle } = marker;
    if (projection && marker.visible === false) return;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(size, size);
    if (marker.visible === false) {
      ctx.rotate(angle);
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = .66;
      ctx.strokeStyle = PALETTE[id].color;
      ctx.lineWidth = 1.7;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(15, 0);
      ctx.lineTo(-8, -10);
      ctx.lineTo(-8, 10);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
      return;
    }
    if (!projection) {
      const glow = effects.rings ? 72 : 35;
      ctx.globalAlpha = selected ? 0.73 : 0.54;
      ctx.drawImage(this.flares[id], -glow, -glow, glow * 2, glow * 2);
      ctx.globalAlpha = 1;
    }
    if (effects.vectors) this.drawVector(ctx, marker, time);
    if (effects.rings) this.drawRings(ctx, marker, time, selected);

    if (!projection) {
      ctx.rotate(angle);
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = PALETTE[id].color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(17, 0);
      ctx.lineTo(-10, -11);
      ctx.lineTo(-6, 0);
      ctx.lineTo(-10, 11);
      ctx.closePath();
      ctx.fillStyle = '#0c1426';
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#eefaff';
      ctx.beginPath();
      ctx.arc(0, 0, 2.6, 0, TAU);
      ctx.fill();
    }
    ctx.restore();
  }

  drawLabel(ctx, marker, layout, selected) {
    const { id } = marker;
    const { width, height, scale, offsetX, offsetY, fxScale } = layout;
    const x = marker.x * scale + offsetX;
    const y = marker.y * scale + offsetY;
    const right = x < width - 104;
    const above = y > 76;
    const unit = clamp(fxScale * scale, 0.56, 1.12);
    const leader = 29 * unit;
    const size = selected ? 47 : 42;
    const labelX = x + (right ? leader + 10 : -leader - 10 - size);
    const labelY = Math.min(height - 22, y + (above ? -leader - 12 : leader - 7));
    const labelMiddle = labelY + 9;
    ctx.save();
    if (marker.visible === false) {
      ctx.globalAlpha = .68;
      ctx.setLineDash([3, 3]);
    }
    ctx.beginPath();
    ctx.moveTo(x + (right ? 16 : -16) * unit, y + (above ? -15 : 15) * unit);
    ctx.lineTo(x + (right ? leader : -leader), labelMiddle);
    ctx.lineTo(right ? labelX : labelX + size, labelMiddle);
    ctx.strokeStyle = tint(id, selected ? 0.7 : 0.44);
    ctx.lineWidth = 1;
    ctx.stroke();
    roundBox(ctx, labelX, labelY, size, 18, 3);
    ctx.fillStyle = selected ? '#102436' : '#0c1426';
    ctx.fill();
    ctx.strokeStyle = tint(id, selected ? 0.62 : 0.29);
    ctx.stroke();
    ctx.font = `${selected ? 700 : 500} 10px ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = selected ? '#edfdff' : PALETTE[id].color;
    ctx.fillText(`R–0${id}`, labelX + size / 2, labelMiddle + 0.5);
    ctx.restore();
  }

  render({ field, markers, time, dt, effects, selected, projection, quality = 1,
    live = false, depthImage, paths, wallTime = 0 }) {
    const detail = clamp(quality, 0, 1);
    const layout = this.resize(field, projection, detail, live);
    if (!layout) return;
    this.visibleMarkers = markers;
    const activeMarkers = live ? markers.filter((marker) => marker.visible) : markers;
    const ctx = this.ctx;
    const { dpr, offsetX, offsetY, scale, fxScale, width, height } = layout;
    const [left, top, right, bottom] = field.rect;
    this.updateTracks(activeMarkers, time, dt, live ? { ...effects, trails: false } : effects, layout, detail);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (projection) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);
    } else {
      ctx.drawImage(this.backdrop, 0, 0, width, height);
    }
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.beginPath();
    ctx.rect(left, top, right - left, bottom - top);
    ctx.clip();
    if (live && depthImage && !projection) {
      ctx.globalAlpha = .76;
      ctx.drawImage(depthImage, 0, 0, field.width, field.height);
      ctx.globalAlpha = 1;
    }
    ctx.globalCompositeOperation = 'lighter';
    if (live) {
      if (effects.trails && paths) this.drawMeasuredPaths(ctx, paths, wallTime, scale);
    } else {
      this.drawTrails(ctx, time, fxScale, effects.trails);
    }
    for (const marker of markers) {
      this.drawMarker(ctx, marker, time, effects, marker.id === selected, fxScale, projection);
    }
    if (projection) {
      // Keep ArUco centers dark so the projected overlay preserves tag contrast.
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = '#000';
      ctx.beginPath();
      const maskRadius = 32 * fxScale;
      for (const { x, y } of activeMarkers) {
        ctx.moveTo(x + maskRadius, y);
        ctx.arc(x, y, maskRadius, 0, TAU);
      }
      ctx.fill();
    }
    ctx.restore();
    if (!projection) {
      for (const marker of markers) this.drawLabel(ctx, marker, layout, marker.id === selected);
    }
  }
}
