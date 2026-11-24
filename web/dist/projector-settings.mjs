import { requestJson } from './live-feed.mjs';

const appearanceLabels = {
  raw: 'Исходная карта', balanced: 'Сбалансированная', smooth: 'Сглаженная',
};
const fillLabels = {
  conservative: 'Локальное заполнение', surface: 'Восстановление поверхности',
};
const resolutionValue = ({ width, height }) => `${width}x${height}`;
const resolutionLabel = ({ width, height }) => `${width} × ${height}`;

export function projectorSettings() {
  const byId = (id) => document.getElementById(id);
  const open = byId('projector-settings');
  const dialog = byId('projector-dialog');
  const fields = byId('projector-fields');
  const display = byId('display-mode');
  const resolution = byId('display-resolution');
  const appearance = byId('projection-appearance');
  const fill = byId('projection-fill');
  const fullscreen = byId('display-fullscreen');
  const apply = byId('projector-apply');
  const message = byId('projector-message');
  let origin;
  let state;
  let session;

  function setEnabled(enabled) {
    fields.disabled = !enabled;
    apply.disabled = !enabled;
  }

  function updateFullscreen() {
    fullscreen.disabled = resolution.selectedOptions[0]?.dataset.windowOnly === 'true';
    if (fullscreen.disabled) fullscreen.checked = false;
  }

  function showResolutions(preferred) {
    const available = state.displays.find((entry) => entry.id === display.value)?.modes ?? [];
    const options = available.map((entry) => new Option(resolutionLabel(entry), resolutionValue(entry)));
    if (options.length === 0 || (preferred && !options.some((option) => option.value === preferred))) {
      const reference = state.calibration;
      const fallback = new Option(`${resolutionLabel(reference)} (окно)`, resolutionValue(reference));
      fallback.dataset.windowOnly = 'true';
      if (!options.some((option) => option.value === fallback.value)) options.unshift(fallback);
    }
    resolution.replaceChildren(...options);
    if (options.some((option) => option.value === preferred)) resolution.value = preferred;
    updateFullscreen();
  }

  function render(next) {
    state = next;
    const { settings, displays = [], modes, calibration } = next;
    if (!next.controllable || !settings || displays.length === 0) {
      message.textContent = next.presentation_error || 'Настройки физического проектора доступны при запуске start.cmd.';
      setEnabled(false);
      return;
    }
    display.replaceChildren(...displays.map((entry) => {
      const size = entry.width && entry.height ? ` · ${resolutionLabel(entry)}` : '';
      return new Option(`${entry.name || entry.id}${size}`, entry.id);
    }));
    if (displays.some((entry) => entry.id === settings.display)) display.value = settings.display;
    fullscreen.checked = settings.fullscreen;
    showResolutions(resolutionValue(settings));
    appearance.replaceChildren(...modes.appearance.map((value) => new Option(appearanceLabels[value] || value, value)));
    fill.replaceChildren(...modes.fill.map((value) => new Option(fillLabels[value] || value, value)));
    appearance.value = settings.appearance;
    fill.value = settings.fill;
    message.textContent = next.presentation_error || `Калибровка ${resolutionLabel(calibration)} · шумоподавление ${calibration.noise_mm} мм`;
    setEnabled(true);
  }

  async function read(signal) {
    setEnabled(false);
    try {
      const next = await requestJson(new URL('/api/v1/projection', origin), { signal });
      if (!signal.aborted) render(next);
    } catch (error) {
      if (!signal.aborted) message.textContent = error.status === 501
        ? 'Настройки физического проектора доступны при запуске start.cmd.'
        : 'Служба настроек проектора недоступна.';
    }
  }

  open.addEventListener('click', () => {
    session?.abort();
    session = new AbortController();
    state = undefined;
    setEnabled(false);
    message.textContent = 'Загрузка настроек…';
    dialog.showModal();
    if (origin !== location.origin) {
      message.textContent = 'Откройте поле роя по адресу службы AR, чтобы управлять проектором.';
    } else {
      read(session.signal);
    }
  });
  dialog.addEventListener('close', () => session?.abort());
  byId('projector-close').addEventListener('click', () => dialog.close());
  display.addEventListener('change', () => showResolutions());
  resolution.addEventListener('change', updateFullscreen);
  byId('projector-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!state?.controllable || fields.disabled) return;
    const [width, height] = resolution.value.split('x').map(Number);
    const payload = {
      expected_revision: state.revision, display: display.value, width, height,
      fullscreen: fullscreen.checked, appearance: appearance.value, fill: fill.value,
    };
    const { signal } = session;
    setEnabled(false);
    message.textContent = 'Применение настроек…';
    try {
      const result = await requestJson(new URL('/api/v1/projection', origin), {
        signal, timeout: 11500, method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!signal.aborted) {
        render(result);
        if (!result.presentation_error) message.textContent = `Настройки применены · ${resolutionLabel(result.settings)}`;
      }
    } catch (error) {
      if (signal.aborted) return;
      if (error.status === 409) await read(signal);
      if (signal.aborted) return;
      message.textContent = error.message || 'Настройки не применились.';
      if (error.status !== 409) setEnabled(true);
    }
  });

  return (next) => {
    if (origin !== next && dialog.open) dialog.close();
    origin = next;
    open.hidden = !origin;
  };
}
