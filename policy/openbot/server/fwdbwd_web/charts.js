// A small canvas line chart: shared x, one y axis, shaded bands, a hover crosshair with a
// tooltip, an optional playhead, click/drag to seek. Colours come from CSS tokens, so
// light and dark mode each use their own palette steps.
import { h } from './lib.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

function bisect(xs, x) {
  let lo = 0, hi = xs.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(xs[lo - 1] - x) < Math.abs(xs[lo] - x)) return lo - 1;
  return lo;
}

/**
 * opts: { x: number[], series: [{label, color: '--token', values, dash, width}],
 *         bands: [{from, to, color: '--token', label}], yMin, yMax, height,
 *         xFormat(x) (tooltip), xTick(x) (axis, default xFormat), yFormat(y), markers: [{x, label}], onSeek(x), zeroLine }
 */
export function lineChart(opts) {
  const wrap = h('div', { class: 'chart' });
  const canvas = h('canvas', { role: 'img', 'aria-label': opts.ariaLabel || 'chart' });
  const tip = h('div', { class: 'tip', hidden: true });
  wrap.append(canvas, tip);
  let o = { height: 220, xFormat: (x) => x.toFixed(2), yFormat: (y) => String(Math.round(y)), ...opts };
  let playhead = null, hoverIdx = null, base = null, dragging = false;
  const pad = { l: 46, r: 12, t: 10, b: 24 };

  function domain() {
    const xs = o.x;
    const x0 = o.xMin ?? (xs.length ? xs[0] : 0);
    const x1 = o.xMax ?? (xs.length ? xs[xs.length - 1] : 1);
    let y0 = o.yMin, y1 = o.yMax;
    if (y0 === undefined || y1 === undefined) {
      let lo = Infinity, hi = -Infinity;
      for (const s of o.series) for (const v of s.values) if (v !== null && v !== undefined && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      const m = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
      y0 = y0 ?? lo - m; y1 = y1 ?? hi + m;
    }
    return { x0, x1: x1 === x0 ? x0 + 1 : x1, y0, y1 };
  }

  function geometry() {
    const w = wrap.clientWidth || 600, hgt = o.height;
    const d = domain();
    const sx = (x) => pad.l + ((x - d.x0) / (d.x1 - d.x0)) * (w - pad.l - pad.r);
    const sy = (y) => pad.t + (1 - (y - d.y0) / (d.y1 - d.y0)) * (hgt - pad.t - pad.b);
    const ix = (px) => d.x0 + ((px - pad.l) / (w - pad.l - pad.r)) * (d.x1 - d.x0);
    return { w, hgt, d, sx, sy, ix };
  }

  function drawBase() {
    const dpr = window.devicePixelRatio || 1;
    const g = geometry();
    canvas.width = Math.round(g.w * dpr);
    canvas.height = Math.round(g.hgt * dpr);
    canvas.style.height = g.hgt + 'px';
    base = document.createElement('canvas');
    base.width = canvas.width; base.height = canvas.height;
    const c = base.getContext('2d');
    c.scale(dpr, dpr);
    const { sx, sy, d } = g;
    // bands
    for (const b of o.bands || []) {
      c.fillStyle = css(b.color);
      c.fillRect(sx(b.from), pad.t, Math.max(1, sx(b.to) - sx(b.from)), g.hgt - pad.t - pad.b);
    }
    // grid + axes
    c.font = '11px system-ui, sans-serif';
    c.fillStyle = css('--muted');
    c.strokeStyle = css('--grid');
    c.lineWidth = 1;
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const t of niceTicks(d.y0, d.y1, 4)) {
      const y = Math.round(sy(t)) + 0.5;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(g.w - pad.r, y); c.stroke();
      c.fillText(o.yFormat(t), pad.l - 6, y);
    }
    if (o.zeroLine && d.y0 < 0 && d.y1 > 0) {
      c.strokeStyle = css('--axis');
      const y = Math.round(sy(0)) + 0.5;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(g.w - pad.r, y); c.stroke();
    }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (const t of niceTicks(d.x0, d.x1, Math.max(3, Math.floor(g.w / 110)))) {
      const x = sx(t);
      if (x < pad.l - 1 || x > g.w - pad.r + 1) continue;
      const label = (o.xTick || o.xFormat)(t);
      const half = c.measureText(label).width / 2;
      if (x - half < 0 || x + half > g.w) continue;
      c.fillText(label, x, g.hgt - pad.b + 6);
    }
    // markers
    for (const m of o.markers || []) {
      const x = Math.round(sx(m.x)) + 0.5;
      c.strokeStyle = css('--axis'); c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, g.hgt - pad.b); c.stroke();
      c.setLineDash([]);
      c.fillStyle = css('--text-2'); c.textAlign = 'left'; c.textBaseline = 'top';
      c.fillText(m.label, x + 4, pad.t + 2);
    }
    // series
    for (const s of o.series) {
      c.strokeStyle = css(s.color);
      c.lineWidth = s.width || 2;
      c.lineJoin = 'round'; c.lineCap = 'round';
      c.setLineDash(s.dash ? [5, 4] : []);
      c.beginPath();
      let pen = false;
      for (let i = 0; i < o.x.length; i++) {
        const v = s.values[i];
        if (v === null || v === undefined || !isFinite(v)) { pen = false; continue; }
        const x = sx(o.x[i]), y = sy(v);
        if (pen) c.lineTo(x, y); else { c.moveTo(x, y); pen = true; }
      }
      c.stroke();
      if (s.dots) {
        c.fillStyle = css(s.color);
        for (let i = 0; i < o.x.length; i++) {
          const v = s.values[i];
          if (v === null || v === undefined) continue;
          c.beginPath(); c.arc(sx(o.x[i]), sy(v), 2.5, 0, 7); c.fill();
        }
      }
    }
    c.setLineDash([]);
    render();
  }

  function render() {
    if (!base) return;
    const dpr = window.devicePixelRatio || 1;
    const c = canvas.getContext('2d');
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.drawImage(base, 0, 0);
    c.scale(dpr, dpr);
    const g = geometry();
    const line = (x, color, width) => {
      c.strokeStyle = color; c.lineWidth = width;
      c.beginPath(); c.moveTo(x, pad.t); c.lineTo(x, g.hgt - pad.b); c.stroke();
    };
    if (playhead !== null) line(Math.round(g.sx(playhead)) + 0.5, css('--text'), 1.5);
    if (hoverIdx !== null && hoverIdx >= 0) {
      const x = g.sx(o.x[hoverIdx]);
      line(Math.round(x) + 0.5, css('--axis'), 1);
      for (const s of o.series) {
        const v = s.values[hoverIdx];
        if (v === null || v === undefined) continue;
        c.fillStyle = css('--surface');
        c.beginPath(); c.arc(x, g.sy(v), 5, 0, 7); c.fill();
        c.fillStyle = css(s.color);
        c.beginPath(); c.arc(x, g.sy(v), 3.5, 0, 7); c.fill();
      }
    }
  }

  function showTip(ev) {
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const g = geometry();
    hoverIdx = bisect(o.x, g.ix(px));
    if (hoverIdx < 0) return;
    const rows = o.series
      .filter((s) => s.values[hoverIdx] !== null && s.values[hoverIdx] !== undefined)
      .map((s) => h('div', {}, h('span', { class: 'sw line', style: { background: css(s.color) } }), s.label + ' ', h('b', {}, o.yFormat(s.values[hoverIdx]))));
    const extra = o.tipExtra ? o.tipExtra(hoverIdx) : null;
    tip.replaceChildren(h('div', { class: 'muted' }, o.xFormat(o.x[hoverIdx])), ...rows, extra || '');
    tip.hidden = false;
    const tx = g.sx(o.x[hoverIdx]);
    tip.style.left = Math.min(Math.max(0, tx + 12), g.w - tip.offsetWidth - 4) + 'px';
    tip.style.top = pad.t + 'px';
    render();
  }

  function seek(ev) {
    if (!o.onSeek) return;
    const rect = canvas.getBoundingClientRect();
    o.onSeek(Math.min(Math.max(geometry().ix(ev.clientX - rect.left), domain().x0), domain().x1));
  }

  canvas.addEventListener('pointermove', (ev) => { showTip(ev); if (dragging) seek(ev); });
  canvas.addEventListener('pointerleave', () => { hoverIdx = null; tip.hidden = true; dragging = false; render(); });
  canvas.addEventListener('pointerdown', (ev) => { if (o.onSeek) { dragging = true; canvas.setPointerCapture(ev.pointerId); seek(ev); } });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  if (o.onSeek) canvas.style.cursor = 'pointer';

  let pending = false;
  const ro = new ResizeObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; drawBase(); });
  });
  ro.observe(wrap);
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const onScheme = () => drawBase();
  mq.addEventListener('change', onScheme);

  wrap.update = (next) => { o = { ...o, ...next }; drawBase(); };
  wrap.setPlayhead = (x) => { playhead = x; render(); };
  wrap.destroy = () => { ro.disconnect(); mq.removeEventListener('change', onScheme); };
  return wrap;
}

// Legend for >= 2 series: swatch + name, text in text tokens.
export function legend(items) {
  return h('div', { class: 'chart-legend' }, items.map((it) => h('span', {},
    it.band
      ? h('span', { class: 'sw', style: { background: css(it.color) } })
      : h('span', { class: it.dash ? 'sw dash' : 'sw line', style: it.dash ? { borderColor: css(it.color) } : { background: css(it.color) } }),
    it.label)));
}
