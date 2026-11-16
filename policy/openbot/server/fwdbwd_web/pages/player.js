// Player: a recording's frames at their recorded timing, with steering, throttle and cmd
// per frame, and a timeline of the whole recording (reverse stretches shaded; for an
// Autopilot Session log also the model's prediction and your takeovers).
import { h, clear, api, fileUrl, fmtS, recDate } from '../lib.js';
import { lineChart, legend } from '../charts.js';

const CMD = { '-1': ['◀ LEFT', 'var(--series-1)'], 0: ['▲ STRAIGHT', 'var(--text-2)'], 1: ['RIGHT ▶', 'var(--series-1)'], 2: ['▼ REVERSE', 'var(--series-2)'] };
const BACK = { inbox: ['#/inbox', 'Inbox'], sessions: ['#/sessions', 'Sessions'], runs: ['#/runs', 'Autopilot runs'] };

function spans(t, test) {
  const out = [];
  let start = null;
  for (let i = 0; i < t.length; i++) {
    if (test(i) && start === null) start = t[i];
    if (start !== null && (!test(i) || i === t.length - 1)) {
      out.push([start, t[i]]);
      start = null;
    }
  }
  return out;
}

// Steering as a semicircle needle; the prediction as a dashed second needle.
function steeringGauge() {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '-60 -58 120 66');
  svg.style.width = '100%';
  svg.innerHTML = `
    <path d="M -50 0 A 50 50 0 0 1 50 0" fill="none" stroke="var(--border)" stroke-width="8" stroke-linecap="round"/>
    <line x1="0" y1="-54" x2="0" y2="-44" stroke="var(--muted)" stroke-width="1.5"/>
    <line class="pred" x1="0" y1="0" x2="0" y2="-46" stroke="var(--series-1)" stroke-width="2" stroke-dasharray="4 3" opacity="0"/>
    <line class="cur" x1="0" y1="0" x2="0" y2="-46" stroke="var(--series-1)" stroke-width="3" stroke-linecap="round"/>
    <circle r="4" fill="var(--text)"/>`;
  const set = (line, v) => {
    const a = (Math.max(-255, Math.min(255, v)) / 255) * 90 * Math.PI / 180;
    line.setAttribute('x2', (46 * Math.sin(a)).toFixed(1));
    line.setAttribute('y2', (-46 * Math.cos(a)).toFixed(1));
  };
  svg.set = (v, pred) => {
    set(svg.querySelector('.cur'), v);
    const p = svg.querySelector('.pred');
    if (pred === null || pred === undefined) p.setAttribute('opacity', 0);
    else { p.setAttribute('opacity', 1); set(p, pred); }
  };
  return svg;
}

// Throttle as a bar from the centre: up forward, down reverse.
function throttleBar() {
  const bar = h('div', { style: { position: 'relative', height: '120px', width: '34px', margin: '0 auto', background: 'var(--surface-2)', borderRadius: '6px', overflow: 'hidden' } });
  const fill = h('div', { style: { position: 'absolute', left: '0', right: '0', background: 'var(--series-2)', borderRadius: '4px' } });
  const pred = h('div', { style: { position: 'absolute', left: '-2px', right: '-2px', height: '0', borderTop: '2px dashed var(--text)', display: 'none' } });
  const mid = h('div', { style: { position: 'absolute', left: '0', right: '0', top: '50%', borderTop: '1px solid var(--axis)' } });
  bar.append(fill, mid, pred);
  bar.set = (v, p) => {
    const f = Math.max(-1, Math.min(1, v / 255)) * 50;
    fill.style.top = (f >= 0 ? 50 - f : 50) + '%';
    fill.style.height = Math.abs(f) + '%';
    if (p === null || p === undefined) pred.style.display = 'none';
    else { pred.style.display = ''; pred.style.top = (50 - Math.max(-1, Math.min(1, p / 255)) * 50) + '%'; }
  };
  return bar;
}

export async function render(view, params) {
  const path = params.path;
  const [backHref, backLabel] = BACK[params.back] || ['javascript:history.back()', 'Back'];
  view.append(h('div', { class: 'page-head' }, h('a', { href: backHref }, '← ' + backLabel)));
  let data;
  try {
    data = await api('recording?path=' + encodeURIComponent(path));
  } catch (e) {
    view.append(h('div', { class: 'card note bad' }, e.message));
    return;
  }
  const f = data.frames, s = data.summary, n = f.t.length;
  const isRun = !!f.source;
  if (!n) {
    view.append(h('div', { class: 'card empty' }, 'No frames with images in this recording.'));
    return;
  }

  // --- layout
  const img = h('img', { alt: 'frame' });
  const osd = h('div', { class: 'osd' });
  const srcBadge = h('div', { class: 'src' });
  const gauge = steeringGauge();
  const tbar = throttleBar();
  const steerVal = h('div', { class: 'val' });
  const steerPred = h('div', { class: 'muted mono' });
  const thrVal = h('div', { class: 'val' });
  const thrPred = h('div', { class: 'muted mono' });
  const cmdBadge = h('div', { class: 'cmd-badge' });
  const timeEl = h('span', { class: 'time' });
  const playBtn = h('button', { class: 'primary', style: { minWidth: '76px' } }, '▶ Play');
  const speed = h('select', {}, [0.25, 0.5, 1, 2, 4].map((v) => h('option', { value: v, selected: v === 1 }, v + '×')));
  const loop = h('input', { type: 'checkbox' });

  const bands = spans(f.t, (i) => f.throttle[i] < 0).map(([a, b]) => ({ from: a, to: b, color: '--band-reverse' }));
  const series = [
    { label: 'steering', color: '--series-1', values: f.steering },
    { label: 'throttle', color: '--series-2', values: f.throttle },
  ];
  const legendItems = [
    { label: 'steering', color: '--series-1' },
    { label: 'throttle', color: '--series-2' },
  ];
  if (isRun) {
    series.push({ label: 'model steering', color: '--series-1', values: f.pred_s, dash: true, width: 1.5 });
    series.push({ label: 'model throttle', color: '--series-2', values: f.pred_t, dash: true, width: 1.5 });
    legendItems.push({ label: 'model steering', color: '--series-1', dash: true }, { label: 'model throttle', color: '--series-2', dash: true });
    for (const [a, b] of spans(f.t, (i) => f.source[i] === 'human')) bands.push({ from: a, to: b, color: '--band-takeover' });
    legendItems.push({ label: 'you drove (takeover)', color: '--band-takeover', band: true });
  }
  legendItems.push({ label: 'reversing', color: '--band-reverse', band: true });

  let idx = 0, playing = false, raf = null, clockStart = 0, tStart = 0;
  const cache = new Map();
  const preload = (from) => {
    for (let i = from; i < Math.min(n, from + 60); i++) {
      if (!cache.has(i)) { const im = new Image(); im.src = fileUrl(s.path + '/images/' + f.img[i]); cache.set(i, im); }
    }
    if (cache.size > 600) for (const k of cache.keys()) { if (k < from - 120 || k > from + 300) cache.delete(k); if (cache.size < 400) break; }
  };

  const chart = lineChart({
    x: f.t, series, bands, yMin: -260, yMax: 260, height: 210, zeroLine: true,
    xFormat: (x) => x.toFixed(1) + ' s', yFormat: (y) => String(Math.round(y)),
    ariaLabel: 'steering and throttle over the recording',
    tipExtra: (i) => h('div', { class: 'muted' }, `cmd ${f.cmd[i]}` + (isRun ? ` · ${f.source[i]}` : '')),
    onSeek: (x) => { pause(); show(nearest(x)); },
  });

  function nearest(t) {
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (f.t[m] <= t) lo = m; else hi = m - 1; }
    return lo;
  }

  function show(i) {
    idx = Math.max(0, Math.min(n - 1, i));
    preload(idx);
    const im = cache.get(idx);
    img.src = im ? im.src : fileUrl(s.path + '/images/' + f.img[idx]);
    const st = f.steering[idx], th = f.throttle[idx];
    const ps = isRun ? f.pred_s[idx] : null, pt = isRun ? f.pred_t[idx] : null;
    gauge.set(st, ps);
    tbar.set(th, pt);
    steerVal.textContent = (st > 0 ? '+' : '') + Math.round(st);
    thrVal.textContent = (th > 0 ? '+' : '') + Math.round(th);
    steerPred.textContent = ps !== null && ps !== undefined ? `model ${Math.round(ps)}` : '';
    thrPred.textContent = pt !== null && pt !== undefined ? `model ${Math.round(pt)}` : '';
    const [label, color] = CMD[f.cmd[idx]] || [`cmd ${f.cmd[idx]}`, 'var(--text)'];
    cmdBadge.textContent = label + ` (cmd ${f.cmd[idx]})`;
    cmdBadge.style.color = color;
    osd.textContent = `#${idx + 1}/${n}  ${f.img[idx]}`;
    if (isRun) {
      const src = f.source[idx];
      clear(srcBadge, h('span', { class: 'tag ' + (src === 'human' ? 'bad' : src === 'model' ? 'good' : 'warn') }, `${src.toUpperCase()} · ${f.direction[idx]} · ${f.model[idx] || '–'}`));
    }
    timeEl.textContent = `${f.t[idx].toFixed(2)} / ${f.t[n - 1].toFixed(2)} s`;
    chart.setPlayhead(f.t[idx]);
  }

  function tick(now) {
    if (!playing) return;
    const t = tStart + ((now - clockStart) / 1000) * Number(speed.value);
    if (t >= f.t[n - 1]) {
      if (loop.checked) { tStart = 0; clockStart = now; show(0); }
      else { show(n - 1); pause(); return; }
    } else {
      const i = nearest(t);
      if (i !== idx) show(i);
    }
    raf = requestAnimationFrame(tick);
  }
  function play() {
    if (idx >= n - 1) show(0);
    playing = true; playBtn.textContent = '❚❚ Pause';
    tStart = f.t[idx]; clockStart = performance.now();
    raf = requestAnimationFrame(tick);
  }
  function pause() {
    playing = false; playBtn.textContent = '▶ Play';
    cancelAnimationFrame(raf);
  }
  playBtn.onclick = () => (playing ? pause() : play());
  speed.onchange = () => { if (playing) { tStart = f.t[idx]; clockStart = performance.now(); } };
  const step = (d) => { pause(); show(idx + d); };

  const onKey = (e) => {
    if (e.target.matches('input, textarea, select')) return;
    if (e.key === ' ') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.key === 'ArrowRight') step(e.shiftKey ? 10 : 1);
    else if (e.key === 'ArrowLeft') step(e.shiftKey ? -10 : -1);
    else if (e.key === 'Home') step(-n);
    else if (e.key === 'End') step(n);
  };
  document.addEventListener('keydown', onKey);

  const flags = [];
  if (s.reason && !isRun) flags.push(h('span', { class: 'tag warn', title: 'split_direction.py parks it instead of splitting' }, `⚠ ${s.reason}`));
  view.append(
    h('div', { class: 'page-head' },
      h('h1', { class: 'mono' }, s.name),
      h('span', { class: 'muted' }, recDate(s.name)),
      h('span', { class: 'tag' }, fmtS(s.seconds)), h('span', { class: 'tag' }, `${n} frames`),
      h('span', { class: 'tag fwd' }, `fwd ${fmtS(s.fwd_s)}`), h('span', { class: 'tag bwd' }, `bwd ${fmtS(s.bwd_s)}`),
      flags, h('span', { class: 'spacer' }), h('span', { class: 'muted mono' }, s.path)),
    h('div', { class: 'player' },
      h('div', {},
        h('div', { class: 'screen' }, img, osd, srcBadge),
        h('div', { class: 'controls' },
          h('button', { onclick: () => step(-n), title: 'Home' }, '⏮'),
          h('button', { onclick: () => step(-1), title: '←, shift for 10' }, '◀'),
          playBtn,
          h('button', { onclick: () => step(1), title: '→, shift for 10' }, '▶'),
          h('button', { onclick: () => step(n), title: 'End' }, '⏭'),
          speed, h('label', { class: 'row dim' }, loop, 'loop'), timeEl,
          h('span', { class: 'muted', style: { marginLeft: 'auto', fontSize: '12px' } }, 'space play · ←/→ step · click the chart to jump')),
        h('div', { class: 'card' }, legend(legendItems), chart)),
      h('div', { class: 'gauges' },
        h('div', { class: 'gauge' }, h('div', { class: 'lbl' }, h('span', {}, 'steering'), h('span', {}, '−255 left … right +255')), gauge, h('div', { class: 'row' }, steerVal, steerPred)),
        h('div', { class: 'gauge' }, h('div', { class: 'lbl' }, h('span', {}, 'throttle'), h('span', {}, '+ fwd / − rev')),
          h('div', { class: 'row', style: { alignItems: 'center', gap: '16px' } }, tbar, h('div', {}, thrVal, thrPred))),
        h('div', { class: 'gauge' }, h('div', { class: 'lbl' }, 'cmd (the model\'s indicator input)'), cmdBadge))));

  show(0);
  return () => { pause(); document.removeEventListener('keydown', onKey); chart.destroy(); };
}
