// Small helpers shared by the pages: DOM building, API calls, toasts, formatting, job logs.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === undefined || c === null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el, ...children) {
  el.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  return el;
}

export async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  };
  const res = await fetch('/fb/api/' + path, opts);
  const text = await res.text();
  if (!res.ok) {
    let msg = text;
    try { msg = JSON.parse(text).error || text; } catch (e) { /* plain text */ }
    throw new Error(msg.replace(/^\d{3}: /, '') || res.statusText);
  }
  return text ? JSON.parse(text) : null;
}

export function toast(msg, kind = '') {
  const el = h('div', { class: 'toast ' + kind }, msg);
  document.getElementById('toasts').append(el);
  setTimeout(() => el.remove(), kind === 'bad' ? 9000 : 4500);
}

export async function attempt(fn, okMsg) {
  try {
    const r = await fn();
    if (okMsg) toast(okMsg, 'good');
    return r;
  } catch (e) {
    toast(e.message, 'bad');
    return undefined;
  }
}

export const fileUrl = (rel) => '/fb/file/' + rel.split('/').map(encodeURIComponent).join('/');
export const modelFileUrl = (rel) => '/fb/modelfile/' + rel.split('/').map(encodeURIComponent).join('/');

export function fmtS(s) {
  if (s === null || s === undefined) return '–';
  s = Math.round(s);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}m`;
}

export function fmtAgo(t) {
  if (!t) return '';
  const d = Date.now() / 1000 - t;
  if (d < 60) return 'just now';
  if (d < 3600) return Math.round(d / 60) + ' min ago';
  if (d < 86400) return Math.round(d / 3600) + ' h ago';
  return new Date(t * 1000).toLocaleDateString();
}

export function fmtDate(t) {
  return t ? new Date(t * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
}

export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
export const pct = (v) => (v === null || v === undefined ? '–' : (100 * v).toFixed(1) + '%');
export const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB';

// Recording names are yyyymmdd_hhmmss (optionally prefixed autopilot_).
export function recDate(name) {
  const m = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/.exec(name);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : '';
}

export function statusTag(status) {
  const kind = { done: 'good', running: '', queued: '', failed: 'bad', stopped: 'warn' }[status] ?? '';
  const icon = { done: '✓', running: '●', queued: '…', failed: '✕', stopped: '■' }[status] ?? '';
  return h('span', { class: 'tag ' + kind }, `${icon} ${status}`);
}

// A job's output as it grows; Keras progress bars use \r and \b, keep what a terminal shows.
export function cleanLog(text) {
  return text
    .split('\n')
    .map((line) => {
      const parts = line.split('\r');
      const s = parts.length > 1 ? (parts.filter((p) => p.length).pop() || '') : line;
      if (!s.includes('\x08')) return s;
      const out = [];
      for (const ch of s) { if (ch === '\x08') out.pop(); else out.push(ch); }
      return out.join('');
    })
    .join('\n');
}

export function jobLog(job, { tall = false, onDone, onUpdate } = {}) {
  const pre = h('div', { class: 'log' + (tall ? ' tall' : '') });
  const head = h('div', { class: 'jobline' });
  const box = h('div', { class: 'stack' }, head, pre);
  let offset = 0, text = '', timer = null, stopped = false, finished = false;

  const renderHead = (j) => clear(head,
    statusTag(j.status), h('b', {}, j.label),
    j.started ? h('span', { class: 'muted' }, fmtS((j.ended || Date.now() / 1000) - j.started)) : null,
    j.status === 'running' || j.status === 'queued'
      ? h('button', { class: 'small danger', onclick: () => attempt(() => api(`jobs/${j.id}/stop`, {}), 'stop sent') }, j.status === 'queued' ? 'Cancel' : 'Stop')
      : null,
  );
  renderHead(job);

  async function poll() {
    if (stopped) return;
    try {
      const r = await api(`jobs/${job.id}/log?offset=${offset}`);
      offset = r.offset;
      if (r.text) {
        text = (text + r.text).slice(-400000);
        const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
        pre.textContent = cleanLog(text).split('\n').slice(-1500).join('\n');
        if (atBottom) pre.scrollTop = pre.scrollHeight;
      }
      renderHead(r.job);
      onUpdate?.(r.job, text);
      if (['done', 'failed', 'stopped'].includes(r.job.status)) {
        if (!finished) { finished = true; onDone?.(r.job, cleanLog(text)); }
        if (!r.text) return; // drained
      }
    } catch (e) { /* server restarting: keep trying */ }
    timer = setTimeout(poll, r_delay());
  }
  const r_delay = () => (document.hidden ? 4000 : 1200);
  poll();
  box.stop = () => { stopped = true; clearTimeout(timer); };
  return box;
}

// Polls fn every ms while the page is shown; returns a stop function.
export function every(ms, fn) {
  let t = null, dead = false;
  const loop = async () => {
    if (dead) return;
    try { await fn(); } catch (e) { /* next round */ }
    if (!dead) t = setTimeout(loop, document.hidden ? Math.max(ms, 5000) : ms);
  };
  loop();
  return () => { dead = true; clearTimeout(t); };
}

export function phoneType(model) {
  if (['donkey_memory', 'donkey_memory_reverse', 'cil_mobile_fast_memory'].includes(model)) return 'CMDNAV_MEM';
  if (['donkey_rnn', 'donkey_3d'].includes(model)) return 'CMDNAV_SEQ';
  return 'CMDNAV';
}

export const MODEL_INFO = {
  donkey_memory: "donkeycar's memory pilot: the image plus its last outputs. Forward only (sigmoid throttle). Judge it closed loop.",
  donkey_memory_reverse: 'donkey_memory with a tanh throttle, so it can reverse.',
  cil_mobile_fast_memory: 'cil_mobile_fast plus a memory of its last outputs; trains like cil (batch norm, flip, turn-weighted loss).',
  cil_mobile_fast: 'The single-frame cil baseline; linear output, so it can drive either way.',
  cil_mobile_fast_cat: 'cil with steering picked from 21 bins and decoded in the model; a plain CMDNAV on the phone.',
  pilot_net: 'Upstream PilotNet, single frame.',
  donkey_rnn: "donkeycar's rnn pilot: a CNN per frame and two LSTMs over the frames 0, 3 and 6 back.",
  donkey_3d: "donkeycar's 3D CNN over the same frame stack (2D convolutions underneath, for TFLite 2.4).",
};
