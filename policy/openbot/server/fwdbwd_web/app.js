// Router and the status bar. Each page module exports render(view, params) and may return
// a cleanup function (timers, charts), called when the page changes.
import { api, fmtS } from './lib.js';

const PAGES = {
  inbox: () => import('./pages/inbox.js'),
  play: () => import('./pages/player.js'),
  sessions: () => import('./pages/sessions.js'),
  train: () => import('./pages/train.js'),
  models: () => import('./pages/models.js'),
  model: () => import('./pages/model.js'),
  runs: () => import('./pages/runs.js'),
  jobs: () => import('./pages/jobs.js'),
};
const NAV_FOR = { play: null, model: 'models' };

let cleanup = null;
let seq = 0;

async function route() {
  const [path, query] = (location.hash.replace(/^#\/?/, '') || 'inbox').split('?');
  const page = PAGES[path] ? path : 'inbox';
  const params = Object.fromEntries(new URLSearchParams(query || ''));
  const nav = page in NAV_FOR ? NAV_FOR[page] : page;
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.page === nav));
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  const view = document.getElementById('view');
  view.replaceChildren();
  const mine = ++seq;
  const mod = await PAGES[page]();
  if (mine !== seq) return;
  window.scrollTo(0, 0);
  cleanup = (await mod.render(view, params)) || null;
}

async function status() {
  try {
    const s = await api('status');
    const gpu = document.getElementById('status-gpu');
    if (s.gpu) {
      const used = s.gpu.used_mb / 1024, total = s.gpu.total_mb / 1024;
      gpu.textContent = `GPU ${used.toFixed(1)}/${total.toFixed(0)} GB · ${s.gpu.util}%`;
      gpu.title = `${s.gpu.name}\n` + s.gpu.processes.map((p) => `${p.pid} ${p.name.split('/').pop()} ${p.used_mb} MB`).join('\n');
      gpu.classList.toggle('hot', s.gpu.used_mb / s.gpu.total_mb > 0.6 && !s.running.some((j) => j.gpu));
    } else {
      gpu.textContent = 'no GPU info';
    }
    const job = document.getElementById('status-job');
    const train = s.running.find((j) => j.kind === 'train');
    const other = s.running.find((j) => j.kind !== 'train');
    if (train) {
      const p = train.progress;
      job.textContent = `● training ${train.label}` + (p ? ` · epoch ${p.epochs}` : '') + (s.queued ? ` · ${s.queued} queued` : '');
      job.href = '#/train';
      job.hidden = false; job.classList.add('busy');
    } else if (other) {
      job.textContent = `● ${other.label} · ${fmtS(Date.now() / 1000 - other.started)}`;
      job.href = '#/jobs';
      job.hidden = false; job.classList.add('busy');
    } else {
      job.hidden = true;
    }
    const badge = document.getElementById('nav-inbox');
    badge.textContent = s.inbox;
    badge.hidden = !s.inbox;
  } catch (e) {
    document.getElementById('status-gpu').textContent = 'server offline';
  }
  setTimeout(status, document.hidden ? 8000 : 3000);
}

window.addEventListener('hashchange', route);
route();
status();
