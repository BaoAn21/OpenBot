// Train: pick direction, sessions, model and options; the queue runs train_fwdbwd.py /
// train_both.py one at a time. Live: epoch, loss curves from logs/log.csv, the log.
import { h, clear, api, attempt, every, fmt, fmtS, fmtAgo, jobLog, statusTag, MODEL_INFO, phoneType, toast } from '../lib.js';
import { lineChart, legend } from '../charts.js';

const DIR_INFO = {
  fwd: 'Forward model: fwd runs, cmd as recorded (−1, 0, 1).',
  bwd: 'Backward model: the reverse parts, cmd 2.',
  both: 'One model for both: fwd and bwd together (train_both.py); pick it in both Session spinners.',
};

function countFor(s, direction) {
  const c = s.counts;
  if (direction === 'both') return [c.fwd_train + c.bwd_train, c.fwd_test + c.bwd_test];
  return [c[`${direction}_train`], c[`${direction}_test`]];
}

function form(opts, onQueued) {
  const st = {
    direction: localStorage.getItem('fb.dir') || 'fwd',
    sessions: new Set(JSON.parse(localStorage.getItem('fb.sessions') || '[]')),
    model: null,
    options: new Set(),
    mem_len: 3, seq_offsets: '0,3,6',
    ...opts.common,
    run_name: '', resume: false,
  };
  const box = h('div', { class: 'card' });
  const preview = h('div', { class: 'log', style: { maxHeight: '120px' } });
  let previewTimer = null;

  const payload = () => ({
    direction: st.direction, model: st.model, sessions: [...st.sessions],
    options: [...st.options], mem_len: st.mem_len, seq_offsets: st.seq_offsets,
    num_epochs: st.num_epochs, early_stop: st.early_stop, batch_size: st.batch_size, learning_rate: st.learning_rate,
    run_name: st.run_name, resume: st.resume,
  });

  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      if (!st.sessions.size) {
        preview.textContent = 'pick at least one session';
        preview.style.color = 'var(--muted)';
        return;
      }
      try {
        const r = await api('train', { ...payload(), preview: true });
        preview.textContent = r.command.replace(/^\S*python\S* /, 'python ');
        preview.style.color = '';
      } catch (e) {
        preview.textContent = e.message;
        preview.style.color = 'var(--bad)';
      }
    }, 200);
  }

  function draw() {
    const models = opts.directions[st.direction];
    if (!models[st.model]) st.model = Object.keys(models)[0];
    const m = models[st.model];
    for (const o of [...st.options]) if (!(o in m.options)) st.options.delete(o);
    localStorage.setItem('fb.dir', st.direction);
    localStorage.setItem('fb.sessions', JSON.stringify([...st.sessions]));

    const sessionRows = opts.sessions.slice().reverse().map((s) => {
      const [tr, te] = countFor(s, st.direction);
      const has = tr + te > 0;
      if (!has) st.sessions.delete(s.number);
      return h('label', { class: 'check', style: { opacity: has ? 1 : 0.45 } },
        h('input', { type: 'checkbox', disabled: !has, checked: st.sessions.has(s.number), onchange: (e) => { e.target.checked ? st.sessions.add(s.number) : st.sessions.delete(s.number); draw(); } }),
        h('span', {}, h('b', {}, `session ${s.number}`), ' ',
          h('span', { class: 'muted' }, `${tr} train / ${te} test`), h('br'),
          h('span', { class: 'dim', style: { fontSize: '12px' } }, (s.note || '').slice(0, 110))));
    });
    const chosen = opts.sessions.filter((s) => st.sessions.has(s.number));
    const tot = chosen.reduce((a, s) => { const [tr, te] = countFor(s, st.direction); return [a[0] + tr, a[1] + te]; }, [0, 0]);

    const num = (key, step = 1) => h('input', { type: 'number', value: st[key], step, onchange: (e) => { st[key] = e.target.value; schedulePreview(); } });

    clear(box,
      h('h2', {}, 'New training run'),
      h('div', { class: 'field' }, h('label', {}, 'Direction'),
        h('div', { class: 'seg' }, ['fwd', 'bwd', 'both'].map((d) => h('button', { class: st.direction === d ? 'on' : '', onclick: () => { st.direction = d; draw(); } }, d)))),
      h('div', { class: 'muted', style: { fontSize: '12px', marginLeft: '130px' } }, DIR_INFO[st.direction]),
      h('h3', {}, 'Sessions'),
      h('div', { class: 'row', style: { marginBottom: '4px' } },
        h('button', { class: 'small', onclick: () => { opts.sessions.forEach((s) => { const [a, b] = countFor(s, st.direction); if (a + b) st.sessions.add(s.number); }); draw(); } }, 'all'),
        h('button', { class: 'small', onclick: () => { st.sessions.clear(); draw(); } }, 'none'),
        h('span', { class: 'muted' }, `${tot[0]} train / ${tot[1]} test frames`),
        chosen.length && !tot[1] ? h('span', { class: 'tag bad' }, '✕ no test data') : null),
      h('div', { style: { maxHeight: '260px', overflow: 'auto', paddingRight: '4px' } }, sessionRows),
      h('h3', {}, 'Model'),
      h('select', { style: { width: '100%' }, onchange: (e) => { st.model = e.target.value; draw(); } },
        Object.keys(models).map((name) => h('option', { value: name, selected: name === st.model }, name))),
      h('div', { class: 'dim', style: { fontSize: '12.5px', margin: '6px 0' } }, MODEL_INFO[st.model] || '', ' ',
        h('span', { class: 'tag' }, `phone: ${phoneType(st.model)}`)),
      m.fixed.length ? h('div', { class: 'row', style: { margin: '6px 0' } }, h('span', { class: 'muted' }, 'always:'), m.fixed.map((f) => h('span', { class: 'tag' }, f))) : null,
      Object.entries(m.options).map(([k, desc]) => h('label', { class: 'check' },
        h('input', { type: 'checkbox', checked: st.options.has(k), onchange: (e) => { e.target.checked ? st.options.add(k) : st.options.delete(k); schedulePreview(); } }),
        h('span', {}, h('code', {}, '--' + k), ' ', h('span', { class: 'dim' }, desc)))),
      m.memory ? h('div', { class: 'field' }, h('label', {}, 'Memory length'), num('mem_len')) : null,
      m.seq ? h('div', { class: 'field' }, h('label', {}, 'Frame offsets'),
        h('input', { type: 'text', value: st.seq_offsets, onchange: (e) => { st.seq_offsets = e.target.value; schedulePreview(); } })) : null,
      h('details', {}, h('summary', {}, 'Epochs, early stop, batch, learning rate, run name'),
        h('div', { class: 'field' }, h('label', {}, 'Max epochs'), num('num_epochs')),
        h('div', { class: 'field' }, h('label', {}, 'Early stop'), h('div', { class: 'row' }, num('early_stop'), h('span', { class: 'muted' }, 'epochs without a better val loss (0 = off)'))),
        h('div', { class: 'field' }, h('label', {}, 'Batch size'), num('batch_size')),
        h('div', { class: 'field' }, h('label', {}, 'Learning rate'), num('learning_rate', 0.0001)),
        h('div', { class: 'field' }, h('label', {}, 'Run name'),
          h('input', { type: 'text', value: st.run_name, placeholder: `${st.direction}_s${[...st.sessions].sort((a, b) => a - b).join('-')}`, onchange: (e) => { st.run_name = e.target.value; schedulePreview(); } })),
        h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: st.resume, onchange: (e) => { st.resume = e.target.checked; schedulePreview(); } }), h('span', {}, h('code', {}, '--resume'), h('span', { class: 'dim' }, ' continue an existing model folder')))),
      h('h3', {}, 'Command'),
      preview,
      h('div', { class: 'row', style: { marginTop: '10px' } },
        h('button', { class: 'primary', disabled: !st.sessions.size, onclick: async () => {
          const job = await attempt(() => api('train', payload()));
          if (job) { toast(job.status === 'running' ? 'Training started' : 'Queued; starts when the current run ends', 'good'); onQueued(); }
        } }, 'Start / queue training')));
    schedulePreview();
  }
  draw();
  return box;
}

function epochsOf(job) {
  const m = /--num_epochs (\d+)/.exec(job.command);
  return m ? Number(m[1]) : 100;
}

function livePanel(job) {
  const info = h('div', { class: 'row' });
  const chartBox = h('div');
  const log = jobLog(job);
  let chart = null, lastEpochs = -1;
  const panel = h('div', { class: 'card' }, h('h2', {}, job.label), info, chartBox, h('h3', {}, 'Log'), log);

  panel.update = async (j) => {
    const p = j.progress;
    const max = epochsOf(j);
    const el = (j.ended || Date.now() / 1000) - j.started;
    clear(info,
      statusTag(j.status),
      h('span', { class: 'dim' }, p ? `epoch ${p.epochs} / ${max}` : 'building records / starting…'),
      h('span', { class: 'muted' }, fmtS(el)),
      p && p.epochs ? h('span', { class: 'muted' }, `≈ ${fmtS(el / p.epochs)} per epoch`) : null,
      p && p.best ? h('span', { class: 'tag good' }, `best val loss ${fmt(p.best.val_loss)} @ epoch ${p.best.epoch + 1}`) : null,
      j.meta.model ? h('a', { href: `#/model?name=${encodeURIComponent(j.meta.model)}` }, 'model →') : null);
    if (j.meta.model && p && p.epochs !== lastEpochs && p.epochs > 0) {
      lastEpochs = p.epochs;
      const d = await api('models/' + encodeURIComponent(j.meta.model));
      const col = (k) => d.rows.map((r) => r[d.header.indexOf(k)]);
      const x = d.rows.map((r) => r[0] + 1);
      const best = p.best ? p.best.epoch + 1 : null;
      const opts = {
        x, series: [
          { label: 'train loss', color: '--series-1', values: col('loss'), dots: x.length < 40 },
          { label: 'val loss', color: '--series-2', values: col('val_loss'), dots: x.length < 40 },
        ],
        xMin: 1, xMax: Math.max(x[x.length - 1], 2), height: 200,
        xFormat: (v) => 'epoch ' + Math.round(v), xTick: (v) => String(Math.round(v)), yFormat: (v) => String(+v.toPrecision(3)),
        markers: best ? [{ x: best, label: 'best' }] : [],
      };
      if (!chart) {
        chart = lineChart(opts);
        clear(chartBox, legend([{ label: 'train loss', color: '--series-1' }, { label: 'val loss', color: '--series-2' }]), chart);
      } else chart.update(opts);
    }
  };
  panel.destroy = () => { log.stop(); chart?.destroy(); };
  return panel;
}

export async function render(view) {
  const opts = await api('train/options');
  const right = h('div');
  const liveBox = h('div');
  const queueBox = h('div');
  right.append(liveBox, queueBox);
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Train'),
      h('span', { class: 'muted' }, 'Runs go through train_fwdbwd.py / train_both.py, one at a time; they keep going if the server restarts.')),
    h('div', { class: 'grid2' }, form(opts, () => refresh()), right));
  const panels = new Map();

  async function refresh() {
    const jobs = (await api('jobs')).filter((j) => j.kind === 'train');
    const running = jobs.filter((j) => j.status === 'running');
    for (const [id, p] of panels) {
      if (!running.some((j) => j.id === id)) { p.destroy(); p.remove(); panels.delete(id); }
    }
    for (const j of running) {
      if (!panels.has(j.id)) { const p = livePanel(j); panels.set(j.id, p); liveBox.append(p); }
      await panels.get(j.id).update(j);
    }
    const empty = liveBox.querySelector(':scope > .empty');
    if (!running.length) { if (!empty) liveBox.append(h('div', { class: 'card empty' }, 'Nothing training right now.')); }
    else empty?.remove();
    const queued = jobs.filter((j) => j.status === 'queued').reverse();
    const done = jobs.filter((j) => ['done', 'failed', 'stopped'].includes(j.status)).slice(0, 8);
    clear(queueBox,
      h('div', { class: 'card' }, h('h2', {}, `Queue (${queued.length})`),
        queued.length ? h('table', {}, h('tbody', {}, queued.map((j, i) => h('tr', {},
          h('td', { class: 'num muted' }, i + 1), h('td', {}, j.label, h('div', { class: 'muted mono', style: { fontSize: '11.5px' } }, j.command.replace(/^\S*python\S* /, ''))),
          h('td', { class: 'right' }, h('button', { class: 'small danger', onclick: async () => { await attempt(() => api(`jobs/${j.id}/stop`, {}), 'Removed from the queue'); refresh(); } }, 'Cancel'))))))
          : h('div', { class: 'muted' }, 'Empty. Queue several runs and they train one after another (overnight, say).')),
      h('div', { class: 'card' }, h('h2', {}, 'Recent runs'),
        done.length ? h('table', {}, h('tbody', {}, done.map((j) => h('tr', {},
          h('td', {}, statusTag(j.status)),
          h('td', {}, j.meta.model ? h('a', { href: `#/model?name=${encodeURIComponent(j.meta.model)}` }, j.label) : j.label,
            j.status === 'failed' ? h('div', {}, h('a', { href: `#/jobs?id=${j.id}`, class: 'muted' }, 'see the log')) : null),
          h('td', { class: 'num muted' }, j.started ? fmtS(j.ended - j.started) : ''),
          h('td', { class: 'num muted' }, fmtAgo(j.ended))))))
          : h('div', { class: 'muted' }, 'No runs from this app yet.')));
  }
  const stop = every(3000, refresh);
  return () => { stop(); for (const p of panels.values()) p.destroy(); };
}
