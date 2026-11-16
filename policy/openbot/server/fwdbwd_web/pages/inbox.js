// Inbox: what the phone uploaded and isn't split yet, the split into a session, parked
// recordings, and where phone uploads go.
import { h, clear, api, attempt, every, fileUrl, fmtS, fmtAgo, fmtMB, recDate, jobLog, toast } from '../lib.js';
import { dropzone } from '../upload.js';

function dirBar(r) {
  const total = r.fwd_s + r.bwd_s + r.stop_s || 1;
  return h('div', { class: 'stack' },
    h('div', { class: 'dirbar', title: `forward ${r.fwd_s}s · reverse ${r.bwd_s}s · stopped ${r.stop_s}s` },
      r.fwd_s ? h('span', { class: 'f', style: { width: (100 * r.fwd_s / total) + '%' } }) : null,
      r.bwd_s ? h('span', { class: 'b', style: { width: (100 * r.bwd_s / total) + '%' } }) : null,
      r.stop_s ? h('span', { class: 's', style: { width: (100 * r.stop_s / total) + '%' } }) : null),
    h('div', { class: 'legend-inline' },
      h('span', {}, h('span', { class: 'sw', style: { background: 'var(--series-1)' } }), `fwd ${fmtS(r.fwd_s)}`),
      h('span', {}, h('span', { class: 'sw', style: { background: 'var(--text-2)' } }), `bwd ${fmtS(r.bwd_s)}`),
      r.stop_s ? h('span', { class: 'muted' }, `stopped ${fmtS(r.stop_s)}`) : null));
}

function card(r, refresh) {
  const flags = [];
  if (r.reason) flags.push(h('span', { class: 'tag warn', title: 'split_direction.py parks it instead of splitting' }, `⚠ ${r.reason} → parked`));
  if (r.duplicate_of) flags.push(h('span', { class: 'tag bad', title: 'split refuses a recording a session already has' }, `✕ already in session ${r.duplicate_of}`));
  if (r.error) flags.push(h('span', { class: 'tag bad', title: r.error }, '✕ unreadable'));
  const play = () => { location.hash = `#/play?path=${encodeURIComponent(r.path)}&back=inbox`; };
  return h('div', { class: 'rec' },
    r.thumb ? h('img', { class: 'thumb', src: fileUrl(r.thumb), loading: 'lazy', alt: '', onclick: play }) : h('div', { class: 'thumb' }),
    h('div', { class: 'body' },
      h('div', { class: 'row' }, h('span', { class: 'title' }, r.name), h('span', { class: 'muted' }, recDate(r.name))),
      h('div', { class: 'dim' }, `${fmtS(r.seconds)} · ${r.frames} frames`),
      dirBar(r),
      flags.length ? h('div', { class: 'row' }, flags) : null,
      h('div', { class: 'actions' },
        h('button', { class: 'small', onclick: play }, '▶ Play'),
        h('button', { class: 'small danger', onclick: async () => {
          if (!confirm(`Move ${r.name} out of the inbox (to parked/removed/, restorable)?`)) return;
          await attempt(() => api('inbox/remove', { name: r.name }), `${r.name} parked`);
          refresh();
        } }, 'Remove'))));
}

function splitPanel(state, refresh) {
  const box = h('div', { class: 'card' });
  let mode = 'new';
  const note = h('input', { type: 'text', placeholder: 'what this batch is: date, light, track, "takeovers of …"', style: { width: '100%' } });
  const frac = h('input', { type: 'number', value: '0.2', step: '0.05', min: '0', max: '0.9' });
  const existing = h('select', {}, state.sessions.slice().reverse().map((n) => h('option', { value: n }, `session ${n}`)));
  const out = h('div', { class: 'stack' });
  const seg = h('div', { class: 'seg' });
  const renderSeg = () => clear(seg,
    h('button', { class: mode === 'new' ? 'on' : '', onclick: () => { mode = 'new'; renderSeg(); } }, `New session ${state.next_session}`),
    h('button', { class: mode === 'add' ? 'on' : '', disabled: !state.sessions.length, onclick: () => { mode = 'add'; renderSeg(); } }, 'Add to session'));
  renderSeg();

  const params = (dry) => ({
    dry_run: dry,
    note: note.value.trim(),
    test_fraction: frac.value,
    session: mode === 'add' ? Number(existing.value) : null,
  });
  const run = async (dry) => {
    if (!dry) {
      if (mode === 'new' && !note.value.trim() && !confirm('Split without a note? The note is how you tell sessions apart later.')) return;
      if (!confirm(`Split ${state.recordings.length} recording(s) into ${mode === 'new' ? 'a new session ' + state.next_session : 'session ' + existing.value}? They move out of the inbox.`)) return;
    }
    const job = await attempt(() => api('split', params(dry)));
    if (!job) return;
    if (out.firstChild?.stop) out.firstChild.stop();
    clear(out, jobLog(job, {
      onDone: (j) => {
        if (!dry && j.status === 'done') toast('Split done — see Sessions', 'good');
        refresh();
      },
    }));
  };
  const n = state.recordings.length;
  box.append(
    h('h2', {}, 'Split the inbox'),
    h('div', { class: 'dim', style: { marginBottom: '8px' } },
      'Each recording is cut by throttle sign into forward runs and one backward part (cmd 2), and a random share of the recordings goes to test, both directions together.'),
    h('div', { class: 'row' }, seg, h('span', { class: 'muted' }, mode === 'add' ? existing : null)),
    h('div', { class: 'field' }, h('label', {}, 'Note'), note),
    h('div', { class: 'field' }, h('label', {}, 'Test share'), h('div', { class: 'row' }, frac, h('span', { class: 'muted' }, 'of the recordings (0 for takeover sessions)'))),
    h('div', { class: 'row', style: { marginTop: '8px' } },
      h('button', { disabled: !n, onclick: () => run(true) }, 'Preview split'),
      h('button', { class: 'primary', disabled: !n, onclick: () => run(false) }, `Split ${n} recording${n === 1 ? '' : 's'}`)),
    out);
  return box;
}

export async function render(view) {
  const head = h('div', { class: 'page-head' }, h('h1', {}, 'Inbox'));
  const body = h('div');
  view.append(head, body);
  let lastKey = '';
  let splitBox = null;

  async function refresh(force) {
    const s = await api('inbox');
    const key = JSON.stringify([s.recordings.map((r) => [r.name, r.frames, r.duplicate_of]), s.zips, s.parked.length, s.uploads.length, s.settings, s.sessions]);
    if (key === lastKey && !force) return;
    lastKey = key;
    const keepSplit = splitBox && splitBox.querySelector('.log');

    const uploadTo = h('select', { onchange: async () => {
      await attempt(() => api('settings', { upload_to: uploadTo.value }), 'Saved');
      refresh(true);
    } },
      h('option', { value: 'inbox', selected: s.settings.upload_to === 'inbox' }, 'fwdbwd inbox (this page)'),
      h('option', { value: 'legacy', selected: s.settings.upload_to === 'legacy' }, 'old dataset/uploaded/'));

    clear(head, h('h1', {}, 'Inbox'),
      h('span', { class: 'muted' }, `${s.recordings.length} recording${s.recordings.length === 1 ? '' : 's'} waiting`),
      h('span', { class: 'spacer' }),
      h('label', { class: 'dim row' }, 'Phone uploads go to', uploadTo));

    if (!keepSplit) splitBox = splitPanel(s, () => refresh(true));
    const totals = s.recordings.reduce((a, r) => ({ f: a.f + r.fwd_s, b: a.b + r.bwd_s }), { f: 0, b: 0 });

    clear(body,
      h('div', { class: 'grid2' },
        h('div', {},
          splitBox,
          h('div', { class: 'card' }, h('h2', {}, 'Add recordings'),
            dropzone({ hint: 'Drop the phone\'s recording zips here, or', onDone: () => refresh(true) }),
            h('div', { class: 'muted', style: { marginTop: '8px', fontSize: '12px' } },
              'The phone uploads by itself: Data Logger → Save as Server, with this PC picked as the server. autopilot_*.zip go to Autopilot runs.'),
            s.zips.length ? h('div', { class: 'row', style: { marginTop: '8px' } },
              h('span', { class: 'tag warn' }, `${s.zips.length} zip(s) not unpacked`),
              h('button', { class: 'small', onclick: async () => { await attempt(() => api('inbox/unpack', {}), 'Unpacked'); refresh(true); } }, 'Unpack')) : null),
          h('div', { class: 'card' }, h('h2', {}, 'Recent uploads'),
            s.uploads.length ? h('table', {}, h('tbody', {}, s.uploads.slice(0, 12).map((u) => h('tr', {},
              h('td', { class: 'mono' }, u.name),
              h('td', {}, h('span', { class: 'tag ' + (u.status === 'ok' ? 'good' : u.status.startsWith('broken') ? 'bad' : '') }, u.status)),
              h('td', { class: 'num muted' }, fmtMB(u.size)),
              h('td', { class: 'num muted' }, fmtAgo(u.time))))))
              : h('div', { class: 'muted' }, 'None since the server started.')),
          s.parked.length ? h('details', { class: 'card' },
            h('summary', {}, `Parked (${s.parked.length})`),
            h('table', {}, h('tbody', {}, s.parked.map((p) => h('tr', {},
              h('td', {}, h('span', { class: 'tag' }, p.reason)),
              h('td', { class: 'mono' }, h('a', { href: `#/play?path=${encodeURIComponent(p.path)}&back=inbox` }, p.name)),
              h('td', { class: 'right' }, h('button', { class: 'small', onclick: async () => {
                await attempt(() => api('parked/restore', { reason: p.reason, name: p.name }), 'Back in the inbox');
                refresh(true);
              } }, 'Restore'))))))) : null),
        h('div', {},
          s.recordings.length
            ? h('div', { class: 'stack' },
              h('div', { class: 'dim' }, `Forward ${fmtS(totals.f)} · reverse ${fmtS(totals.b)} in total`),
              h('div', { class: 'cards' }, s.recordings.map((r) => card(r, () => refresh(true)))))
            : h('div', { class: 'card empty' }, 'The inbox is empty. Record with the Data Logger (Save as: Server) and the recordings appear here.'))));
  }

  await refresh(true);
  return every(4000, () => refresh(false));
}
