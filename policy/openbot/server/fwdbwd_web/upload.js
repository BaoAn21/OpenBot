// Drop zone for the phone's zips; posts to /upload like the phone does, one at a time.
import { h, clear, toast, fmtMB } from './lib.js';

function send(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload');
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
      else {
        let msg = xhr.responseText;
        try { msg = JSON.parse(msg).error; } catch (e) { /* text */ }
        reject(new Error(`${file.name}: ${msg}`));
      }
    };
    xhr.onerror = () => reject(new Error(`${file.name}: upload failed`));
    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  });
}

export function dropzone({ hint, onDone }) {
  const input = h('input', { type: 'file', accept: '.zip', multiple: true, hidden: true });
  const status = h('div', { class: 'stack' });
  const zone = h('div', { class: 'dropzone' },
    h('div', {}, hint, ' ', h('a', { href: '#', onclick: (e) => { e.preventDefault(); input.click(); } }, 'choose files')),
    status, input);

  async function upload(files) {
    files = [...files].filter((f) => f.name.endsWith('.zip'));
    if (!files.length) { toast('Only .zip files (as the phone saves them)', 'bad'); return; }
    let ok = 0;
    for (const [i, f] of files.entries()) {
      const bar = h('span');
      clear(status, h('div', { class: 'muted' }, `${i + 1}/${files.length} ${f.name} (${fmtMB(f.size)})`), h('div', { class: 'progress' }, bar));
      try {
        await send(f, (p) => { bar.style.width = (p * 100).toFixed(0) + '%'; });
        ok++;
      } catch (e) {
        toast(e.message, 'bad');
      }
    }
    clear(status);
    if (ok) toast(`${ok} upload${ok > 1 ? 's' : ''} done`, 'good');
    onDone?.();
  }

  input.addEventListener('change', () => upload(input.files));
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); upload(e.dataTransfer.files); });
  return zone;
}
