(() => {
'use strict';
const $ = id => document.getElementById(id);
let current = null, selected = null, busy = false;
const text = (tag, value, className) => { const node = document.createElement(tag); node.textContent = value; if (className) node.className = className; return node; };
function list(title, entries) {
  const section = document.createElement('section'); section.append(text('h4', title));
  if (!entries.length) section.append(text('p', 'None detected in this scan.', 'muted'));
  else { const ul = document.createElement('ul'); entries.forEach(entry => ul.append(text('li', entry))); section.append(ul); }
  return section;
}
function selectFile(file) {
  selected = file.path;
  const d = $('detail'); d.replaceChildren(text('h3', file.path), text('p', file.language + ' · ' + file.lines + ' lines', 'muted'));
  d.append(list('Symbols', file.symbols.map(s => s.name + ' — line ' + s.line)));
  d.append(list('Imports', file.imports.map(i => i.name + ' — line ' + i.line)));
  d.append(list('Resolved local connections', current.report.edges.filter(e => e.from === file.path).map(e => e.to + ' — line ' + e.line)));
  d.append(list('Route patterns', file.routes.map(r => r.method + ' ' + r.route + ' — line ' + r.line)));
  d.append(list('Boundaries to review', file.signals.map(s => s.kind + ' — line ' + s.line)));
  d.append(text('p', 'These are source patterns, not confirmed runtime behavior.', 'muted'));
  renderFiles();
}
function renderFiles() {
  const q = $('filter').value.toLowerCase();
  const files = current.report.files.filter(f => (f.path + ' ' + f.language).toLowerCase().includes(q));
  $('files').replaceChildren(); $('empty').hidden = !!files.length;
  files.forEach(file => { const b = text('button', file.path, 'file-button'); b.type = 'button'; b.setAttribute('aria-pressed', String(selected === file.path)); b.onclick = () => selectFile(file); $('files').append(b); });
}
document.querySelectorAll('[data-target]').forEach(b => b.onclick = () => { $('target').value = b.dataset.target; $('target').focus(); });
$('filter').oninput = () => current && renderFiles();
$('inspect-form').onsubmit = async event => {
  event.preventDefault(); if (busy) return; busy = true;
  $('inspect-button').disabled = true; $('inspect-button').textContent = 'Inspecting…';
  $('error').hidden = true; $('results').hidden = true; current = null;
  $('status').textContent = 'Reading source and resolving local imports…';
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch('/api/reverse-engineering', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:$('target').value.trim()}),signal:controller.signal});
    const payload = await response.json(); if (!response.ok) throw new Error(payload.error || 'Inspection failed.');
    current = payload; selected = null; $('filter').value = '';
    const s = payload.report.summary; $('stats').replaceChildren();
    [[s.files,'source files'],[s.lines.toLocaleString(),'lines'],[payload.report.edges.length,'local connections']].forEach(([v,l]) => { const el = text('div','','stat'); el.append(text('strong',v),text('span',l)); $('stats').append(el); });
    $('coverage').textContent = (s.truncated ? 'Partial coverage: a scan limit was reached. Choose a smaller folder. ' : 'Scan complete. ') + s.skipped + ' entries excluded. ' + payload.report.warnings.length + ' read warnings.';
    $('results').hidden = false; renderFiles();
    if (payload.report.files.length) selectFile(payload.report.files[0]);
    else $('detail').replaceChildren(text('h3','No source files found'),text('p','Choose a folder containing supported source files.'));
    $('status').textContent = 'Inspection complete for ' + payload.report.target + '.';
    $('results-heading').focus();
  } catch (error) {
    $('error').hidden = false; $('error').textContent = error.name === 'AbortError' ? 'This inspection timed out. Choose a smaller folder and try again.' : error.message;
    $('status').textContent = 'Inspection did not finish.';
  } finally { clearTimeout(timeout); busy = false; $('inspect-button').disabled = false; $('inspect-button').textContent = 'Inspect project ↗'; }
};
$('download').onclick = () => {
  if (!current) return;
  const url = URL.createObjectURL(new Blob([current.markdown],{type:'text/markdown;charset=utf-8'}));
  const a = document.createElement('a'); a.href = url; a.download = 'AXEN-reverse-engineering.md'; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
};
fetch('/api/reverse-engineering/status').then(r => {if (!r.ok) throw new Error();return r.json();}).then(data => {
  const t = data.training;
  $('training-status').textContent = t ? 'Latest run: ' + t.steps_run + ' training steps. ' + (t.promoted ? 'The evaluated candidate was activated.' : 'Candidate saved separately; existing brain retained.') + ' This is a small-model experiment, not frontier-model training.' : 'No evaluated training run recorded yet. Source inspection works independently of the language model.';
}).catch(() => $('training-status').textContent = 'Training status could not be loaded.');
})();
