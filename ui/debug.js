const qs = (id) => document.getElementById(id);

function fillSelect(el, items, valueFn, labelFn, selected) {
  el.innerHTML = '';
  items.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = valueFn(item);
    opt.textContent = labelFn(item);
    if (opt.value === selected) opt.selected = true;
    el.appendChild(opt);
  });
}

function rowClass(c) {
  if (!c || c.scored === false) return 'gray';
  return c.matched ? 'pass' : 'fail';
}

async function main() {
  const [metaRes, debugRes] = await Promise.all([fetch('/api/meta'), fetch('/api/debug')]);
  const meta = await metaRes.json();
  const debug = await debugRes.json();
  qs('debug-meta').textContent = `Runs: ${debug.runs?.length || 0} | Source: ${meta.debug_path || 'n/a'}`;

  const docs = debug.documents || [];
  const runs = debug.runs || [];

  const domains = [...new Set(docs.map((d) => d.domain))].sort();
  let domain = domains[0] || '';
  let docId = '';
  let modelKey = 'all';
  let runNo = '';

  const render = () => {
    const domainDocs = docs.filter((d) => d.domain === domain).sort((a, b) => a.document_id.localeCompare(b.document_id));
    if (!domainDocs.find((d) => d.document_id === docId)) docId = domainDocs[0]?.document_id || '';

    fillSelect(qs('domain'), domains, (x) => x, (x) => x, domain);
    fillSelect(qs('document'), domainDocs, (x) => x.document_id, (x) => x.document_id, docId);

    const doc = domainDocs.find((d) => d.document_id === docId);
    qs('ground-truth').textContent = doc ? JSON.stringify(doc.ground_truth, null, 2) : '';
    qs('pdf-frame').src = doc ? `/api/document?source_pdf=${encodeURIComponent(doc.source_pdf)}` : 'about:blank';

    const docRuns = runs.filter((r) => r.domain === domain && r.document_id === docId).sort((a, b) => a.model_label.localeCompare(b.model_label) || a.run_number - b.run_number);
    const models = [{ key: 'all', label: 'All models' }, ...[...new Set(docRuns.map((r) => `${r.model_key}|||${r.model_label}`))].map((x) => {
      const [key, label] = x.split('|||');
      return { key, label };
    })];
    if (!models.find((m) => m.key === modelKey)) modelKey = 'all';
    fillSelect(qs('model'), models, (x) => x.key, (x) => x.label, modelKey);

    const runNums = [...new Set(docRuns.map((r) => String(r.run_number)))].sort((a, b) => Number(a) - Number(b));
    if (!runNums.includes(runNo)) runNo = runNums[0] || '';
    fillSelect(qs('run'), runNums, (x) => x, (x) => x, runNo);

    const chosen = docRuns.filter((r) => (modelKey === 'all' || r.model_key === modelKey) && String(r.run_number) === runNo);
    const active = chosen[0] || null;

    const table = qs('key-table');
    table.innerHTML = '';
    if (active) {
      for (const c of active.key_comparisons || []) {
        const tr = document.createElement('tr');
        tr.className = rowClass(c);
        tr.innerHTML = `<td>${c.key}${c.critical ? ' *' : ''}</td><td>${(c.expected_values || []).join(' | ')}</td><td>${c.extracted_value || ''}</td><td>${c.matched ? 'pass' : c.scored ? 'fail' : 'n/a'}</td>`;
        table.appendChild(tr);
      }
      qs('raw-output').textContent = active.raw_output || '';
    } else {
      qs('raw-output').textContent = '';
    }
  };

  qs('domain').onchange = (e) => {
    domain = e.target.value;
    render();
  };
  qs('document').onchange = (e) => {
    docId = e.target.value;
    render();
  };
  qs('model').onchange = (e) => {
    modelKey = e.target.value;
    render();
  };
  qs('run').onchange = (e) => {
    runNo = e.target.value;
    render();
  };

  render();
}

main().catch((err) => {
  qs('debug-meta').textContent = `Failed to load debug: ${err.message || String(err)}`;
});
