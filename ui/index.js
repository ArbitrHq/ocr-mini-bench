const fmtPct = (v) => (Number.isFinite(v) ? `${v.toFixed(1)}%` : '');
const fmtUsd = (v) => (Number.isFinite(v) ? `$${v.toFixed(4)}` : 'n/a');
const fmtSec = (v) => (Number.isFinite(v) ? `${(v / 1000).toFixed(1)}s` : 'n/a');

function weightedAll(byDomain) {
  const include = new Set(['receipts', 'invoices', 'logistics']);
  const domains = (byDomain || []).filter((d) => include.has(String(d.domain || '').toLowerCase()));
  if (!domains.length) return [];

  const byModel = new Map();
  for (const d of domains) {
    for (const row of d.rows || []) {
      const list = byModel.get(row.model_key) || [];
      list.push(row);
      byModel.set(row.model_key, list);
    }
  }

  const avg = (arr, key) => {
    const vals = arr.map((x) => x[key]).filter((v) => Number.isFinite(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const rows = [...byModel.values()].map((arr) => {
    const b = arr[0];
    return {
      ...b,
      success_rate_pct: avg(arr, 'success_rate_pct') ?? 0,
      pass_at_2_strict_pct: avg(arr, 'pass_at_2_strict_pct'),
      pass_at_3_strict_pct: avg(arr, 'pass_at_3_strict_pct'),
      pass_at_10_strict_pct: avg(arr, 'pass_at_10_strict_pct'),
      avg_total_field_pass_pct: avg(arr, 'avg_total_field_pass_pct') ?? avg(arr, 'avg_field_accuracy_pct') ?? 0,
      avg_critical_accuracy_pct: avg(arr, 'avg_critical_accuracy_pct') ?? 0,
      avg_cost_per_doc_usd: avg(arr, 'avg_cost_per_doc_usd'),
      avg_latency_ms: avg(arr, 'avg_latency_ms') ?? 0,
    };
  });

  rows.sort((a, b) => b.success_rate_pct - a.success_rate_pct || b.avg_critical_accuracy_pct - a.avg_critical_accuracy_pct);
  return rows;
}

function renderTable(rows) {
  const body = document.getElementById('leaderboard-body');
  body.innerHTML = '';
  rows.forEach((row, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${row.model_label}</td>
      <td>${row.provider}</td>
      <td>${fmtPct(row.success_rate_pct)}</td>
      <td>${fmtPct(row.pass_at_2_strict_pct)}</td>
      <td>${fmtPct(row.pass_at_3_strict_pct)}</td>
      <td>${fmtPct(row.pass_at_10_strict_pct)}</td>
      <td>${fmtPct(row.avg_total_field_pass_pct ?? row.avg_field_accuracy_pct)}</td>
      <td>${fmtPct(row.avg_critical_accuracy_pct)}</td>
      <td>${fmtUsd(row.avg_cost_per_doc_usd)}</td>
      <td>${fmtSec(row.avg_latency_ms)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderTabs(snapshot) {
  const tabs = document.getElementById('domain-tabs');
  tabs.innerHTML = '';

  const domainRows = new Map((snapshot.by_domain || []).map((d) => [String(d.domain).toLowerCase(), d.rows || []]));
  const allRows = weightedAll(snapshot.by_domain || []);

  const items = [{ id: 'all', label: 'All', rows: allRows }];
  for (const d of ['receipts', 'invoices', 'logistics']) {
    if (domainRows.has(d)) items.push({ id: d, label: d[0].toUpperCase() + d.slice(1), rows: domainRows.get(d) });
  }

  let active = items[0]?.id || 'all';
  const draw = () => {
    const current = items.find((i) => i.id === active);
    renderTable(current?.rows || []);
    [...tabs.querySelectorAll('button')].forEach((btn) => btn.classList.toggle('active', btn.dataset.id === active));
  };

  items.forEach((item) => {
    const btn = document.createElement('button');
    btn.className = 'tab';
    btn.dataset.id = item.id;
    btn.textContent = item.label;
    btn.onclick = () => {
      active = item.id;
      draw();
    };
    tabs.appendChild(btn);
  });

  draw();
}

async function main() {
  const [metaRes, dataRes] = await Promise.all([fetch('/api/meta'), fetch('/api/leaderboard')]);
  const meta = await metaRes.json();
  const snapshot = await dataRes.json();

  document.getElementById('meta').textContent = `Updated: ${new Date(snapshot.generated_at).toLocaleString()} | Runs: ${snapshot.run_count} | Source: ${meta.leaderboard_path || 'n/a'}`;
  renderTabs(snapshot);
}

main().catch((err) => {
  document.getElementById('meta').textContent = `Failed to load leaderboard: ${err.message || String(err)}`;
});
