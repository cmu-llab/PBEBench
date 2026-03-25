/* ===== PBEBench Website JS ===== */

let allModels = [];
let sortCol = 'pass';
let sortDir = 'desc'; // desc = highest first

const COMPANY_ICONS = {
  openai:    'images/company-icons/openai.svg',
  anthropic: 'images/company-icons/anthropic.svg',
  google:    'images/company-icons/google.svg',
  deepseek:  'images/company-icons/deepseek.svg',
  qwen:      null,   // no icon, show text
  mistral:   null,
};

function getCompanyIcon(company) {
  const path = COMPANY_ICONS[company];
  if (!path) return '';
  return `<img src="${path}" alt="${company}" class="company-icon" title="${company}">`;
}

function renderBadges(model) {
  let html = '';
  if (model.is_reasoning) html += `<span class="badge badge-reasoning" title="Reasoning model">★</span>`;
  if (model.is_moe)       html += `<span class="badge badge-moe"       title="Mixture-of-Experts">■</span>`;
  if (model.is_closed)    html += `<span class="badge badge-closed"     title="Closed-source">✸</span>`;
  return html;
}

function fmtNum(val, decimals = 1) {
  if (val === null || val === undefined) return '<td class="na">—</td>';
  const cls = val < 0 ? 'num negative' : 'num';
  return `<td class="${cls}">${val.toFixed(decimals)}%</td>`;
}

function fmtComplexity(val) {
  if (val === null || val === undefined) return '<td class="na">—</td>';
  return `<td class="num">${val.toFixed(2)}</td>`;
}

function sortModels(col) {
  if (sortCol === col) {
    sortDir = sortDir === 'desc' ? 'asc' : 'desc';
  } else {
    sortCol = col;
    sortDir = 'desc';
  }
  renderLeaderboard();
  updateSortHeaders();
}

function updateSortHeaders() {
  document.querySelectorAll('table.leaderboard thead th').forEach(th => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.col === sortCol) {
      th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
    }
  });
}

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody || !allModels.length) return;

  const data = [...allModels].sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    // nulls last
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    return sortDir === 'desc' ? vb - va : va - vb;
  });

  let rank = 0;
  let lastVal = null;
  let skipCount = 0;

  tbody.innerHTML = data.map((m, i) => {
    // Compute display rank (skip nulls)
    const val = m[sortCol];
    if (val !== null) {
      if (val !== lastVal) {
        rank += 1 + skipCount;
        skipCount = 0;
        lastVal = val;
      } else {
        skipCount++;
      }
    }

    const rowClass = rank <= 3 && val !== null ? 'top-3' : '';
    const typeLabel = m.is_closed ? 'Closed' : 'Open';

    return `
      <tr class="${rowClass}">
        <td class="rank">${val !== null ? rank : '—'}</td>
        <td class="model-cell">
          <div class="model-name-wrap">
            ${getCompanyIcon(m.company)}
            <span class="model-name">${m.name}</span>
            ${renderBadges(m)}
          </div>
        </td>
        <td class="type-cell">${typeLabel}</td>
        ${fmtNum(m.pass)}
        ${fmtNum(m.edit_sim)}
        ${fmtNum(m.valid_rate)}
        ${fmtComplexity(m.complexity)}
        ${fmtNum(m.reorder_acc)}
        ${fmtNum(m.reorder_uacc)}
      </tr>`;
  }).join('');

  updateSortHeaders();
}

async function init() {
  try {
    const resp = await fetch('data/model-data.json');
    allModels = await resp.json();
    renderLeaderboard();
    setupSortListeners();
  } catch (e) {
    console.error('Failed to load model data:', e);
    const tbody = document.getElementById('leaderboard-tbody');
    if (tbody) tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:#999;padding:2rem">Failed to load leaderboard data.</td></tr>';
  }
}

function setupSortListeners() {
  document.querySelectorAll('table.leaderboard thead th[data-col]').forEach(th => {
    th.addEventListener('click', () => sortModels(th.dataset.col));
  });
}

function copyCitation() {
  const text = document.getElementById('citation-text').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  });
}

document.addEventListener('DOMContentLoaded', init);
