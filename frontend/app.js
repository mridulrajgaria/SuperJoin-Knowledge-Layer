/**
 * Superjoin — Evidence Intelligence Workspace
 * Frontend Application Controller
 * Connects directly to FastAPI backend at http://localhost:8000
 */

const API_BASE = window.location.origin.includes(':8000') 
  ? window.location.origin 
  : 'http://localhost:8000';

// Global state
const state = {
  currentView: 'dashboard',
  stats: null,
  allEntities: [],
  facts: [],
  filteredFacts: [],
  relationships: [],
  filteredRelationships: [],
  selectedDocId: '',
  selectedEntity: '',
  selectedRelType: '',
  searchQuery: '',
  selectedUploadFile: null,
};

// DOM Elements
const elements = {
  // Navigation
  navButtons: document.querySelectorAll('.nav-item'),
  viewPanels: document.querySelectorAll('.view-panel'),
  navFactsCount: document.getElementById('nav-facts-count'),
  navRelsCount: document.getElementById('nav-rels-count'),
  backendStatusText: document.getElementById('backend-status-text'),
  globalSearch: document.getElementById('global-search'),

  // Dashboard
  btnRefreshStats: document.getElementById('btn-refresh-stats'),
  statTotalFacts: document.getElementById('stat-total-facts'),
  statTotalDocs: document.getElementById('stat-total-docs'),
  statCorroborates: document.getElementById('stat-corroborates'),
  statReconciled: document.getElementById('stat-reconciled'),
  statContradicts: document.getElementById('stat-contradicts'),
  docInventoryMeta: document.getElementById('doc-inventory-meta'),
  tbodyDocuments: document.getElementById('tbody-documents'),

  // Facts View
  filterEntity: document.getElementById('filter-entity'),
  filterDocument: document.getElementById('filter-document'),
  btnResetFactsFilter: document.getElementById('btn-reset-facts-filter'),
  factsFilterCount: document.getElementById('facts-filter-count'),
  factsContainer: document.getElementById('facts-container'),

  // Relationships View
  relTabs: document.querySelectorAll('.rel-tab'),
  tabCountCorroborates: document.getElementById('tab-count-corroborates'),
  tabCountReconciled: document.getElementById('tab-count-reconciled'),
  tabCountContradicts: document.getElementById('tab-count-contradicts'),
  relationshipsFilterCount: document.getElementById('relationships-filter-count'),
  relationshipsContainer: document.getElementById('relationships-container'),

  // Modal
  btnOpenUpload: document.getElementById('btn-open-upload'),
  uploadModal: document.getElementById('upload-modal'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnModalCancel: document.getElementById('btn-modal-cancel'),
  btnModalSubmit: document.getElementById('btn-modal-submit'),
  btnModalViewFacts: document.getElementById('btn-modal-view-facts'),
  modalDropzone: document.getElementById('modal-dropzone'),
  fileInputModal: document.getElementById('file-input-modal'),
  modalFileName: document.getElementById('modal-file-name'),
  modalBodyIdle: document.getElementById('modal-body-idle'),
  modalBodyProcessing: document.getElementById('modal-body-processing'),
  modalBodyResult: document.getElementById('modal-body-result'),
  resultTitle: document.getElementById('result-title'),
  resultDesc: document.getElementById('result-desc'),
  resultMetrics: document.getElementById('result-metrics'),
  resultErrors: document.getElementById('result-errors'),
  resultStatusBox: document.getElementById('result-status-box'),
};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  initIcons();
  attachEventListeners();
  loadInitialData();
});

function initIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function attachEventListeners() {
  // Navigation
  elements.navButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      switchView(view);
    });
  });

  // Global search
  elements.globalSearch.addEventListener('input', (e) => {
    state.searchQuery = e.target.value.trim().toLowerCase();
    applyFilters();
  });

  // Dashboard actions
  elements.btnRefreshStats.addEventListener('click', loadStats);

  // Facts filter actions
  elements.filterEntity.addEventListener('change', (e) => {
    state.selectedEntity = e.target.value;
    loadFacts();
  });

  elements.filterDocument.addEventListener('change', (e) => {
    state.selectedDocId = e.target.value;
    loadFacts();
  });

  elements.btnResetFactsFilter.addEventListener('click', () => {
    state.selectedEntity = '';
    state.selectedDocId = '';
    state.searchQuery = '';
    elements.filterEntity.value = '';
    elements.filterDocument.value = '';
    elements.globalSearch.value = '';
    loadFacts();
  });

  // Relationships category tabs
  elements.relTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      elements.relTabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      state.selectedRelType = tab.dataset.type || '';
      loadRelationships();
    });
  });

  // Modal actions
  elements.btnOpenUpload.addEventListener('click', openUploadModal);
  elements.btnCloseModal.addEventListener('click', closeUploadModal);
  elements.btnModalCancel.addEventListener('click', closeUploadModal);
  elements.btnModalSubmit.addEventListener('click', executeUpload);

  // Drag & drop for modal
  elements.modalDropzone.addEventListener('click', () => elements.fileInputModal.click());
  elements.fileInputModal.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

  ['dragenter', 'dragover'].forEach((eventName) => {
    elements.modalDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.modalDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    elements.modalDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.modalDropzone.classList.remove('dragover');
    });
  });

  elements.modalDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });
}

function switchView(viewName) {
  state.currentView = viewName;
  elements.navButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });
  elements.viewPanels.forEach((panel) => {
    panel.classList.toggle('active', panel.id === `view-${viewName}`);
  });

  if (viewName === 'facts' && state.facts.length === 0) {
    loadFacts();
  } else if (viewName === 'relationships' && state.relationships.length === 0) {
    loadRelationships();
  }
}

// ==========================================================================
// API Operations
// ==========================================================================

async function loadInitialData() {
  await loadStats();
  await loadFacts();
  await loadRelationships();
}

async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const stats = await res.json();
    state.stats = stats;

    // Update connection status
    elements.backendStatusText.textContent = `Connected: ${API_BASE.replace('http://', '')}`;
    document.querySelector('.status-dot').classList.remove('disconnected');

    // Update stat numbers
    elements.statTotalFacts.textContent = Number(stats.total_facts || 0).toLocaleString();
    elements.statTotalDocs.textContent = Number(stats.total_documents || 0).toLocaleString();
    elements.statCorroborates.textContent = Number(stats.relationships_by_type?.corroborates || 0).toLocaleString();
    elements.statReconciled.textContent = Number(stats.relationships_by_type?.reconciled_by_context || 0).toLocaleString();
    elements.statContradicts.textContent = Number(stats.relationships_by_type?.contradicts || 0).toLocaleString();

    // Update sidebar badges
    elements.navFactsCount.textContent = stats.total_facts || '0';
    elements.navRelsCount.textContent = stats.total_relationships || '0';

    // Update relationship tab counts
    elements.tabCountCorroborates.textContent = stats.relationships_by_type?.corroborates || '0';
    elements.tabCountReconciled.textContent = stats.relationships_by_type?.reconciled_by_context || '0';
    elements.tabCountContradicts.textContent = stats.relationships_by_type?.contradicts || '0';

    // Render documents table
    renderDocumentsTable(stats.documents || []);
    populateDocumentSelect(stats.documents || []);
  } catch (err) {
    console.error('Failed to load stats:', err);
    elements.backendStatusText.textContent = 'API Offline';
    document.querySelector('.status-dot').classList.add('disconnected');
  }
}

function renderDocumentsTable(documents) {
  elements.docInventoryMeta.textContent = `${documents.length} documents loaded`;

  if (documents.length === 0) {
    elements.tbodyDocuments.innerHTML = `<tr><td colspan="3" class="table-empty">No documents currently in storage.</td></tr>`;
    return;
  }

  elements.tbodyDocuments.innerHTML = documents
    .map(
      (doc) => `
      <tr>
        <td>
          <div style="font-weight: 500; color: var(--text-primary);">${escapeHtml(doc.source_doc_id)}</div>
        </td>
        <td>
          <span style="font-weight: 600; color: var(--text-primary); font-variant-numeric: tabular-nums;">${doc.fact_count}</span> facts
        </td>
        <td class="text-right">
          <button class="btn btn-secondary btn-sm" onclick="filterByDoc('${escapeHtml(doc.source_doc_id)}')">
            <i data-lucide="filter" class="btn-icon"></i> View Facts
          </button>
        </td>
      </tr>
    `
    )
    .join('');

  initIcons();
}

window.filterByDoc = function (docId) {
  state.selectedDocId = docId;
  elements.filterDocument.value = docId;
  switchView('facts');
  loadFacts();
};

function populateDocumentSelect(documents) {
  const current = state.selectedDocId || elements.filterDocument.value || '';
  elements.filterDocument.innerHTML = '<option value="">All Documents</option>' +
    documents
      .map(
        (doc) => `<option value="${escapeHtml(doc.source_doc_id)}" ${doc.source_doc_id === current ? 'selected' : ''}>${escapeHtml(doc.source_doc_id)} (${doc.fact_count})</option>`
      )
      .join('');
}

// ==========================================================================
// Facts View Operations
// ==========================================================================

async function ensureAllEntities() {
  if (state.allEntities && state.allEntities.length > 0) return;
  try {
    const res = await fetch(`${API_BASE}/facts`);
    if (res.ok) {
      const allFacts = await res.json();
      state.allEntities = Array.from(new Set(allFacts.map((f) => f.entity).filter(Boolean))).sort();
      populateEntitySelect();
    }
  } catch (err) {
    console.warn('Failed to prefetch master entity list:', err);
  }
}

async function loadFacts() {
  try {
    elements.factsContainer.innerHTML = '<div class="empty-state">Loading facts...</div>';

    const params = new URLSearchParams();
    if (state.selectedEntity) params.append('entity', state.selectedEntity);
    if (state.selectedDocId) params.append('source_doc_id', state.selectedDocId);

    const url = `${API_BASE}/facts?${params.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const facts = await res.json();
    state.facts = facts;

    // If master allEntities list has not been established yet or if this is an unfiltered fetch, establish it
    if (state.allEntities.length === 0 || (!state.selectedEntity && !state.selectedDocId)) {
      state.allEntities = Array.from(new Set(facts.map((f) => f.entity).filter(Boolean))).sort();
    }
    // Always render dropdown from the full master list of all distinct entities
    populateEntitySelect();
    applyFilters();
  } catch (err) {
    console.error('Failed to load facts:', err);
    elements.factsContainer.innerHTML = `<div class="empty-state" style="color: var(--color-contradicts);">Failed to load facts: ${escapeHtml(err.message)}</div>`;
  }
}

function populateEntitySelect() {
  const currentVal = state.selectedEntity || '';
  elements.filterEntity.innerHTML =
    '<option value="">All Entities</option>' +
    state.allEntities
      .map(
        (e) => `<option value="${escapeHtml(e)}" ${e === currentVal ? 'selected' : ''}>${escapeHtml(e)}</option>`
      )
      .join('');
}

function applyFilters() {
  let filtered = [...state.facts];

  if (state.searchQuery) {
    const q = state.searchQuery;
    filtered = filtered.filter(
      (f) =>
        (f.entity && f.entity.toLowerCase().includes(q)) ||
        (f.attribute && f.attribute.toLowerCase().includes(q)) ||
        (f.value && String(f.value).toLowerCase().includes(q)) ||
        (f.evidence_text && f.evidence_text.toLowerCase().includes(q)) ||
        (f.source_doc_id && f.source_doc_id.toLowerCase().includes(q))
    );
  }

  state.filteredFacts = filtered;
  renderFactsList(filtered);
}

function renderFactsList(facts) {
  elements.factsFilterCount.textContent = `Showing ${facts.length} of ${state.facts.length} facts`;

  if (facts.length === 0) {
    elements.factsContainer.innerHTML = `
      <div class="empty-state">
        <p>No facts match the selected filter criteria.</p>
      </div>
    `;
    return;
  }

  elements.factsContainer.innerHTML = facts
    .map((fact) => {
      const valueFormatted = formatFactValue(fact);
      const asOfHtml = fact.as_of
        ? `<span class="badge-as-of">As of: ${escapeHtml(fact.as_of)}</span>`
        : '';
      const inferredBadge = fact.extra?.as_of_inferred
        ? `<span class="badge-inferred">inferred</span>`
        : '';

      return `
        <article class="fact-card" data-id="${escapeHtml(fact.id)}">
          <div class="fact-header">
            <div class="fact-entity-attr">
              <span class="badge-entity">${escapeHtml(fact.entity)}</span>
              <span class="fact-attr">${escapeHtml(fact.attribute)}</span>
            </div>
            <div class="fact-header-right">
              ${asOfHtml}
              ${inferredBadge}
            </div>
          </div>

          <div class="fact-body">
            <span class="fact-value-hero">${valueFormatted}</span>
            ${fact.unit ? `<span class="fact-unit">${escapeHtml(fact.unit)}</span>` : ''}
            ${fact.scope ? `<span class="fact-scope">(${escapeHtml(fact.scope)})</span>` : ''}
          </div>

          <blockquote class="evidence-quote-block">
            <strong>Evidence Quote:</strong> &ldquo;${escapeHtml(fact.evidence_text)}&rdquo;
          </blockquote>

          <div class="fact-footer">
            <div class="fact-citation">
              <i data-lucide="book-open" class="btn-icon"></i>
              <span class="fact-doc-tag">${escapeHtml(fact.source_doc_id)}</span>
              <span>&bull; Page ${fact.page_number}</span>
            </div>
            <div>Confidence: ${(fact.confidence * 100).toFixed(0)}%</div>
          </div>
        </article>
      `;
    })
    .join('');

  initIcons();
}

function formatFactValue(fact) {
  if (fact.value === null || fact.value === undefined) return 'N/A';
  return escapeHtml(String(fact.value));
}

// ==========================================================================
// Relationships View Operations
// ==========================================================================

async function loadRelationships() {
  try {
    elements.relationshipsContainer.innerHTML = '<div class="empty-state">Loading cross-document relationships...</div>';

    const params = new URLSearchParams();
    if (state.selectedRelType) params.append('type', state.selectedRelType);

    const res = await fetch(`${API_BASE}/relationships?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const rels = await res.json();
    state.relationships = rels;
    renderRelationshipsList(rels);
  } catch (err) {
    console.error('Failed to load relationships:', err);
    elements.relationshipsContainer.innerHTML = `<div class="empty-state" style="color: var(--color-contradicts);">Failed to load relationships: ${escapeHtml(err.message)}</div>`;
  }
}

function renderRelationshipsList(relationships) {
  const typeLabel = state.selectedRelType 
    ? state.selectedRelType.replace(/_/g, ' ') 
    : 'all';
  elements.relationshipsFilterCount.textContent = `Showing ${relationships.length} ${typeLabel} relationships`;

  if (relationships.length === 0) {
    elements.relationshipsContainer.innerHTML = `
      <div class="empty-state">
        <p>No relationships found for the selected category.</p>
      </div>
    `;
    return;
  }

  elements.relationshipsContainer.innerHTML = relationships
    .map((rel) => {
      const typeClass = `badge-${rel.type}`;
      const typeText = rel.type.replace(/_/g, ' ').toUpperCase();

      return `
        <article class="relationship-card">
          <div class="rel-header">
            <span class="rel-badge ${typeClass}">
              ${typeText}
            </span>
            <span class="rel-confidence">Judgement Confidence: ${(rel.confidence * 100).toFixed(0)}%</span>
          </div>

          <div class="rel-comparison-grid">
            <!-- Fact A -->
            <div class="fact-panel fact-panel-a">
              <div class="panel-tag">Source Fact A</div>
              <div class="panel-entity-attr">
                <span class="panel-entity">${escapeHtml(rel.fact_a.entity)}</span>: ${escapeHtml(rel.fact_a.attribute)}
              </div>
              <div class="panel-value-row">
                <span class="panel-value">${escapeHtml(String(rel.fact_a.value))}</span>
                ${rel.fact_a.unit ? `<span class="fact-unit">${escapeHtml(rel.fact_a.unit)}</span>` : ''}
                ${rel.fact_a.as_of ? `<span class="badge-as-of">${escapeHtml(rel.fact_a.as_of)}</span>` : ''}
              </div>
              <div class="panel-citation">
                <i data-lucide="file-text" style="width: 12px; height: 12px; display: inline-block; vertical-align: middle;"></i>
                ${escapeHtml(rel.fact_a.source_doc_id)} (Page ${rel.fact_a.page_number})
              </div>
              <div class="panel-evidence">
                &ldquo;${escapeHtml(rel.fact_a.evidence_text)}&rdquo;
              </div>
            </div>

            <!-- Fact B -->
            <div class="fact-panel fact-panel-b">
              <div class="panel-tag">Source Fact B</div>
              <div class="panel-entity-attr">
                <span class="panel-entity">${escapeHtml(rel.fact_b.entity)}</span>: ${escapeHtml(rel.fact_b.attribute)}
              </div>
              <div class="panel-value-row">
                <span class="panel-value">${escapeHtml(String(rel.fact_b.value))}</span>
                ${rel.fact_b.unit ? `<span class="fact-unit">${escapeHtml(rel.fact_b.unit)}</span>` : ''}
                ${rel.fact_b.as_of ? `<span class="badge-as-of">${escapeHtml(rel.fact_b.as_of)}</span>` : ''}
              </div>
              <div class="panel-citation">
                <i data-lucide="file-text" style="width: 12px; height: 12px; display: inline-block; vertical-align: middle;"></i>
                ${escapeHtml(rel.fact_b.source_doc_id)} (Page ${rel.fact_b.page_number})
              </div>
              <div class="panel-evidence">
                &ldquo;${escapeHtml(rel.fact_b.evidence_text)}&rdquo;
              </div>
            </div>
          </div>

          <!-- Un-truncated Reasoning Section -->
          <div class="rel-reasoning-section">
            <div class="reasoning-heading">Cross-Document Reasoning & Analysis</div>
            <p class="reasoning-text">${escapeHtml(rel.reasoning)}</p>
          </div>
        </article>
      `;
    })
    .join('');

  initIcons();
}

// ==========================================================================
// Upload Workflow Operations
// ==========================================================================

function openUploadModal() {
  resetModal();
  elements.uploadModal.style.display = 'flex';
  initIcons();
}

function closeUploadModal() {
  elements.uploadModal.style.display = 'none';
  resetModal();
}

function resetModal() {
  state.selectedUploadFile = null;
  elements.fileInputModal.value = '';
  elements.modalFileName.textContent = 'Click or drag PDF here';
  elements.btnModalSubmit.disabled = true;

  elements.modalBodyIdle.style.display = 'block';
  elements.modalBodyProcessing.style.display = 'none';
  elements.modalBodyResult.style.display = 'none';

  elements.btnModalCancel.style.display = 'inline-flex';
  elements.btnModalCancel.textContent = 'Cancel';
  elements.btnModalSubmit.style.display = 'inline-flex';
  elements.btnModalViewFacts.style.display = 'none';

  elements.resultStatusBox.style.backgroundColor = 'var(--bg-corroborates)';
  elements.resultStatusBox.style.borderColor = 'var(--border-corroborates)';
  const icon = elements.resultStatusBox.querySelector('i');
  if (icon) {
    icon.setAttribute('data-lucide', 'check-circle');
    icon.style.color = 'var(--color-corroborates)';
  }
  elements.resultTitle.style.color = 'var(--color-corroborates)';
  initIcons();
}

function handleFileSelect(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.pdf')) {
    alert('Invalid file format. Please select a valid PDF file.');
    return;
  }

  state.selectedUploadFile = file;
  elements.modalFileName.textContent = file.name;
  elements.btnModalSubmit.disabled = false;
}

async function executeUpload() {
  if (!state.selectedUploadFile) return;

  const file = state.selectedUploadFile;
  const formData = new FormData();
  formData.append('file', file);

  // Transition modal to processing state
  elements.modalBodyIdle.style.display = 'none';
  elements.modalBodyProcessing.style.display = 'block';
  elements.btnModalSubmit.disabled = true;
  elements.btnModalCancel.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    const result = await res.json();

    if (!res.ok) {
      throw new Error(result.detail || `Upload failed with status ${res.status}`);
    }

    // Display result state
    elements.modalBodyProcessing.style.display = 'none';
    elements.modalBodyResult.style.display = 'block';
    elements.btnModalCancel.disabled = false;
    elements.btnModalCancel.textContent = 'Close';
    elements.btnModalSubmit.style.display = 'none';

    if (result.facts_extracted === 0) {
      elements.resultStatusBox.style.backgroundColor = 'var(--bg-reconciled)';
      elements.resultStatusBox.style.borderColor = 'var(--border-reconciled)';
      const icon = elements.resultStatusBox.querySelector('i');
      if (icon) {
        icon.setAttribute('data-lucide', 'alert-triangle');
        icon.style.color = 'var(--color-reconciled)';
      }
      elements.resultTitle.textContent = 'No Facts Extracted';
      elements.resultTitle.style.color = 'var(--color-reconciled)';
      elements.resultDesc.textContent =
        'Document processed but no facts were extracted — the PDF may have no extractable text or no factual content in a format the system recognizes.';
      elements.btnModalViewFacts.style.display = 'none';
    } else {
      elements.resultStatusBox.style.backgroundColor = 'var(--bg-corroborates)';
      elements.resultStatusBox.style.borderColor = 'var(--border-corroborates)';
      const icon = elements.resultStatusBox.querySelector('i');
      if (icon) {
        icon.setAttribute('data-lucide', 'check-circle');
        icon.style.color = 'var(--color-corroborates)';
      }
      elements.resultTitle.textContent = 'Pipeline Completed Successfully';
      elements.resultTitle.style.color = 'var(--color-corroborates)';
      elements.resultDesc.textContent = `Document '${result.doc_id}' parsed, stored, and cross-referenced.`;
      elements.btnModalViewFacts.style.display = 'inline-flex';
    }

    elements.resultMetrics.innerHTML = `
      <div><strong>Document ID:</strong> ${escapeHtml(result.doc_id)}</div>
      <div><strong>Facts Extracted & Upserted:</strong> ${result.facts_extracted}</div>
      <div><strong>Cross-Document Relationships Discovered:</strong> ${result.relationships_found}</div>
    `;

    if (result.errors && result.errors.length > 0) {
      elements.resultErrors.style.display = 'block';
      elements.resultErrors.innerHTML = `
        <strong>Extraction Notice:</strong> ${result.errors.length} chunk(s) encountered issues and were safely skipped.
      `;
    } else {
      elements.resultErrors.style.display = 'none';
    }

    elements.btnModalViewFacts.onclick = () => {
      closeUploadModal();
      filterByDoc(result.doc_id);
    };

    initIcons();

    // Refresh overall workspace state
    await loadStats();
    await loadFacts();
    await loadRelationships();
  } catch (err) {
    console.error('Upload failed:', err);
    elements.modalBodyProcessing.style.display = 'none';
    elements.modalBodyResult.style.display = 'block';
    elements.resultStatusBox.style.backgroundColor = 'var(--bg-contradicts)';
    elements.resultStatusBox.style.borderColor = 'var(--border-contradicts)';
    elements.resultStatusBox.querySelector('i').setAttribute('data-lucide', 'alert-circle');
    elements.resultStatusBox.querySelector('i').style.color = 'var(--color-contradicts)';
    elements.resultTitle.textContent = 'Upload Pipeline Failed';
    elements.resultTitle.style.color = 'var(--color-contradicts)';
    elements.resultDesc.textContent = err.message;
    elements.btnModalCancel.disabled = false;
    elements.btnModalCancel.textContent = 'Dismiss';
    elements.btnModalSubmit.style.display = 'none';
    initIcons();
  }
}

// ==========================================================================
// Utilities
// ==========================================================================
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
