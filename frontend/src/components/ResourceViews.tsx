type DocumentRow = {
  id: number;
  name: string;
  file_type: string;
  char_count: number;
  chunk_count: number;
  created_at: string | null;
  source_material_ref: string;
};

type DocumentChunk = {
  id: number;
  chunk_index: number;
  char_count: number;
  content: string;
};

type DocumentDetail = {
  document: DocumentRow;
  chunks: DocumentChunk[];
};

type ExportRow = {
  id: number;
  status: string;
  export_format: string;
  dataset_ids: number[];
  total_examples: number;
  train_examples: number;
  val_examples: number;
  output_filename: string | null;
  has_artifact: boolean;
  created_at: string | null;
};

type DocumentsViewProps = {
  documents: DocumentRow[];
  documentsError: string;
  documentsLoading: boolean;
  documentFilter: string;
  selectedDocumentId: number;
  selectedDocumentDetail: DocumentDetail | null;
  documentDetailLoading: boolean;
  documentActionPending: boolean;
  documentChunksExpanded: boolean;
  isDocumentViewOpen: boolean;
  isDocumentDeleteConfirmOpen: boolean;
  onSelectDocument: (documentId: number) => void;
  onRefresh: () => void;
  onView: () => void;
  onCloseView: () => void;
  onOpenDeleteConfirm: () => void;
  onCloseDeleteConfirm: () => void;
  onDelete: () => void;
  onCopyRef: () => void;
  onCopyChunkContent: (content: string) => void;
  onDocumentFilterChange: (value: string) => void;
  uploadPending: boolean;
  scraperPending: boolean;
  uploadMessage: string;
  scraperMessage: string;
  uploadFileName: string;
  uploadMode: "examples" | "source_material" | "pretraining_data";
  uploadAdvancedOpen: boolean;
  uploadChunkSize: string;
  uploadChunkOverlap: string;
  scraperText: string;
  scraperDatasetName: string;
  onUploadFileChange: (file: File | null) => void;
  onUploadModeChange: (mode: "examples" | "source_material" | "pretraining_data") => void;
  onUploadAdvancedToggle: () => void;
  onUploadChunkSizeChange: (value: string) => void;
  onUploadChunkOverlapChange: (value: string) => void;
  onUploadSubmit: () => void;
  onToggleChunkExpansion: () => void;
  onScraperTextChange: (value: string) => void;
  onScraperDatasetNameChange: (value: string) => void;
  onScraperSubmit: () => void;
};

type ExportsViewProps = {
  exportsHistory: ExportRow[];
  exportsError: string;
  exportsLoading: boolean;
  formatDate: (value: string | null) => string;
  exportActionPendingId: number | null;
  exportCreatePending: boolean;
  exportMessage: string;
  exportDatasetIds: string;
  selectedExportDatasetIds: number[];
  exportFormat: "sharegpt" | "chatml" | "alpaca";
  exportMinScore: string;
  exportMaxExamples: string;
  exportTrainValSplit: string;
  exportDedupePass: boolean;
  exportShuffle: boolean;
  availableDatasets: Array<{ id: number; name: string }>;
  exportPickerQuery: string;
  isExportPickerOpen: boolean;
  onExportFieldChange: (
    field: "datasetIds" | "format" | "minScore" | "maxExamples" | "trainValSplit" | "dedupePass" | "shuffle",
    value: string | boolean,
  ) => void;
  onExportPickerQueryChange: (value: string) => void;
  onOpenExportPicker: () => void;
  onCloseExportPicker: () => void;
  onToggleExportDataset: (datasetId: number) => void;
  onClearExportDatasets: () => void;
  onExportSubmit: () => void;
  onDownload: (exportId: number) => void;
  onRerun: (exportId: number) => void;
};

function compactDocumentDate(value: string | null) {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function DocumentsView({
  documents,
  documentsError,
  documentsLoading,
  documentFilter,
  selectedDocumentId,
  selectedDocumentDetail,
  documentDetailLoading,
  documentActionPending,
  documentChunksExpanded,
  isDocumentViewOpen,
  isDocumentDeleteConfirmOpen,
  onSelectDocument,
  onRefresh,
  onView,
  onCloseView,
  onOpenDeleteConfirm,
  onCloseDeleteConfirm,
  onDelete,
  onCopyRef,
  onCopyChunkContent,
  onDocumentFilterChange,
  uploadPending,
  scraperPending,
  uploadMessage,
  scraperMessage,
  uploadFileName,
  uploadMode,
  uploadAdvancedOpen,
  uploadChunkSize,
  uploadChunkOverlap,
  scraperText,
  scraperDatasetName,
  onUploadFileChange,
  onUploadModeChange,
  onUploadAdvancedToggle,
  onUploadChunkSizeChange,
  onUploadChunkOverlapChange,
  onUploadSubmit,
  onToggleChunkExpansion,
  onScraperTextChange,
  onScraperDatasetNameChange,
  onScraperSubmit,
}: DocumentsViewProps) {
  const selectedDocument = documents.find((document) => document.id === selectedDocumentId) ?? null;

  return (
    <section className="placeholder-panel" aria-label="Documents">
      <div className="generation-panel-header">
        <div>
          <p className="card-eyebrow">Documents</p>
          <h2 className="placeholder-title">Source material library</h2>
        </div>
        <button className="ghost-button ghost-button-compact" onClick={onRefresh} type="button">
          Refresh
        </button>
      </div>

      <label className="field document-filter-field">
        <span className="field-label">Filter Documents</span>
        <input
          onChange={(event) => onDocumentFilterChange(event.target.value)}
          placeholder="Search by name, file type, or source ref"
          type="text"
          value={documentFilter}
        />
      </label>

      {documentsLoading ? (
        <div className="loading-stack" aria-label="Loading documents">
          <div className="loading-card">
            <div className="loading-line loading-line-strong" />
            <div className="loading-line" />
          </div>
          <div className="loading-card">
            <div className="loading-line loading-line-strong" />
            <div className="loading-line" />
          </div>
        </div>
      ) : null}

      <div className="document-list">
        {documents.map((document) => (
          <button
            key={document.id}
            className={document.id === selectedDocumentId ? "document-card document-card-active" : "document-card"}
            onClick={() => onSelectDocument(document.id)}
            type="button"
          >
            <div>
              <p className="dataset-row-name">{document.name}</p>
              <p className="dataset-row-meta">
                {document.file_type} - {document.chunk_count} chunks - {document.char_count.toLocaleString()} chars
              </p>
            </div>
            <div className="document-card-side">
              <p className="dataset-row-updated">{document.source_material_ref}</p>
              <p className="dataset-row-meta">{compactDocumentDate(document.created_at)}</p>
            </div>
          </button>
        ))}
        {!documentsLoading && documents.length === 0 ? (
          <div className="empty-state">{documentsError || "No source documents are stored yet."}</div>
        ) : null}
      </div>

      {selectedDocument ? (
        <div className="document-actions">
          <button className="ghost-button ghost-button-compact action-button" onClick={onView} type="button">
            <span className="action-button-mark" aria-hidden="true">
              i
            </span>
            <span>View</span>
          </button>
          <button className="ghost-button ghost-button-compact" onClick={onCopyRef} type="button">
            Copy Ref
          </button>
          <button className="danger-button danger-button-slim" onClick={onOpenDeleteConfirm} type="button">
            Delete
          </button>
        </div>
      ) : null}

      <div className="document-intake-grid">
        <section className="document-intake-card">
          <div className="panel-heading">
            <p className="card-eyebrow">File intake</p>
            <h3 className="generation-detail-title">Upload a document</h3>
            <p className="document-intake-copy">
              Send a file straight into source material, pre-training chunks, or example import without leaving the workspace.
            </p>
          </div>

          <label className="field">
            <span className="field-label">Intake Mode</span>
            <select value={uploadMode} onChange={(event) => onUploadModeChange(event.target.value as "examples" | "source_material" | "pretraining_data")}>
              <option value="source_material">Source Material</option>
              <option value="pretraining_data">Pre-training Data</option>
              <option value="examples">Examples</option>
            </select>
          </label>

          <label className="field">
            <span className="field-label">File</span>
            <input
              onChange={(event) => onUploadFileChange(event.target.files?.[0] ?? null)}
              type="file"
            />
          </label>

          <p className="generation-section-note">{uploadFileName || "No file selected yet."}</p>

          <button className="ghost-button ghost-button-compact document-advanced-toggle" onClick={onUploadAdvancedToggle} type="button">
            {uploadAdvancedOpen ? "Hide advanced" : "Show advanced"}
          </button>

          {uploadAdvancedOpen ? (
            <div className="document-advanced-grid">
              <label className="field">
                <span className="field-label">Chunk Size</span>
                <input
                  onChange={(event) => onUploadChunkSizeChange(event.target.value)}
                  type="number"
                  value={uploadChunkSize}
                />
              </label>
              <label className="field">
                <span className="field-label">Chunk Overlap</span>
                <input
                  onChange={(event) => onUploadChunkOverlapChange(event.target.value)}
                  type="number"
                  value={uploadChunkOverlap}
                />
              </label>
            </div>
          ) : null}

          <div className="generation-actions">
            <button className="topbar-button" disabled={uploadPending || !uploadFileName} onClick={onUploadSubmit} type="button">
              <span className="topbar-button-mark" aria-hidden="true">
                +
              </span>
              <span>{uploadPending ? "Uploading..." : "Upload File"}</span>
            </button>
          </div>

          {uploadMessage ? <p className="inline-message">{uploadMessage}</p> : null}
        </section>

        <section className="document-intake-card">
          <div className="panel-heading">
            <p className="card-eyebrow">Scraper intake</p>
            <h3 className="generation-detail-title">Paste normalized text</h3>
            <p className="document-intake-copy">
              Useful for scraper output or one-off research notes that already exist as plain text.
            </p>
          </div>

          <label className="field">
            <span className="field-label">Dataset Name</span>
            <input
              onChange={(event) => onScraperDatasetNameChange(event.target.value)}
              type="text"
              value={scraperDatasetName}
            />
          </label>

          <label className="field">
            <span className="field-label">Scraped Text</span>
            <textarea
              onChange={(event) => onScraperTextChange(event.target.value)}
              rows={7}
              value={scraperText}
            />
          </label>

          <div className="generation-actions">
            <button className="topbar-button" disabled={scraperPending || !scraperText.trim()} onClick={onScraperSubmit} type="button">
              <span className="topbar-button-mark" aria-hidden="true">
                +
              </span>
              <span>{scraperPending ? "Importing..." : "Import Text"}</span>
            </button>
          </div>

          {scraperMessage ? <p className="inline-message">{scraperMessage}</p> : null}
        </section>
      </div>

      {isDocumentViewOpen && selectedDocument ? (
        <div className="modal-backdrop" role="presentation">
          <div className="detail-modal" aria-modal="true" role="dialog">
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Document detail</p>
                <h3 className="placeholder-title">{selectedDocument.name}</h3>
                <p className="card-body">Use the source ref in generation, or lift specific chunks directly into notes and prompts.</p>
              </div>
              <button className="ghost-button ghost-button-compact" onClick={onCloseView} type="button">
                Close
              </button>
            </div>

            <dl className="detail-grid">
              <div>
                <dt>File Type</dt>
                <dd>{selectedDocument.file_type}</dd>
              </div>
              <div>
                <dt>Source Ref</dt>
                <dd>{selectedDocument.source_material_ref}</dd>
              </div>
              <div>
                <dt>Characters</dt>
                <dd>{selectedDocument.char_count.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Chunks</dt>
                <dd>{selectedDocument.chunk_count}</dd>
              </div>
            </dl>

            <div className="document-actions">
              <button className="ghost-button ghost-button-compact" onClick={onCopyRef} type="button">
                Copy source ref
              </button>
              {selectedDocumentDetail?.chunks[0] ? (
                <button
                  className="ghost-button ghost-button-compact"
                  onClick={() => onCopyChunkContent(selectedDocumentDetail.chunks[0].content)}
                  type="button"
                >
                  Copy first chunk
                </button>
              ) : null}
              {selectedDocumentDetail && selectedDocumentDetail.chunks.length > 8 ? (
                <button className="ghost-button ghost-button-compact" onClick={onToggleChunkExpansion} type="button">
                  {documentChunksExpanded ? "Show fewer chunks" : "Show more chunks"}
                </button>
              ) : null}
            </div>

            <div className="generation-section-head document-section-head">
              <p className="card-eyebrow">Chunk preview</p>
              <p className="generation-section-note">
                {documentDetailLoading
                  ? "Loading chunks..."
                  : selectedDocumentDetail
                    ? `${selectedDocumentDetail.chunks.length} chunks loaded`
                    : "No chunk detail loaded"}
              </p>
            </div>

            {documentDetailLoading ? (
              <div className="document-loading-card">Loading document detail...</div>
            ) : null}

            {!documentDetailLoading && selectedDocumentDetail ? (
              <div className="example-preview-list">
                {selectedDocumentDetail.chunks.slice(0, documentChunksExpanded ? 20 : 8).map((chunk) => (
                  <article key={chunk.id} className="example-preview-card">
                    <p className="example-preview-label">
                      Chunk {chunk.chunk_index} - {chunk.char_count.toLocaleString()} chars
                    </p>
                    <p className="example-preview-body">{chunk.content}</p>
                    <div className="generation-event-actions">
                      <button
                        className="ghost-button ghost-button-compact"
                        onClick={() => onCopyChunkContent(chunk.content)}
                        type="button"
                      >
                        Copy chunk
                      </button>
                    </div>
                  </article>
                ))}
                {selectedDocumentDetail.chunks.length > (documentChunksExpanded ? 20 : 8) ? (
                  <p className="inline-message">
                    Showing the first {documentChunksExpanded ? 20 : 8} chunks. The document contains {selectedDocumentDetail.chunks.length} chunks total.
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {isDocumentDeleteConfirmOpen && selectedDocument ? (
        <div className="modal-backdrop" role="presentation">
          <div className="detail-modal detail-modal-compact" aria-modal="true" role="dialog">
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Delete document</p>
                <h3 className="placeholder-title">{selectedDocument.name}</h3>
              </div>
              <button className="ghost-button ghost-button-compact" onClick={onCloseDeleteConfirm} type="button">
                Close
              </button>
            </div>

            <p className="card-body">
              This removes the stored document and all of its chunks. The source reference will stop working anywhere it is used.
            </p>

            <div className="dataset-actions">
              <button className="ghost-button ghost-button-compact" onClick={onCloseDeleteConfirm} type="button">
                Cancel
              </button>
              <button
                className="danger-button danger-button-slim"
                disabled={documentActionPending}
                onClick={onDelete}
                type="button"
              >
                {documentActionPending ? "Deleting..." : "Delete document"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function ExportsView({
  exportsHistory,
  exportsError,
  exportsLoading,
  formatDate,
  exportActionPendingId,
  exportCreatePending,
  exportMessage,
  exportDatasetIds,
  selectedExportDatasetIds,
  exportFormat,
  exportMinScore,
  exportMaxExamples,
  exportTrainValSplit,
  exportDedupePass,
  exportShuffle,
  availableDatasets,
  exportPickerQuery,
  isExportPickerOpen,
  onExportFieldChange,
  onExportPickerQueryChange,
  onOpenExportPicker,
  onCloseExportPicker,
  onToggleExportDataset,
  onClearExportDatasets,
  onExportSubmit,
  onDownload,
  onRerun,
}: ExportsViewProps) {
  return (
    <section className="placeholder-panel" aria-label="Exports">
      <div className="generation-panel-header">
        <div>
          <p className="card-eyebrow">Exports</p>
          <h2 className="placeholder-title">Export history</h2>
        </div>
      </div>

      <section className="document-intake-card export-builder-card">
        <div className="panel-heading">
          <p className="card-eyebrow">Create export</p>
          <h3 className="generation-detail-title">Build a fresh artifact</h3>
          <p className="document-intake-copy">
            Pick datasets, shape the export, and generate a new artifact without leaving history view.
          </p>
        </div>

        <div className="dataset-picker-card">
          <div className="generation-section-head dataset-picker-head">
            <p className="card-eyebrow">Dataset picker</p>
            <p className="generation-section-note">{selectedExportDatasetIds.length} selected</p>
          </div>
          <div className="dataset-picker-grid">
            {availableDatasets.slice(0, 18).map((dataset) => {
              const isSelected = selectedExportDatasetIds.includes(dataset.id);
              return (
                <button
                  key={dataset.id}
                  className={isSelected ? "dataset-picker-chip dataset-picker-chip-active" : "dataset-picker-chip"}
                  onClick={() => onToggleExportDataset(dataset.id)}
                  type="button"
                >
                  <span className="dataset-picker-id">{dataset.id}</span>
                  <span className="dataset-picker-name">{dataset.name}</span>
                </button>
              );
            })}
          </div>
          <div className="dataset-picker-actions">
            <button className="ghost-button ghost-button-compact" onClick={onOpenExportPicker} type="button">
              Browse All
            </button>
            <button
              className="ghost-button ghost-button-compact"
              disabled={selectedExportDatasetIds.length === 0}
              onClick={onClearExportDatasets}
              type="button"
            >
              Clear All
            </button>
          </div>
        </div>

        <div className="settings-grid export-builder-grid">
          <label className="field">
            <span className="field-label">Dataset IDs</span>
            <textarea
              onChange={(event) => onExportFieldChange("datasetIds", event.target.value)}
              rows={4}
              value={exportDatasetIds}
            />
          </label>

          <label className="field">
            <span className="field-label">Format</span>
            <select
              onChange={(event) => onExportFieldChange("format", event.target.value)}
              value={exportFormat}
            >
              <option value="sharegpt">ShareGPT</option>
              <option value="chatml">ChatML</option>
              <option value="alpaca">Alpaca</option>
            </select>
          </label>

          <label className="field">
            <span className="field-label">Min Score</span>
            <input
              onChange={(event) => onExportFieldChange("minScore", event.target.value)}
              placeholder="optional"
              step="0.1"
              type="number"
              value={exportMinScore}
            />
          </label>

          <label className="field">
            <span className="field-label">Max Examples</span>
            <input
              onChange={(event) => onExportFieldChange("maxExamples", event.target.value)}
              placeholder="optional"
              type="number"
              value={exportMaxExamples}
            />
          </label>

          <label className="field">
            <span className="field-label">Train / Val Split</span>
            <input
              onChange={(event) => onExportFieldChange("trainValSplit", event.target.value)}
              placeholder="optional"
              step="0.01"
              type="number"
              value={exportTrainValSplit}
            />
          </label>

          <div className="export-toggle-group">
            <ToggleField
              checked={exportDedupePass}
              compact
              label="Dedupe pass"
              onChange={(checked) => onExportFieldChange("dedupePass", checked)}
            />
            <ToggleField
              checked={exportShuffle}
              compact
              label="Shuffle"
              onChange={(checked) => onExportFieldChange("shuffle", checked)}
            />
          </div>
        </div>

        <div className="available-id-strip">
          {availableDatasets.slice(0, 8).map((dataset) => (
            <span key={dataset.id} className="status-tag status-tag-ready">
              {dataset.id}: {dataset.name}
            </span>
          ))}
        </div>

        <div className="generation-actions">
          <button className="topbar-button" disabled={exportCreatePending || !exportDatasetIds.trim()} onClick={onExportSubmit} type="button">
            <span className="topbar-button-mark" aria-hidden="true">
              +
            </span>
            <span>{exportCreatePending ? "Exporting..." : "Create Export"}</span>
          </button>
        </div>

      {exportMessage ? <p className="inline-message">{exportMessage}</p> : null}
      </section>

      {exportsLoading ? (
        <div className="loading-stack" aria-label="Loading exports">
          <div className="loading-card">
            <div className="loading-line loading-line-strong" />
            <div className="loading-line" />
          </div>
          <div className="loading-card">
            <div className="loading-line loading-line-strong" />
            <div className="loading-line" />
          </div>
        </div>
      ) : null}

      <div className="document-list">
        {exportsHistory.map((exportRow) => (
          <article key={exportRow.id} className="document-card">
            <div>
              <p className="dataset-row-name">{exportRow.output_filename || `export-${exportRow.id}`}</p>
              <p className="dataset-row-meta">
                {exportRow.export_format} - {exportRow.total_examples} examples - {exportRow.dataset_ids.length} datasets
              </p>
            </div>
            <div className="document-card-side">
              <p className="dataset-row-updated">
                {exportRow.has_artifact ? "artifact ready" : "no artifact"} - {formatDate(exportRow.created_at)}
              </p>
              <div className="document-actions">
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={!exportRow.has_artifact}
                  onClick={() => onDownload(exportRow.id)}
                  type="button"
                >
                  Download
                </button>
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={exportActionPendingId === exportRow.id}
                  onClick={() => onRerun(exportRow.id)}
                  type="button"
                >
                  {exportActionPendingId === exportRow.id ? "Rerunning..." : "Rerun"}
                </button>
              </div>
            </div>
          </article>
        ))}
        {!exportsLoading && exportsHistory.length === 0 ? (
          <div className="empty-state">{exportsError || "No export history records are available yet."}</div>
        ) : null}
      </div>

      {isExportPickerOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="detail-modal export-picker-modal" aria-modal="true" role="dialog">
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Dataset picker</p>
                <h3 className="placeholder-title">Select datasets for export</h3>
              </div>
              <button className="ghost-button ghost-button-compact" onClick={onCloseExportPicker} type="button">
                Close
              </button>
            </div>

            <div className="export-picker-toolbar">
              <label className="field export-picker-search">
                <span className="field-label">Search datasets</span>
                <input
                  onChange={(event) => onExportPickerQueryChange(event.target.value)}
                  placeholder="Search by name or id"
                  type="text"
                  value={exportPickerQuery}
                />
              </label>

              <div className="dataset-picker-actions">
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={selectedExportDatasetIds.length === 0}
                  onClick={onClearExportDatasets}
                  type="button"
                >
                  Clear All
                </button>
              </div>
            </div>

            <div className="generation-section-head dataset-picker-head">
              <p className="generation-section-note">{availableDatasets.length} datasets shown</p>
              <p className="generation-section-note">{selectedExportDatasetIds.length} selected</p>
            </div>

            <div className="dataset-picker-grid dataset-picker-grid-expanded">
              {availableDatasets.map((dataset) => {
                const isSelected = selectedExportDatasetIds.includes(dataset.id);
                return (
                  <button
                    key={dataset.id}
                    className={isSelected ? "dataset-picker-chip dataset-picker-chip-active" : "dataset-picker-chip"}
                    onClick={() => onToggleExportDataset(dataset.id)}
                    type="button"
                  >
                    <span className="dataset-picker-id">{dataset.id}</span>
                    <span className="dataset-picker-name">{dataset.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function SettingsPlaceholderView() {
  return null;
}

type SettingsValues = {
  default_model: string;
  grading_model: string;
  naming_model: string;
  threshold: number;
  min_grading_score: number;
  min_response_char_length: number;
  max_grading_json_retries: number;
  max_naming_json_retries: number;
  max_low_quality_retries: number;
  max_generation_retries: number;
  min_save_ratio: number;
};

type SettingsViewProps = {
  settings: SettingsValues | null;
  initialSettings: SettingsValues | null;
  loading: boolean;
  saving: boolean;
  error: string;
  validationError: string;
  message: string;
  onFieldChange: (field: keyof SettingsValues, value: string) => void;
  onOpenModelPicker: (field: "default_model" | "grading_model" | "naming_model") => void;
  onReset: () => void;
  onSave: () => void;
};

export function SettingsView({
  settings,
  initialSettings,
  loading,
  saving,
  error,
  validationError,
  message,
  onFieldChange,
  onOpenModelPicker,
  onReset,
  onSave,
}: SettingsViewProps) {
  const isDirty = settings !== null && initialSettings !== null && JSON.stringify(settings) !== JSON.stringify(initialSettings);

  return (
    <section className="placeholder-panel" aria-label="Settings">
      <p className="card-eyebrow">Settings</p>
      <h2 className="placeholder-title">Workspace settings</h2>
      <p className="card-body">Update the persisted backend defaults that drive generation, grading, naming, and ingest behavior.</p>

      {loading ? <p className="inline-message">Loading settings...</p> : null}
      {error ? <p className="inline-message">{error}</p> : null}
      {validationError ? <p className="inline-message">{validationError}</p> : null}
      {message ? <p className="inline-message">{message}</p> : null}

      {settings ? (
        <div className="settings-grid">
          <label className="field">
            <span className="field-label">Default Model</span>
            <input value={settings.default_model} onChange={(event) => onFieldChange("default_model", event.target.value)} type="text" />
            <span className="field-hint">Default model used when a batch does not specify one explicitly.</span>
            <button className="ghost-button ghost-button-compact field-picker-button" onClick={() => onOpenModelPicker("default_model")} type="button">
              Browse models
            </button>
          </label>
          <label className="field">
            <span className="field-label">Grading Model</span>
            <input value={settings.grading_model} onChange={(event) => onFieldChange("grading_model", event.target.value)} type="text" />
            <span className="field-hint">Model used to score generated examples before ingest decisions are made.</span>
            <button className="ghost-button ghost-button-compact field-picker-button" onClick={() => onOpenModelPicker("grading_model")} type="button">
              Browse models
            </button>
          </label>
          <label className="field">
            <span className="field-label">Naming Model</span>
            <input value={settings.naming_model} onChange={(event) => onFieldChange("naming_model", event.target.value)} type="text" />
            <span className="field-hint">Model used to generate dataset names and descriptions during save.</span>
            <button className="ghost-button ghost-button-compact field-picker-button" onClick={() => onOpenModelPicker("naming_model")} type="button">
              Browse models
            </button>
          </label>
          <label className="field">
            <span className="field-label">Duplicate Threshold</span>
            <input value={settings.threshold} onChange={(event) => onFieldChange("threshold", event.target.value)} type="number" step="0.01" />
            <span className="field-hint">Higher values make deduplication stricter when comparing embeddings.</span>
          </label>
          <label className="field">
            <span className="field-label">Min Grading Score</span>
            <input value={settings.min_grading_score} onChange={(event) => onFieldChange("min_grading_score", event.target.value)} type="number" step="0.1" />
            <span className="field-hint">Minimum dataset score required before the run can be saved.</span>
          </label>
          <label className="field">
            <span className="field-label">Min Response Length</span>
            <input value={settings.min_response_char_length} onChange={(event) => onFieldChange("min_response_char_length", event.target.value)} type="number" />
            <span className="field-hint">Shortest acceptable response length before quality checks reject an example.</span>
          </label>
          <label className="field">
            <span className="field-label">Max Grading JSON Retries</span>
            <input value={settings.max_grading_json_retries} onChange={(event) => onFieldChange("max_grading_json_retries", event.target.value)} type="number" />
            <span className="field-hint">Extra attempts when the grading response comes back malformed.</span>
          </label>
          <label className="field">
            <span className="field-label">Max Naming JSON Retries</span>
            <input value={settings.max_naming_json_retries} onChange={(event) => onFieldChange("max_naming_json_retries", event.target.value)} type="number" />
            <span className="field-hint">Extra attempts when dataset naming metadata fails to parse.</span>
          </label>
          <label className="field">
            <span className="field-label">Max Low Quality Retries</span>
            <input value={settings.max_low_quality_retries} onChange={(event) => onFieldChange("max_low_quality_retries", event.target.value)} type="number" />
            <span className="field-hint">Recovery attempts allowed when a generation is graded as low quality.</span>
          </label>
          <label className="field">
            <span className="field-label">Max Generation Retries</span>
            <input value={settings.max_generation_retries} onChange={(event) => onFieldChange("max_generation_retries", event.target.value)} type="number" />
            <span className="field-hint">Extra tries when generation fails or the returned JSON cannot be used.</span>
          </label>
          <label className="field">
            <span className="field-label">Min Save Ratio</span>
            <input value={settings.min_save_ratio} onChange={(event) => onFieldChange("min_save_ratio", event.target.value)} type="number" step="0.01" />
            <span className="field-hint">Portion of accepted examples needed before the run is worth saving.</span>
          </label>
        </div>
      ) : null}

      {settings && initialSettings ? (
        <div className="settings-status-row">
          <span className={isDirty ? "status-tag status-tag-review" : "status-tag status-tag-ready"}>
            {isDirty ? "unsaved changes" : "saved"}
          </span>
          <p className="generation-section-note">
            {isDirty
              ? "You have local edits that have not been written back to the backend yet."
              : "The form matches the currently persisted backend settings."}
          </p>
        </div>
      ) : null}

      <div className="generation-actions">
        <button className="ghost-button" disabled={!settings || saving || !isDirty} onClick={onReset} type="button">
          Reset
        </button>
        <button className="topbar-button" disabled={!settings || saving || Boolean(validationError)} onClick={onSave} type="button">
          <span className="topbar-button-mark" aria-hidden="true">
            +
          </span>
          <span>{saving ? "Saving..." : "Save Settings"}</span>
        </button>
      </div>
    </section>
  );
}
import { ToggleField } from "./ToggleField";
