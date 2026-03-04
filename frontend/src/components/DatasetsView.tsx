type DatasetRow = {
  id: number;
  name: string;
  description: string;
  category: string | null;
  model: string | null;
  exampleCount: number;
  generationCost: number;
  gradingCost: number;
  totalCost: number;
};

type ExamplePreview = {
  id: number | null;
  instruction: string;
  response: string;
};

type DatasetDetail = DatasetRow & {
  examplesPreview: ExamplePreview[];
  examplesPreviewOffset: number;
  examplesPreviewLimit: number;
};

type DatasetsViewProps = {
  datasets: DatasetRow[];
  datasetsError: string;
  trimmedSearch: string;
  categories: string[];
  activeCategory: string;
  onCategoryChange: (category: string) => void;
  selectedDataset: DatasetDetail | DatasetRow | null;
  onSelectDataset: (datasetId: number) => void;
  onRefresh: () => void;
  onExampleShift: (step: number) => void;
  onView: () => void;
  onExport: () => void;
  onDelete: () => void;
  onCopyId: () => void;
  onOpenDeleteConfirm: () => void;
  deletePending: boolean;
  isViewOpen: boolean;
  onCloseView: () => void;
  isDeleteConfirmOpen: boolean;
  onCloseDeleteConfirm: () => void;
};

export function DatasetsView({
  datasets,
  datasetsError,
  trimmedSearch,
  categories,
  activeCategory,
  onCategoryChange,
  selectedDataset,
  onSelectDataset,
  onRefresh,
  onExampleShift,
  onView,
  onExport,
  onDelete,
  onCopyId,
  onOpenDeleteConfirm,
  deletePending,
  isViewOpen,
  onCloseView,
  isDeleteConfirmOpen,
  onCloseDeleteConfirm,
}: DatasetsViewProps) {
  return (
    <>
      <div className="datasets-page-head">
        <p className="card-eyebrow">Datasets</p>
        <h2 className="placeholder-title">Dataset library</h2>
      </div>

      <section className="datasets-layout" aria-label="Dataset library">
        <div className="datasets-panel">
          <div className="datasets-toolbar">
            <p className="datasets-count">
              {datasetsError
                ? datasetsError
                : trimmedSearch
                  ? `${datasets.length} datasets matching "${trimmedSearch}"`
                  : `${datasets.length} datasets loaded from the current API`}
            </p>
            <div className="datasets-toolbar-actions">
              <button className="ghost-button ghost-button-compact" onClick={onRefresh} type="button">
                Refresh
              </button>
              <div className="filter-group">
              <button
                className={activeCategory === "all" ? "filter-pill filter-pill-active" : "filter-pill"}
                onClick={() => onCategoryChange("all")}
                type="button"
              >
                all
              </button>
              <button
                className={activeCategory === "uncategorized" ? "filter-pill filter-pill-active" : "filter-pill"}
                onClick={() => onCategoryChange("uncategorized")}
                type="button"
              >
                uncategorized
              </button>
              {categories.map((category) => (
                <button
                  key={category}
                  className={activeCategory === category ? "filter-pill filter-pill-active" : "filter-pill"}
                  onClick={() => onCategoryChange(category)}
                  type="button"
                >
                  {category.split("_").join(" ")}
                </button>
              ))}
              </div>
            </div>
          </div>

          <div className="dataset-list">
            {datasets.map((dataset) => (
              <button
                key={dataset.id}
                className={dataset.id === selectedDataset?.id ? "dataset-row dataset-row-active" : "dataset-row"}
                onClick={() => onSelectDataset(dataset.id)}
                type="button"
              >
                <div className="dataset-row-main">
                  <p className="dataset-row-name">{dataset.name}</p>
                  <p className="dataset-row-meta">
                    {dataset.category ?? "uncategorized"} - {dataset.model ?? "unknown model"}
                  </p>
                </div>
                <div className="dataset-row-side">
                  <span className="status-tag status-tag-ready">{dataset.exampleCount} examples</span>
                  <span className="dataset-row-updated">${dataset.totalCost.toFixed(2)} total cost</span>
                </div>
              </button>
            ))}
            {datasets.length === 0 && !datasetsError ? (
              <div className="empty-state">No datasets were returned from the current endpoint.</div>
            ) : null}
          </div>
        </div>

        <aside className="dataset-detail">
          <p className="card-eyebrow">Selected dataset</p>
          <h2 className="placeholder-title">{selectedDataset?.name ?? "No dataset selected"}</h2>

          <dl className="detail-grid">
            <div>
              <dt>Category</dt>
              <dd>{selectedDataset?.category ?? "-"}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{selectedDataset?.model ?? "-"}</dd>
            </div>
            <div>
              <dt>Examples</dt>
              <dd>{selectedDataset?.exampleCount ?? 0}</dd>
            </div>
            <div>
              <dt>Total Cost</dt>
              <dd>{selectedDataset ? `$${selectedDataset.totalCost.toFixed(2)}` : "$0.00"}</dd>
            </div>
            <div>
              <dt>Generation Cost</dt>
              <dd>{selectedDataset ? `$${selectedDataset.generationCost.toFixed(2)}` : "$0.00"}</dd>
            </div>
            <div>
              <dt>Grading Cost</dt>
              <dd>{selectedDataset ? `$${selectedDataset.gradingCost.toFixed(2)}` : "$0.00"}</dd>
            </div>
          </dl>

          <p className="card-body">
            {selectedDataset
              ? selectedDataset.description || "No description returned for this dataset."
              : "Choose a dataset from the list to inspect metadata, preview examples, or run actions."}
          </p>

          <div className="dataset-actions">
            <button className="ghost-button ghost-button-slim action-button" disabled={!selectedDataset} onClick={onView} type="button">
              <span className="action-button-mark" aria-hidden="true">
                i
              </span>
              <span>View</span>
            </button>
            <button className="ghost-button ghost-button-slim" disabled={!selectedDataset} onClick={onExport} type="button">
              Export
            </button>
            <button
              className="danger-button danger-button-slim"
              disabled={!selectedDataset || deletePending}
              onClick={onOpenDeleteConfirm}
              type="button"
            >
              {deletePending ? "Deleting..." : "Delete"}
            </button>
          </div>
        </aside>
      </section>

      {isViewOpen && selectedDataset ? (
        <div className="modal-backdrop" onClick={onCloseView} role="presentation">
          <section
            aria-label="Dataset detail modal"
            className="detail-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Dataset detail</p>
                <h2 className="placeholder-title">{selectedDataset.name}</h2>
              </div>
              <button className="ghost-button" onClick={onCloseView} type="button">
                Close
              </button>
            </div>

            <dl className="detail-grid">
              <div>
                <dt>Category</dt>
                <dd>{selectedDataset.category ?? "uncategorized"}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{selectedDataset.model ?? "-"}</dd>
              </div>
              <div>
                <dt>Examples</dt>
                <dd>{selectedDataset.exampleCount}</dd>
              </div>
              <div>
                <dt>Total Cost</dt>
                <dd>${selectedDataset.totalCost.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Generation Cost</dt>
                <dd>${selectedDataset.generationCost.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Grading Cost</dt>
                <dd>${selectedDataset.gradingCost.toFixed(2)}</dd>
              </div>
            </dl>

            <p className="card-body">{selectedDataset.description || "No description returned for this dataset."}</p>

            {"examplesPreview" in selectedDataset && selectedDataset.examplesPreview.length > 0 ? (
              <div className="example-preview-list">
                {selectedDataset.examplesPreview.map((example, index) => (
                  <article key={example.id ?? index} className="example-preview-card">
                    <p className="card-eyebrow">Example {selectedDataset.examplesPreviewOffset + index + 1}</p>
                    <p className="example-preview-label">Instruction</p>
                    <p className="card-body example-preview-body">{example.instruction}</p>
                    <p className="example-preview-label">Response</p>
                    <p className="card-body example-preview-body">{example.response}</p>
                  </article>
                ))}
                <div className="example-preview-footer">
                  <p className="dataset-row-meta">
                    Showing {selectedDataset.examplesPreviewOffset + 1}
                    -
                    {Math.min(
                      selectedDataset.examplesPreviewOffset + selectedDataset.examplesPreview.length,
                      selectedDataset.exampleCount,
                    )}{" "}
                    of {selectedDataset.exampleCount}
                  </p>
                  <div className="example-preview-actions">
                    <button
                      className="ghost-button ghost-button-compact"
                      disabled={selectedDataset.examplesPreviewOffset < selectedDataset.examplesPreviewLimit}
                      onClick={() => onExampleShift(-selectedDataset.examplesPreviewLimit)}
                      type="button"
                    >
                      Previous
                    </button>
                    <button
                      className="ghost-button ghost-button-compact"
                      disabled={
                        selectedDataset.examplesPreviewOffset + selectedDataset.examplesPreview.length >=
                        selectedDataset.exampleCount
                      }
                      onClick={() => onExampleShift(selectedDataset.examplesPreviewLimit)}
                      type="button"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            <div className="dataset-actions">
              <button className="ghost-button ghost-button-slim" onClick={onCopyId} type="button">
                Copy ID
              </button>
              <button className="ghost-button ghost-button-slim" onClick={onExport} type="button">
                Export
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {isDeleteConfirmOpen && selectedDataset ? (
        <div className="modal-backdrop" onClick={onCloseDeleteConfirm} role="presentation">
          <section
            aria-label="Delete dataset confirmation"
            className="detail-modal detail-modal-compact"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Confirm delete</p>
                <h2 className="placeholder-title">{selectedDataset.name}</h2>
              </div>
            </div>

            <p className="card-body">
              This will permanently remove the dataset and its examples from the local database.
            </p>

            <div className="dataset-actions">
              <button className="ghost-button" disabled={deletePending} onClick={onCloseDeleteConfirm} type="button">
                Cancel
              </button>
              <button className="danger-button" disabled={deletePending} onClick={onDelete} type="button">
                {deletePending ? "Deleting..." : "Delete dataset"}
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
