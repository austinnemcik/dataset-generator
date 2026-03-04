import { useMemo, useState } from "react";
import { ToggleField } from "./ToggleField";

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
  onRemoveExample: (exampleId: number) => void;
  onUpdateExample: (exampleId: number, instruction: string, response: string) => Promise<void>;
  onView: () => void;
  onExport: () => void;
  onDelete: () => void;
  onCopyId: () => void;
  onOpenTargetedMerge: () => void;
  onCloseTargetedMerge: () => void;
  onSubmitTargetedMerge: () => void;
  onToggleMergeDataset: (datasetId: number) => void;
  onClearMergeDatasets: () => void;
  onTargetedMergeDeleteOriginalsChange: (checked: boolean) => void;
  onTargetedMergeThresholdChange: (value: string) => void;
  onOpenGlobalMergeConfirm: () => void;
  onCloseGlobalMergeConfirm: () => void;
  onConfirmGlobalMerge: () => void;
  onGlobalMergeDeleteOriginalsChange: (checked: boolean) => void;
  onGlobalMergeThresholdChange: (value: string) => void;
  onOpenDeleteConfirm: () => void;
  deletePending: boolean;
  exampleDeletePendingId: number | null;
  exampleSavePendingId: number | null;
  mergePending: boolean;
  globalMergePending: boolean;
  isViewOpen: boolean;
  onCloseView: () => void;
  isDeleteConfirmOpen: boolean;
  onCloseDeleteConfirm: () => void;
  isTargetedMergeOpen: boolean;
  isGlobalMergeConfirmOpen: boolean;
  targetedMergeDatasetIds: number[];
  targetedMergeDeleteOriginals: boolean;
  targetedMergeThreshold: string;
  globalMergeDeleteOriginals: boolean;
  globalMergeThreshold: string;
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
  onRemoveExample,
  onUpdateExample,
  onView,
  onExport,
  onDelete,
  onCopyId,
  onOpenTargetedMerge,
  onCloseTargetedMerge,
  onSubmitTargetedMerge,
  onToggleMergeDataset,
  onClearMergeDatasets,
  onTargetedMergeDeleteOriginalsChange,
  onTargetedMergeThresholdChange,
  onOpenGlobalMergeConfirm,
  onCloseGlobalMergeConfirm,
  onConfirmGlobalMerge,
  onGlobalMergeDeleteOriginalsChange,
  onGlobalMergeThresholdChange,
  onOpenDeleteConfirm,
  deletePending,
  exampleDeletePendingId,
  exampleSavePendingId,
  mergePending,
  globalMergePending,
  isViewOpen,
  onCloseView,
  isDeleteConfirmOpen,
  onCloseDeleteConfirm,
  isTargetedMergeOpen,
  isGlobalMergeConfirmOpen,
  targetedMergeDatasetIds,
  targetedMergeDeleteOriginals,
  targetedMergeThreshold,
  globalMergeDeleteOriginals,
  globalMergeThreshold,
}: DatasetsViewProps) {
  const [mergeQuery, setMergeQuery] = useState("");
  const [pendingExampleRemoval, setPendingExampleRemoval] = useState<{
    id: number;
    label: string;
  } | null>(null);
  const [editingExample, setEditingExample] = useState<{
    id: number;
    label: string;
    instruction: string;
    response: string;
  } | null>(null);
  const [editDraft, setEditDraft] = useState({ instruction: "", response: "" });

  const filteredMergeDatasets = useMemo(() => {
    const query = mergeQuery.trim().toLowerCase();
    if (!query) {
      return datasets;
    }
    return datasets.filter(
      (dataset) =>
        dataset.name.toLowerCase().includes(query) ||
        String(dataset.id).includes(query) ||
        (dataset.category ?? "").toLowerCase().includes(query),
    );
  }, [datasets, mergeQuery]);

  return (
    <>
      <div className="datasets-page-head">
        <p className="card-eyebrow">Datasets</p>
        <h2 className="placeholder-title">Dataset library</h2>
      </div>

      <section className="dataset-library-merge-card" aria-label="Dataset library merge controls">
        <div className="field-toggle-head">
          <div>
            <span className="field-label">Merge Related Datasets</span>
            <p className="dataset-merge-copy">
              Scan the whole dataset library, merge related datasets into larger datasets, and optionally delete the originals after each merge.
            </p>
          </div>
          <button className="topbar-button" disabled={globalMergePending} onClick={onOpenGlobalMergeConfirm} type="button">
            {globalMergePending ? "Merging..." : "Merge All Related"}
          </button>
        </div>

        <div className="targeted-merge-controls">
          <label className="field field-inline field-compact dataset-threshold-field">
            <span className="field-label">Similarity Threshold</span>
            <input
              className="field-input field-input-number"
              max="1"
              min="0.1"
              onChange={(event) => onGlobalMergeThresholdChange(event.target.value)}
              step="0.05"
              type="number"
              value={globalMergeThreshold}
            />
          </label>

          <ToggleField
            checked={globalMergeDeleteOriginals}
            compact
            label="Delete originals after merge"
            onChange={onGlobalMergeDeleteOriginalsChange}
            tone="danger"
          />
        </div>
      </section>

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
            <button className="ghost-button ghost-button-slim" disabled={!selectedDataset} onClick={onOpenTargetedMerge} type="button">
              Targeted Merge
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
                    <div className="example-preview-head">
                      <p className="card-eyebrow">Example {selectedDataset.examplesPreviewOffset + index + 1}</p>
                      {typeof example.id === "number" ? (
                        <div className="example-preview-head-actions">
                          <button
                            className="ghost-button ghost-button-slim"
                            disabled={exampleSavePendingId === example.id}
                            onClick={() => {
                              setEditingExample({
                                id: example.id as number,
                                label: `Example ${selectedDataset.examplesPreviewOffset + index + 1}`,
                                instruction: example.instruction,
                                response: example.response,
                              });
                              setEditDraft({
                                instruction: example.instruction,
                                response: example.response,
                              });
                            }}
                            type="button"
                          >
                            {exampleSavePendingId === example.id ? "Saving..." : "Edit"}
                          </button>
                          <button
                            className="danger-button danger-button-slim"
                            disabled={exampleDeletePendingId === example.id}
                            onClick={() =>
                              setPendingExampleRemoval({
                                id: example.id as number,
                                label: `Example ${selectedDataset.examplesPreviewOffset + index + 1}`,
                              })
                            }
                            type="button"
                          >
                            {exampleDeletePendingId === example.id ? "Removing..." : "Remove"}
                          </button>
                        </div>
                      ) : null}
                    </div>
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

      {isTargetedMergeOpen ? (
        <div className="modal-backdrop" onClick={onCloseTargetedMerge} role="presentation">
          <section
            aria-label="Targeted merge modal"
            className="detail-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Targeted merge</p>
                <h2 className="placeholder-title">Pick exact datasets to merge</h2>
              </div>
              <button className="ghost-button" onClick={onCloseTargetedMerge} type="button">
                Close
              </button>
            </div>

            <div className="dataset-picker-card">
              <div className="dataset-picker-head">
                <p className="card-body">
                  Choose the exact datasets you want to combine. This uses the manual merge path instead of similarity discovery.
                </p>
              </div>
              <label className="field export-picker-search">
                <span className="field-label">Search datasets</span>
                <input
                  onChange={(event) => setMergeQuery(event.target.value)}
                  placeholder="Search by name, id, or category"
                  type="text"
                  value={mergeQuery}
                />
              </label>
              <div className="dataset-picker-grid">
                {filteredMergeDatasets.map((dataset: DatasetRow) => {
                  const isSelected = targetedMergeDatasetIds.includes(dataset.id);
                  return (
                    <button
                      key={dataset.id}
                      className={isSelected ? "dataset-picker-chip dataset-picker-chip-active" : "dataset-picker-chip"}
                      onClick={() => onToggleMergeDataset(dataset.id)}
                      type="button"
                    >
                      <span className="dataset-picker-id">#{dataset.id}</span>
                      <span className="dataset-picker-name">{dataset.name}</span>
                    </button>
                  );
                })}
                {filteredMergeDatasets.length === 0 ? (
                  <div className="empty-state">No datasets matched this search.</div>
                ) : null}
              </div>
            </div>

            <div className="targeted-merge-controls">
              <label className="field field-inline field-compact dataset-threshold-field">
                <span className="field-label">Similarity Threshold</span>
                <input
                  className="field-input field-input-number"
                  max="1"
                  min="0.1"
                  onChange={(event) => onTargetedMergeThresholdChange(event.target.value)}
                  step="0.05"
                  type="number"
                  value={targetedMergeThreshold}
                />
              </label>

              <ToggleField
                checked={targetedMergeDeleteOriginals}
                compact
                label="Delete originals after merge"
                onChange={onTargetedMergeDeleteOriginalsChange}
                tone="danger"
              />
            </div>

            <div className="dataset-picker-actions">
              <button className="ghost-button ghost-button-compact" onClick={onClearMergeDatasets} type="button">
                Clear Selection
              </button>
              <button
                className="topbar-button"
                disabled={mergePending || targetedMergeDatasetIds.length < 2}
                onClick={onSubmitTargetedMerge}
                type="button"
              >
                {mergePending ? "Merging..." : `Merge ${targetedMergeDatasetIds.length} datasets`}
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

      {isGlobalMergeConfirmOpen ? (
        <div className="modal-backdrop" onClick={onCloseGlobalMergeConfirm} role="presentation">
          <section
            aria-label="Merge all related datasets confirmation"
            className="detail-modal detail-modal-compact"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Confirm library merge</p>
                <h2 className="placeholder-title">Merge all related datasets</h2>
              </div>
            </div>

            <p className="card-body">
              This will scan the full dataset library for related datasets and merge the matching groups into bigger datasets.
              {globalMergeDeleteOriginals
                ? " The original datasets will be deleted after each successful merge."
                : " The original datasets will be kept."}
            </p>

            <div className="dataset-actions">
              <button
                className="ghost-button"
                disabled={globalMergePending}
                onClick={onCloseGlobalMergeConfirm}
                type="button"
              >
                Cancel
              </button>
              <button
                className="danger-button"
                disabled={globalMergePending}
                onClick={onConfirmGlobalMerge}
                type="button"
              >
                {globalMergePending ? "Merging..." : "Merge library"}
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {pendingExampleRemoval ? (
        <div className="modal-backdrop" onClick={() => setPendingExampleRemoval(null)} role="presentation">
          <section
            aria-label="Remove example confirmation"
            className="detail-modal detail-modal-compact"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Remove example</p>
                <h2 className="placeholder-title">{pendingExampleRemoval.label}</h2>
              </div>
            </div>

            <p className="card-body">
              This removes just this example from the dataset so you can curate out responses you do not want to keep.
            </p>

            <div className="dataset-actions">
              <button
                className="ghost-button"
                disabled={exampleDeletePendingId === pendingExampleRemoval.id}
                onClick={() => setPendingExampleRemoval(null)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="danger-button"
                disabled={exampleDeletePendingId === pendingExampleRemoval.id}
                onClick={() => {
                  void onRemoveExample(pendingExampleRemoval.id);
                  setPendingExampleRemoval(null);
                }}
                type="button"
              >
                {exampleDeletePendingId === pendingExampleRemoval.id ? "Removing..." : "Remove example"}
              </button>
            </div>
          </section>
        </div>
      ) : null}

      {editingExample ? (
        <div
          className="modal-backdrop"
          onClick={() => {
            if (exampleSavePendingId !== editingExample.id) {
              setEditingExample(null);
            }
          }}
          role="presentation"
        >
          <section
            aria-label="Edit training example"
            className="detail-modal"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Edit example</p>
                <h2 className="placeholder-title">{editingExample.label}</h2>
              </div>
            </div>

            <p className="card-body">
              Make small instruction or response adjustments here without removing the example from the dataset.
            </p>

            <div className="example-edit-grid">
              <label className="field">
                <span className="field-label">Instruction</span>
                <textarea
                  onChange={(event) => setEditDraft((current) => ({ ...current, instruction: event.target.value }))}
                  rows={5}
                  value={editDraft.instruction}
                />
              </label>

              <label className="field">
                <span className="field-label">Response</span>
                <textarea
                  onChange={(event) => setEditDraft((current) => ({ ...current, response: event.target.value }))}
                  rows={7}
                  value={editDraft.response}
                />
              </label>
            </div>

            <div className="dataset-actions">
              <button
                className="ghost-button"
                disabled={exampleSavePendingId === editingExample.id}
                onClick={() => setEditingExample(null)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="topbar-button"
                disabled={
                  exampleSavePendingId === editingExample.id ||
                  !editDraft.instruction.trim() ||
                  !editDraft.response.trim() ||
                  (editDraft.instruction.trim() === editingExample.instruction.trim() &&
                    editDraft.response.trim() === editingExample.response.trim())
                }
                onClick={() => {
                  void onUpdateExample(editingExample.id, editDraft.instruction, editDraft.response)
                    .then(() => setEditingExample(null))
                    .catch(() => undefined);
                }}
                type="button"
              >
                {exampleSavePendingId === editingExample.id ? "Saving..." : "Save changes"}
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
