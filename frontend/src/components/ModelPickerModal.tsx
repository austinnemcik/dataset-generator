type ModelOption = {
  id: string;
  name: string;
};

type ModelPickerModalProps = {
  isOpen: boolean;
  options: ModelOption[];
  query: string;
  targetLabel: string;
  onClose: () => void;
  onQueryChange: (value: string) => void;
  onSelect: (modelId: string) => void;
};

export function ModelPickerModal({
  isOpen,
  options,
  query,
  targetLabel,
  onClose,
  onQueryChange,
  onSelect,
}: ModelPickerModalProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <div className="detail-modal model-picker-modal" aria-modal="true" role="dialog">
        <div className="detail-modal-header">
          <div>
            <p className="card-eyebrow">Model picker</p>
            <h3 className="placeholder-title">Choose a model for {targetLabel}</h3>
          </div>
          <button className="ghost-button ghost-button-compact" onClick={onClose} type="button">
            Close
          </button>
        </div>

        <label className="field export-picker-search">
          <span className="field-label">Search models</span>
          <input
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search by model id"
            type="text"
            value={query}
          />
        </label>

        <div className="model-picker-grid">
          {options.map((model) => (
            <button
              key={model.id}
              className="dataset-picker-chip model-picker-chip"
              onClick={() => onSelect(model.id)}
              type="button"
            >
              <span className="dataset-picker-id">{model.id}</span>
              <span className="dataset-picker-name">{model.name}</span>
            </button>
          ))}
          {options.length === 0 ? (
            <div className="empty-state">No cached models matched this search.</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
