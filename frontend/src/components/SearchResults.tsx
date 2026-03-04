type SearchSection = {
  id: string;
  label: string;
  items: Array<{
    id: string;
    title: string;
    meta: string;
    view: string;
    targetId?: number;
    targetRunId?: string;
    kind?: "dataset" | "run" | "document" | "export";
  }>;
};

type SearchResultsProps = {
  trimmedSearch: string;
  sections: SearchSection[];
  onSelect: (view: string, targetId?: number, targetRunId?: string) => void;
  onDatasetAction: (action: "view" | "export" | "delete", datasetId: number) => void;
};

export function SearchResults({ trimmedSearch, sections, onSelect, onDatasetAction }: SearchResultsProps) {
  if (!trimmedSearch) {
    return null;
  }

  return (
    <section className="search-panel" aria-label="Quick search results">
      {sections.length > 0 ? (
        sections.map((section) => (
          <div key={section.id} className="search-group">
            <p className="card-eyebrow">{section.label}</p>
            <div className="search-results">
              {section.items.map((item) => (
                <button
                  key={item.id}
                  className="search-result"
                  onClick={() => onSelect(item.view, item.targetId, item.targetRunId)}
                  type="button"
                >
                  <span className="search-result-content">
                    <span className="search-result-title">{item.title}</span>
                    <span className="search-result-meta">{item.meta}</span>
                  </span>
                  {item.kind === "dataset" && item.targetId ? (
                    <span className="search-result-actions">
                      <button
                        className="search-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          onDatasetAction("view", item.targetId as number);
                        }}
                        type="button"
                      >
                        View
                      </button>
                      <button
                        className="search-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          onDatasetAction("export", item.targetId as number);
                        }}
                        type="button"
                      >
                        Export
                      </button>
                      <button
                        className="search-action search-action-danger"
                        onClick={(event) => {
                          event.stopPropagation();
                          onDatasetAction("delete", item.targetId as number);
                        }}
                        type="button"
                      >
                        Delete
                      </button>
                    </span>
                  ) : null}
                </button>
              ))}
            </div>
          </div>
        ))
      ) : (
        <div className="empty-state">No matching datasets, runs, documents, or exports.</div>
      )}
    </section>
  );
}
