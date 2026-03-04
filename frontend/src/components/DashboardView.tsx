type DashboardCard = {
  eyebrow: string;
  title: string;
  body: string;
};

type DashboardRun = {
  runId: string;
  topic: string | null;
  status: string;
  saved: number;
  failed: number;
};

type DashboardDataset = {
  id: number;
  name: string;
  category: string | null;
  exampleCount: number;
};

type DashboardDocument = {
  id: number;
  name: string;
  fileType: string;
  chunkCount: number;
};

type DashboardExport = {
  id: number;
  name: string;
  format: string;
  totalExamples: number;
};

type DashboardViewProps = {
  cards: DashboardCard[];
  loading: boolean;
  error: string;
  recentRuns: DashboardRun[];
  recentDatasets: DashboardDataset[];
  recentDocuments: DashboardDocument[];
  recentExports: DashboardExport[];
  attentionItems: string[];
  onOpenRun: (runId: string) => void;
  onOpenDataset: (datasetId: number) => void;
  onOpenDocument: (documentId: number) => void;
  onOpenExports: () => void;
};

function DashboardSection({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="dashboard-section-card">
      <p className="card-eyebrow">{eyebrow}</p>
      <h2 className="dashboard-section-title">{title}</h2>
      {children}
    </section>
  );
}

export function DashboardView({
  cards,
  loading,
  error,
  recentRuns,
  recentDatasets,
  recentDocuments,
  recentExports,
  attentionItems,
  onOpenRun,
  onOpenDataset,
  onOpenDocument,
  onOpenExports,
}: DashboardViewProps) {
  return (
    <>
      <section className="dashboard-grid" aria-label="Dashboard cards">
        {cards.map((card, index) => (
          <article
            key={`${card.eyebrow}-${index}`}
            className={
              loading
                ? index < 2
                  ? "dashboard-card dashboard-card-priority dashboard-card-loading"
                  : "dashboard-card dashboard-card-loading"
                : index < 2
                  ? "dashboard-card dashboard-card-priority"
                  : "dashboard-card"
            }
          >
            <p className="card-eyebrow">{card.eyebrow}</p>
            <h2 className="card-title">{card.title}</h2>
            <p className="card-body">{card.body}</p>
          </article>
        ))}
        {error ? <div className="dashboard-grid-note">{error}</div> : null}
      </section>

      <section className="dashboard-section-grid" aria-label="Dashboard activity">
        <DashboardSection eyebrow="Attention" title="Needs attention">
          {attentionItems.length > 0 ? (
            <div className="dashboard-activity-list">
              {attentionItems.map((item) => (
                <article key={item} className="dashboard-activity-row dashboard-activity-row-static">
                  <p className="dashboard-activity-title">{item}</p>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state empty-state-rich">
              <p className="card-eyebrow">Healthy</p>
              <h3 className="empty-state-title">Nothing urgent right now</h3>
              <p className="empty-state-copy">Recent runs, datasets, exports, and documents all look stable.</p>
            </div>
          )}
        </DashboardSection>

        <DashboardSection eyebrow="Recent runs" title="Latest generation activity">
          {recentRuns.length > 0 ? (
            <div className="dashboard-activity-list">
              {recentRuns.map((run) => (
                <button key={run.runId} className="dashboard-activity-row" onClick={() => onOpenRun(run.runId)} type="button">
                  <div>
                    <p className="dashboard-activity-title">{run.topic || run.runId}</p>
                    <p className="dashboard-activity-meta">
                      {run.saved} saved - {run.failed} failed
                    </p>
                  </div>
                  <span className={`status-tag status-tag-${run.status}`}>{run.status}</span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">No batch runs have landed yet.</div>
          )}
        </DashboardSection>

        <DashboardSection eyebrow="Recent datasets" title="Newest data to inspect">
          {recentDatasets.length > 0 ? (
            <div className="dashboard-activity-list">
              {recentDatasets.map((dataset) => (
                <button key={dataset.id} className="dashboard-activity-row" onClick={() => onOpenDataset(dataset.id)} type="button">
                  <div>
                    <p className="dashboard-activity-title">{dataset.name}</p>
                    <p className="dashboard-activity-meta">
                      {dataset.category ?? "uncategorized"} - {dataset.exampleCount} examples
                    </p>
                  </div>
                  <span className="status-tag status-tag-ready">dataset {dataset.id}</span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">No datasets are loaded yet.</div>
          )}
        </DashboardSection>

        <DashboardSection eyebrow="Documents" title="Source material library">
          {recentDocuments.length > 0 ? (
            <div className="dashboard-activity-list">
              {recentDocuments.map((document) => (
                <button key={document.id} className="dashboard-activity-row" onClick={() => onOpenDocument(document.id)} type="button">
                  <div>
                    <p className="dashboard-activity-title">{document.name}</p>
                    <p className="dashboard-activity-meta">
                      {document.fileType} - {document.chunkCount} chunks
                    </p>
                  </div>
                  <span className="status-tag status-tag-ready">doc {document.id}</span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">No source documents are stored yet.</div>
          )}
        </DashboardSection>

        <DashboardSection eyebrow="Exports" title="Recent artifacts">
          {recentExports.length > 0 ? (
            <div className="dashboard-activity-list">
              {recentExports.map((exportRow) => (
                <button key={exportRow.id} className="dashboard-activity-row" onClick={onOpenExports} type="button">
                  <div>
                    <p className="dashboard-activity-title">{exportRow.name}</p>
                    <p className="dashboard-activity-meta">
                      {exportRow.format} - {exportRow.totalExamples} examples
                    </p>
                  </div>
                  <span className="status-tag status-tag-ready">export {exportRow.id}</span>
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">No exports have been created yet.</div>
          )}
        </DashboardSection>
      </section>
    </>
  );
}
import type { ReactNode } from "react";
