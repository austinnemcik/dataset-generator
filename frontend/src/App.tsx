import { useEffect, useMemo, useState } from "react";
import "./styles.css";

type NavItem = {
  id: string;
  label: string;
  eyebrow: string;
  title: string;
  description: string;
};

type DatasetRow = {
  id: number;
  name: string;
  description: string;
  category: string | null;
  model: string | null;
  generationCost: number;
  gradingCost: number;
  totalCost: number;
};

type BatchRunRow = {
  run_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "paused";
  requested_runs: number;
  saved: number;
  failed: number;
  queued: number;
  running: number;
  topic: string | null;
  requested_agent: string | null;
  created_at: string | null;
};

type DocumentRow = {
  id: number;
  name: string;
  file_type: string;
  char_count: number;
  chunk_count: number;
  created_at: string | null;
  source_material_ref: string;
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

type DashboardStats = {
  datasets: number;
  training_examples: number;
  embedding_time: number;
  ingest_time: number;
  grading_time: number;
  api_cost: number;
};

type BatchGenerateResponse = {
  success: boolean;
  message: string;
  data?: {
    batch_run_ids?: string[];
  };
};

type HealthState = "checking" | "healthy" | "offline";

const navItems: NavItem[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    eyebrow: "Dashboard",
    title: "Hi, Austin",
    description: "A compact overview of the current local workspace and pipeline metrics.",
  },
  {
    id: "datasets",
    label: "Datasets",
    eyebrow: "Datasets",
    title: "Dataset library",
    description: "Browse the current dataset inventory with the metadata available from the existing API surface.",
  },
  {
    id: "generation",
    label: "Generation",
    eyebrow: "Generation",
    title: "Generation workspace",
    description: "Launch a batch with the current backend API and review recent runs from persisted batch history.",
  },
  {
    id: "documents",
    label: "Documents",
    eyebrow: "Documents",
    title: "Source documents",
    description: "Review source material already ingested into the document store.",
  },
  {
    id: "exports",
    label: "Exports",
    eyebrow: "Exports",
    title: "Export artifacts",
    description: "View export history and the artifact records already persisted by the backend.",
  },
  {
    id: "settings",
    label: "Settings",
    eyebrow: "Settings",
    title: "Workspace settings",
    description: "This area still needs a backend settings endpoint before it can become functional.",
  },
];

const emptyDashboardCards = [
  { eyebrow: "Datasets", title: "--", body: "Loading total dataset count." },
  { eyebrow: "Examples", title: "--", body: "Loading stored training examples." },
  { eyebrow: "Embedding", title: "--", body: "Loading average embedding time." },
  { eyebrow: "Ingest", title: "--", body: "Loading average ingest time." },
  { eyebrow: "Grading", title: "--", body: "Loading average grading time." },
  { eyebrow: "API Cost", title: "--", body: "Loading total API cost." },
];

const defaultGenerationForm = {
  topics: "Code review and debugging\nSecurity vulnerability identification",
  agentTypes: "qa, instruction_following, adversarial",
  amount: "120",
  exAmt: "25",
  model: "google/gemini-2.5-flash",
  maxConcurrency: "25",
};

function formatDate(value: string | null): string {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString();
}

function parseDatasetPayload(raw: string): DatasetRow | null {
  try {
    const parsed = JSON.parse(raw) as {
      dataset?: Array<Record<string, string | number | null>>;
    };
    const rows = parsed.dataset ?? [];
    const lookup = Object.assign({}, ...rows);
    return {
      id: Number(lookup.id ?? 0),
      name: String(lookup.name ?? "Unnamed dataset"),
      description: String(lookup.description ?? ""),
      category: lookup.category ? String(lookup.category) : null,
      model: lookup.model ? String(lookup.model) : null,
      generationCost: Number(lookup.generation_cost ?? 0),
      gradingCost: Number(lookup.grading_cost ?? 0),
      totalCost: Number(lookup.total_cost ?? 0),
    };
  } catch {
    return null;
  }
}

function App() {
  const [activeView, setActiveView] = useState<string>("dashboard");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number>(0);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [dashboardError, setDashboardError] = useState<string>("");
  const [healthState, setHealthState] = useState<HealthState>("checking");
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [datasetsError, setDatasetsError] = useState<string>("");
  const [batchRuns, setBatchRuns] = useState<BatchRunRow[]>([]);
  const [batchRunsError, setBatchRunsError] = useState<string>("");
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [documentsError, setDocumentsError] = useState<string>("");
  const [exportsHistory, setExportsHistory] = useState<ExportRow[]>([]);
  const [exportsError, setExportsError] = useState<string>("");
  const [generationForm, setGenerationForm] = useState(defaultGenerationForm);
  const [generationMessage, setGenerationMessage] = useState<string>("");
  const [generationSubmitting, setGenerationSubmitting] = useState<boolean>(false);

  const currentView = useMemo(
    () => navItems.find((item) => item.id === activeView) ?? navItems[0],
    [activeView],
  );

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) ?? datasets[0] ?? null,
    [datasets, selectedDatasetId],
  );

  const statCards = useMemo(() => {
    if (!dashboardStats) {
      return emptyDashboardCards.map((card) => ({
        ...card,
        body: dashboardError || card.body,
      }));
    }

    return [
      {
        eyebrow: "Datasets",
        title: dashboardStats.datasets.toLocaleString(),
        body: "Datasets currently available in the local workspace.",
      },
      {
        eyebrow: "Examples",
        title: dashboardStats.training_examples.toLocaleString(),
        body: "Stored training examples across all datasets.",
      },
      {
        eyebrow: "Embedding",
        title: `${dashboardStats.embedding_time.toFixed(2)}s`,
        body: "Average embedding completion time from benchmark logs.",
      },
      {
        eyebrow: "Ingest",
        title: `${dashboardStats.ingest_time.toFixed(2)}s`,
        body: "Average ingest API duration from benchmark logs.",
      },
      {
        eyebrow: "Grading",
        title: `${dashboardStats.grading_time.toFixed(2)}s`,
        body: "Average grading completion time from benchmark logs.",
      },
      {
        eyebrow: "API Cost",
        title: `$${dashboardStats.api_cost.toFixed(2)}`,
        body: "Accumulated API cost recorded in the benchmark summary.",
      },
    ];
  }, [dashboardError, dashboardStats]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboardStats() {
      try {
        setDashboardError("");
        const response = await fetch("/api/dashboard/", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Dashboard request failed with ${response.status}`);
        }
        const payload = (await response.json()) as DashboardStats;
        setDashboardStats(payload);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setDashboardError(error instanceof Error ? error.message : "Unable to load dashboard stats.");
      }
    }

    void loadDashboardStats();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        setHealthState("checking");
        const response = await fetch("/api/health", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Health request failed with ${response.status}`);
        }
        setHealthState("healthy");
      } catch {
        if (!controller.signal.aborted) {
          setHealthState("offline");
        }
      }
    }

    void loadHealth();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDatasets() {
      try {
        setDatasetsError("");
        const response = await fetch("/api/dataset/amount/25", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Dataset request failed with ${response.status}`);
        }

        const payload = (await response.json()) as {
          datasets?: string[];
        };
        const rows = (payload.datasets ?? [])
          .map(parseDatasetPayload)
          .filter((dataset): dataset is DatasetRow => dataset !== null);
        setDatasets(rows);
        if (rows.length > 0) {
          setSelectedDatasetId((current) => current || rows[0].id);
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setDatasetsError(error instanceof Error ? error.message : "Unable to load datasets.");
      }
    }

    void loadDatasets();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadBatchRuns() {
      try {
        setBatchRunsError("");
        const response = await fetch("/api/dataset/batch", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Batch request failed with ${response.status}`);
        }

        const payload = (await response.json()) as {
          data?: {
            runs?: BatchRunRow[];
          };
        };
        setBatchRuns(payload.data?.runs ?? []);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setBatchRunsError(error instanceof Error ? error.message : "Unable to load batch runs.");
      }
    }

    void loadBatchRuns();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDocuments() {
      try {
        setDocumentsError("");
        const response = await fetch("/api/dataset/documents", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Document request failed with ${response.status}`);
        }

        const payload = (await response.json()) as {
          data?: {
            documents?: DocumentRow[];
          };
        };
        setDocuments(payload.data?.documents ?? []);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setDocumentsError(error instanceof Error ? error.message : "Unable to load source documents.");
      }
    }

    void loadDocuments();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadExports() {
      try {
        setExportsError("");
        const response = await fetch("/api/dataset/exports/history", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Export request failed with ${response.status}`);
        }

        const payload = (await response.json()) as {
          data?: {
            exports?: ExportRow[];
          };
        };
        setExportsHistory(payload.data?.exports ?? []);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setExportsError(error instanceof Error ? error.message : "Unable to load export history.");
      }
    }

    void loadExports();
    return () => controller.abort();
  }, []);

  async function handleBatchLaunch() {
    try {
      setGenerationSubmitting(true);
      setGenerationMessage("");

      const topics = generationForm.topics
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean);
      const agentTypes = generationForm.agentTypes
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);

      const response = await fetch("/api/dataset/batch/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          amount: Number(generationForm.amount),
          topics,
          agent_types: agentTypes,
          ex_amt: Number(generationForm.exAmt),
          random_agent: false,
          max_concurrency: Number(generationForm.maxConcurrency),
          max_retries: 1,
          retry_backoff_seconds: 2,
          model: generationForm.model,
        }),
      });

      const payload = (await response.json()) as BatchGenerateResponse;
      if (!response.ok || !payload.success) {
        throw new Error(payload.message || `Batch request failed with ${response.status}`);
      }

      const batchIds = payload.data?.batch_run_ids ?? [];
      setGenerationMessage(
        batchIds.length > 0
          ? `Batch queued successfully. Run IDs: ${batchIds.join(", ")}`
          : payload.message,
      );

      const refreshedRuns = await fetch("/api/dataset/batch");
      if (refreshedRuns.ok) {
        const refreshedPayload = (await refreshedRuns.json()) as {
          data?: {
            runs?: BatchRunRow[];
          };
        };
        setBatchRuns(refreshedPayload.data?.runs ?? []);
      }
    } catch (error) {
      setGenerationMessage(error instanceof Error ? error.message : "Unable to launch batch.");
    } finally {
      setGenerationSubmitting(false);
    }
  }

  return (
    <main className="dashboard-shell">
      <div className="dashboard-noise" aria-hidden="true" />

      <aside className="sidebar">
        <div className="sidebar-brand">
          <p className="sidebar-kicker">PData</p>
          <strong className="sidebar-title">Control Room</strong>
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={item.id === activeView ? "sidebar-link sidebar-link-active" : "sidebar-link"}
              onClick={() => setActiveView(item.id)}
              type="button"
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <section className="dashboard-main">
        <header className="topbar">
          <label className="topbar-search" aria-label="Search">
            <span className="topbar-search-label">Quick Search</span>
            <input placeholder="Search datasets, runs, documents" type="text" />
          </label>

          <div className="topbar-actions">
            <button className="topbar-button" onClick={() => setActiveView("generation")} type="button">
              <span className="topbar-button-mark" aria-hidden="true">
                +
              </span>
              <span>New Batch</span>
            </button>
          </div>
        </header>

        <header className="dashboard-header">
          <div>
            <p className="dashboard-kicker">{currentView.eyebrow}</p>
            <h1 className="dashboard-title">{currentView.title}</h1>
            <p className="dashboard-copy">{currentView.description}</p>
          </div>

          <div
            className={
              healthState === "healthy"
                ? "dashboard-badge dashboard-badge-healthy"
                : healthState === "offline"
                  ? "dashboard-badge dashboard-badge-offline"
                  : "dashboard-badge dashboard-badge-checking"
            }
          >
            <span className="status-dot" aria-hidden="true" />
            {healthState === "healthy"
              ? "API Healthy"
              : healthState === "offline"
                ? "API Offline"
                : "Checking API"}
          </div>
        </header>

        {activeView === "dashboard" ? (
          <section className="dashboard-grid" aria-label="Dashboard cards">
            {statCards.map((card, index) => (
              <article key={`${card.eyebrow}-${index}`} className="dashboard-card">
                <p className="card-eyebrow">{card.eyebrow}</p>
                <h2 className="card-title">{card.title}</h2>
                <p className="card-body">{card.body}</p>
              </article>
            ))}
          </section>
        ) : null}

        {activeView === "datasets" ? (
          <section className="datasets-layout" aria-label="Dataset library">
            <div className="datasets-panel">
              <div className="datasets-toolbar">
                <p className="datasets-count">
                  {datasetsError ? datasetsError : `${datasets.length} datasets loaded from the current API`}
                </p>
              </div>

              <div className="dataset-list">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    className={dataset.id === selectedDataset?.id ? "dataset-row dataset-row-active" : "dataset-row"}
                    onClick={() => setSelectedDatasetId(dataset.id)}
                    type="button"
                  >
                    <div className="dataset-row-main">
                      <p className="dataset-row-name">{dataset.name}</p>
                      <p className="dataset-row-meta">
                        {dataset.category ?? "uncategorized"} - {dataset.model ?? "unknown model"}
                      </p>
                    </div>
                    <div className="dataset-row-side">
                      <span className="status-tag status-tag-ready">loaded</span>
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
                  <dt>Generation Cost</dt>
                  <dd>${selectedDataset?.generationCost.toFixed(2) ?? "0.00"}</dd>
                </div>
                <div>
                  <dt>Grading Cost</dt>
                  <dd>${selectedDataset?.gradingCost.toFixed(2) ?? "0.00"}</dd>
                </div>
              </dl>

              <p className="card-body">{selectedDataset?.description || "No description returned for this dataset."}</p>
            </aside>
          </section>
        ) : null}

        {activeView === "generation" ? (
          <section className="generation-layout" aria-label="Generation workspace">
            <form
              className="generation-form-panel"
              onSubmit={(event) => {
                event.preventDefault();
                void handleBatchLaunch();
              }}
            >
              <div className="panel-heading">
                <p className="card-eyebrow">Batch setup</p>
                <h2 className="placeholder-title">Start a generation run</h2>
              </div>

              <div className="form-grid">
                <label className="field">
                  <span className="field-label">Topics</span>
                  <textarea
                    onChange={(event) => setGenerationForm((current) => ({ ...current, topics: event.target.value }))}
                    rows={4}
                    value={generationForm.topics}
                  />
                </label>

                <label className="field">
                  <span className="field-label">Agent Types</span>
                  <input
                    onChange={(event) =>
                      setGenerationForm((current) => ({ ...current, agentTypes: event.target.value }))
                    }
                    type="text"
                    value={generationForm.agentTypes}
                  />
                </label>

                <label className="field">
                  <span className="field-label">Amount</span>
                  <input
                    onChange={(event) => setGenerationForm((current) => ({ ...current, amount: event.target.value }))}
                    type="number"
                    value={generationForm.amount}
                  />
                </label>

                <label className="field">
                  <span className="field-label">Examples per Dataset</span>
                  <input
                    onChange={(event) => setGenerationForm((current) => ({ ...current, exAmt: event.target.value }))}
                    type="number"
                    value={generationForm.exAmt}
                  />
                </label>

                <label className="field">
                  <span className="field-label">Model</span>
                  <input
                    onChange={(event) => setGenerationForm((current) => ({ ...current, model: event.target.value }))}
                    type="text"
                    value={generationForm.model}
                  />
                </label>

                <label className="field">
                  <span className="field-label">Max Concurrency</span>
                  <input
                    onChange={(event) =>
                      setGenerationForm((current) => ({ ...current, maxConcurrency: event.target.value }))
                    }
                    type="number"
                    value={generationForm.maxConcurrency}
                  />
                </label>
              </div>

              <div className="generation-actions">
                <button className="topbar-button" disabled={generationSubmitting} type="submit">
                  <span className="topbar-button-mark" aria-hidden="true">
                    +
                  </span>
                  <span>{generationSubmitting ? "Launching..." : "Launch Batch"}</span>
                </button>
                <button
                  className="ghost-button"
                  onClick={() => setGenerationForm(defaultGenerationForm)}
                  type="button"
                >
                  Reset
                </button>
              </div>

              {generationMessage ? <p className="inline-message">{generationMessage}</p> : null}
            </form>

            <aside className="generation-side-panel">
              <div className="panel-heading">
                <p className="card-eyebrow">Recent runs</p>
                <h2 className="placeholder-title">Run monitor</h2>
              </div>

              <div className="run-list">
                {batchRuns.map((run) => (
                  <article key={run.run_id} className="run-card">
                    <div className="run-card-header">
                      <div>
                        <p className="run-id">{run.run_id}</p>
                        <h3 className="run-topic">{run.topic || "Untitled batch run"}</h3>
                      </div>
                      <span className={`status-tag status-tag-${run.status}`}>{run.status}</span>
                    </div>

                    <p className="run-meta">{run.requested_agent || "mixed agents"}</p>
                    <p className="card-body">
                      {run.saved} saved · {run.failed} failed · {run.running} running · {run.queued} queued
                    </p>
                  </article>
                ))}
                {batchRuns.length === 0 ? (
                  <div className="empty-state">{batchRunsError || "No batch runs have been recorded yet."}</div>
                ) : null}
              </div>
            </aside>
          </section>
        ) : null}

        {activeView === "documents" ? (
          <section className="placeholder-panel" aria-label="Documents">
            <p className="card-eyebrow">Documents</p>
            <h2 className="placeholder-title">Source material library</h2>
            <div className="document-list">
              {documents.map((document) => (
                <article key={document.id} className="document-card">
                  <div>
                    <p className="dataset-row-name">{document.name}</p>
                    <p className="dataset-row-meta">
                      {document.file_type} - {document.chunk_count} chunks - {document.char_count.toLocaleString()} chars
                    </p>
                  </div>
                  <p className="dataset-row-updated">{document.source_material_ref}</p>
                </article>
              ))}
              {documents.length === 0 ? (
                <div className="empty-state">{documentsError || "No source documents are stored yet."}</div>
              ) : null}
            </div>
          </section>
        ) : null}

        {activeView === "exports" ? (
          <section className="placeholder-panel" aria-label="Exports">
            <p className="card-eyebrow">Exports</p>
            <h2 className="placeholder-title">Export history</h2>
            <div className="document-list">
              {exportsHistory.map((exportRow) => (
                <article key={exportRow.id} className="document-card">
                  <div>
                    <p className="dataset-row-name">{exportRow.output_filename || `export-${exportRow.id}`}</p>
                    <p className="dataset-row-meta">
                      {exportRow.export_format} - {exportRow.total_examples} examples - {exportRow.dataset_ids.length} datasets
                    </p>
                  </div>
                  <p className="dataset-row-updated">
                    {exportRow.has_artifact ? "artifact ready" : "no artifact"} · {formatDate(exportRow.created_at)}
                  </p>
                </article>
              ))}
              {exportsHistory.length === 0 ? (
                <div className="empty-state">{exportsError || "No export history records are available yet."}</div>
              ) : null}
            </div>
          </section>
        ) : null}

        {activeView === "settings" ? (
          <section className="placeholder-panel" aria-label="Settings">
            <p className="card-eyebrow">Settings</p>
            <h2 className="placeholder-title">Backend settings endpoint needed</h2>
            <p className="card-body">
              This screen is the one remaining gap. There is no existing API surface for writable settings or
              environment-backed preferences, so this page should stay a placeholder until that endpoint exists.
            </p>
          </section>
        ) : null}
      </section>
    </main>
  );
}

export default App;
