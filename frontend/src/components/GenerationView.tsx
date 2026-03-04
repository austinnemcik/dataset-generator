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
  updated_at: string | null;
  completed_at: string | null;
};

type BatchRunResult = {
  index: number;
  run_id: string;
  dataset_id: number | null;
  status: "saved" | "failed";
  topic: string | null;
  agent: string | null;
  error: string | null;
};

type BatchRunDetail = {
  batch_run_id: string;
  status: BatchRunRow["status"];
  requested_runs: number;
  saved: number;
  failed: number;
  queued: number;
  running: number;
  topic: string | null;
  requested_agent: string | null;
  random_agent: boolean;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  results: BatchRunResult[];
};

type BatchStreamEvent = {
  id: string;
  runId: string;
  datasetId: number | null;
  status: string;
  topic: string | null;
  agent: string | null;
  score: number | null;
  cost: number | null;
  category: string | null;
  error: string | null;
};

type GenerationFormState = {
  topics: string;
  agentTypes: string;
  amount: string;
  exAmt: string;
  model: string;
  maxConcurrency: string;
};

type ModelOption = {
  id: string;
  name: string;
};

type GenerationViewProps = {
  form: GenerationFormState;
  onFormChange: (updater: (current: GenerationFormState) => GenerationFormState) => void;
  onSubmit: () => void;
  onReset: () => void;
  submitting: boolean;
  message: string;
  runs: BatchRunRow[];
  runsError: string;
  selectedRunId: string;
  selectedRunDetail: BatchRunDetail | null;
  detailLoading: boolean;
  streamEvents: BatchStreamEvent[];
  streamStatus: "idle" | "connecting" | "live" | "offline";
  detailError: string;
  actionPending: string;
  modelOptions: ModelOption[];
  modelLoading: boolean;
  onOpenModelPicker: () => void;
  onOpenDataset: (datasetId: number) => void;
  onSelectRun: (runId: string) => void;
  onRefreshRuns: () => void;
  onBatchAction: (action: "pause" | "resume" | "stop" | "restart-failed") => void;
};

function compactDate(value: string | null): string {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function percentage(saved: number, failed: number, requested: number) {
  if (requested <= 0) {
    return 0;
  }
  return Math.min(100, ((saved + failed) / requested) * 100);
}

function streamLabel(streamStatus: GenerationViewProps["streamStatus"]) {
  if (streamStatus === "live") {
    return "Live stream";
  }
  if (streamStatus === "connecting") {
    return "Connecting";
  }
  if (streamStatus === "offline") {
    return "Stream paused";
  }
  return "Select a run";
}

function GenerationEmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state empty-state-rich">
      <p className="card-eyebrow">Generation</p>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-copy">{body}</p>
    </div>
  );
}

function summarizeFailures(results: BatchRunResult[]) {
  const failures = results.filter((result) => result.status === "failed");
  const grouped = new Map<string, number>();

  failures.forEach((result) => {
    const key = result.error?.trim() || "Unknown failure";
    grouped.set(key, (grouped.get(key) ?? 0) + 1);
  });

  return Array.from(grouped.entries())
    .sort((left, right) => right[1] - left[1])
    .slice(0, 4);
}

export function GenerationView({
  form,
  onFormChange,
  onSubmit,
  onReset,
  submitting,
  message,
  runs,
  runsError,
  selectedRunId,
  selectedRunDetail,
  detailLoading,
  streamEvents,
  streamStatus,
  detailError,
  actionPending,
  modelOptions,
  modelLoading,
  onOpenModelPicker,
  onOpenDataset,
  onSelectRun,
  onRefreshRuns,
  onBatchAction,
}: GenerationViewProps) {
  const progress = selectedRunDetail
    ? percentage(selectedRunDetail.saved, selectedRunDetail.failed, selectedRunDetail.requested_runs)
    : 0;
  const failureSummary = selectedRunDetail ? summarizeFailures(selectedRunDetail.results) : [];

  return (
    <section className="generation-layout" aria-label="Generation workspace">
      <form
        className="generation-form-panel"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
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
              onChange={(event) => onFormChange((current) => ({ ...current, topics: event.target.value }))}
              rows={4}
              value={form.topics}
            />
          </label>

          <label className="field">
            <span className="field-label">Agent Types</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, agentTypes: event.target.value }))}
              type="text"
              value={form.agentTypes}
            />
          </label>

          <label className="field">
            <span className="field-label">Amount</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, amount: event.target.value }))}
              type="number"
              value={form.amount}
            />
          </label>

          <label className="field">
            <span className="field-label">Examples per Dataset</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, exAmt: event.target.value }))}
              type="number"
              value={form.exAmt}
            />
          </label>

          <label className="field">
            <span className="field-label">Model</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, model: event.target.value }))}
              list="generation-model-options"
              type="text"
              value={form.model}
            />
            <span className="field-hint">
              {modelLoading
                ? "Loading cached models..."
                : modelOptions.length > 0
                  ? `${modelOptions.length} cached models available`
                  : "No cached models available"}
            </span>
            <button className="ghost-button ghost-button-compact field-picker-button" onClick={onOpenModelPicker} type="button">
              Browse models
            </button>
          </label>

          <datalist id="generation-model-options">
            {modelOptions.map((model) => (
              <option key={model.id} label={model.name} value={model.id} />
            ))}
          </datalist>

          <label className="field">
            <span className="field-label">Max Concurrency</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, maxConcurrency: event.target.value }))}
              type="number"
              value={form.maxConcurrency}
            />
          </label>
        </div>

        <div className="generation-actions">
          <button className="topbar-button" disabled={submitting} type="submit">
            <span className="topbar-button-mark" aria-hidden="true">
              +
            </span>
            <span>{submitting ? "Launching..." : "Launch Batch"}</span>
          </button>
          <button className="ghost-button" onClick={onReset} type="button">
            Reset
          </button>
        </div>

        {message ? <p className="inline-message">{message}</p> : null}
      </form>

      <div className="generation-monitor-stack">
        <aside className="generation-side-panel">
          <div className="panel-heading generation-panel-header">
            <div>
              <p className="card-eyebrow">Recent runs</p>
              <h2 className="placeholder-title">Run monitor</h2>
            </div>
            <button className="ghost-button ghost-button-compact" onClick={onRefreshRuns} type="button">
              Refresh
            </button>
          </div>

          <div className="run-list">
            {runs.map((run) => {
              const completion = percentage(run.saved, run.failed, run.requested_runs);
              return (
                <button
                  key={run.run_id}
                  className={run.run_id === selectedRunId ? "run-card run-card-active" : "run-card"}
                  onClick={() => onSelectRun(run.run_id)}
                  type="button"
                >
                  <div className="run-card-header">
                    <div>
                      <p className="run-id">{run.run_id}</p>
                      <h3 className="run-topic">{run.topic || "Untitled batch run"}</h3>
                    </div>
                    <span className={`status-tag status-tag-${run.status}`}>{run.status}</span>
                  </div>

                  <p className="run-meta">
                    {run.requested_agent || "mixed agents"} - {compactDate(run.created_at)}
                  </p>
                  <p className="card-body">
                    {run.saved} saved - {run.failed} failed - {run.running} running - {run.queued} queued
                  </p>

                  <div className="run-progress-track" aria-hidden="true">
                    <span className="run-progress-value" style={{ width: `${completion}%` }} />
                  </div>
                </button>
              );
            })}
            {runs.length === 0 ? (
              <GenerationEmptyState
                body={runsError || "Launch a batch or refresh the monitor to start tracking recent runs here."}
                title="No recent runs yet"
              />
            ) : null}
          </div>
        </aside>

        <section className="generation-detail-panel" aria-label="Selected run detail">
          <div className="panel-heading generation-panel-header">
            <div>
              <p className="card-eyebrow">Selected run</p>
              <h2 className="placeholder-title">Live batch detail</h2>
            </div>
            <div className="generation-live">
              <span className={`status-dot generation-dot generation-dot-${streamStatus}`} aria-hidden="true" />
              <span>{streamLabel(streamStatus)}</span>
            </div>
          </div>

          {detailLoading ? (
            <div className="generation-loading-card">Loading batch detail...</div>
          ) : selectedRunDetail ? (
            <>
              <div className="generation-detail-head">
                <div>
                  <p className="run-id">{selectedRunDetail.batch_run_id}</p>
                  <h3 className="generation-detail-title">
                    {selectedRunDetail.topic || "Untitled generation batch"}
                  </h3>
                  <p className="card-body">
                    {selectedRunDetail.requested_agent || (selectedRunDetail.random_agent ? "random agents" : "mixed agents")}
                    {" - "}
                    {compactDate(selectedRunDetail.created_at)}
                  </p>
                </div>
                <span className={`status-tag status-tag-${selectedRunDetail.status}`}>
                  {selectedRunDetail.status}
                </span>
              </div>

              <div className="generation-stat-grid">
                <article className="generation-stat-card">
                  <span className="generation-stat-label">Requested</span>
                  <strong>{selectedRunDetail.requested_runs}</strong>
                </article>
                <article className="generation-stat-card">
                  <span className="generation-stat-label">Saved</span>
                  <strong>{selectedRunDetail.saved}</strong>
                </article>
                <article className="generation-stat-card">
                  <span className="generation-stat-label">Failed</span>
                  <strong>{selectedRunDetail.failed}</strong>
                </article>
                <article className="generation-stat-card">
                  <span className="generation-stat-label">Running / Queued</span>
                  <strong>
                    {selectedRunDetail.running} / {selectedRunDetail.queued}
                  </strong>
                </article>
              </div>

              <div className="generation-progress-block">
                <div className="generation-progress-copy">
                  <span>Batch progress</span>
                  <span>{progress.toFixed(0)}%</span>
                </div>
                <div className="run-progress-track">
                  <span className="run-progress-value" style={{ width: `${progress}%` }} />
                </div>
              </div>

              {failureSummary.length > 0 ? (
                <div className="generation-failure-block">
                  <div className="generation-section-head">
                    <p className="card-eyebrow">Failure summary</p>
                    <p className="generation-section-note">{selectedRunDetail.failed} failed items</p>
                  </div>
                  <div className="generation-failure-list">
                    {failureSummary.map(([reason, count]) => (
                      <article key={reason} className="generation-failure-card">
                        <strong>{count}x</strong>
                        <p>{reason}</p>
                      </article>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="generation-control-row">
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={actionPending === "resume" || selectedRunDetail.status === "completed" || selectedRunDetail.status === "cancelled"}
                  onClick={() => onBatchAction("resume")}
                  type="button"
                >
                  {actionPending === "resume" ? "Resuming..." : "Resume"}
                </button>
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={actionPending === "pause" || selectedRunDetail.status === "paused" || selectedRunDetail.status === "completed" || selectedRunDetail.status === "cancelled"}
                  onClick={() => onBatchAction("pause")}
                  type="button"
                >
                  {actionPending === "pause" ? "Pausing..." : "Pause"}
                </button>
                <button
                  className="danger-button danger-button-slim"
                  disabled={actionPending === "stop" || selectedRunDetail.status === "cancelled" || selectedRunDetail.status === "completed"}
                  onClick={() => onBatchAction("stop")}
                  type="button"
                >
                  {actionPending === "stop" ? "Stopping..." : "Stop"}
                </button>
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={actionPending === "restart-failed" || selectedRunDetail.failed === 0}
                  onClick={() => onBatchAction("restart-failed")}
                  type="button"
                >
                  {actionPending === "restart-failed" ? "Requeueing..." : "Restart Failed"}
                </button>
              </div>

              <dl className="generation-meta-grid">
                <div>
                  <dt>Started</dt>
                  <dd>{compactDate(selectedRunDetail.started_at)}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{compactDate(selectedRunDetail.updated_at)}</dd>
                </div>
                <div>
                  <dt>Completed</dt>
                  <dd>{compactDate(selectedRunDetail.completed_at)}</dd>
                </div>
                <div>
                  <dt>Run ID</dt>
                  <dd>{selectedRunDetail.batch_run_id}</dd>
                </div>
              </dl>

              <div className="generation-event-section">
                <div className="generation-section-head">
                  <p className="card-eyebrow">Live completions</p>
                  <p className="generation-section-note">
                    {streamEvents.length > 0
                      ? `${streamEvents.length} recent events`
                      : `${selectedRunDetail.results.length} total item results`}
                  </p>
                </div>

                <div className="generation-event-list">
                  {streamEvents.length > 0
                    ? streamEvents.map((event) => (
                        <article key={event.id} className="generation-event-card">
                          <div className="generation-event-head">
                            <strong>{event.topic || "Completed item"}</strong>
                            <span className={`status-tag status-tag-${event.status === "saved" ? "completed" : "failed"}`}>
                              {event.status}
                            </span>
                          </div>
                          <p className="generation-event-meta">
                            {event.agent || "mixed agent"}
                            {event.category ? ` - ${event.category.split("_").join(" ")}` : ""}
                            {typeof event.score === "number" ? ` - score ${event.score.toFixed(1)}` : ""}
                            {typeof event.cost === "number" ? ` - $${event.cost.toFixed(2)}` : ""}
                          </p>
                          {event.datasetId ? (
                            <div className="generation-event-actions">
                              <button
                                className="ghost-button ghost-button-compact"
                                onClick={() => onOpenDataset(event.datasetId as number)}
                                type="button"
                              >
                                Open dataset
                              </button>
                            </div>
                          ) : null}
                          {event.error ? <p className="generation-event-error">{event.error}</p> : null}
                        </article>
                      ))
                    : selectedRunDetail.results
                        .filter((result) => result.status === "saved" || result.status === "failed")
                        .slice(-8)
                        .reverse()
                        .map((result) => (
                          <article key={`${result.run_id}-${result.status}`} className="generation-event-card">
                            <div className="generation-event-head">
                              <strong>{result.topic || "Completed item"}</strong>
                              <span className={`status-tag status-tag-${result.status === "saved" ? "completed" : "failed"}`}>
                                {result.status}
                              </span>
                            </div>
                            <p className="generation-event-meta">
                              #{result.index + 1}
                              {result.agent ? ` - ${result.agent}` : ""}
                              {result.dataset_id ? ` - dataset ${result.dataset_id}` : ""}
                            </p>
                            {result.dataset_id ? (
                              <div className="generation-event-actions">
                                <button
                                  className="ghost-button ghost-button-compact"
                                  onClick={() => onOpenDataset(result.dataset_id as number)}
                                  type="button"
                                >
                                  Open dataset
                                </button>
                              </div>
                            ) : null}
                            {result.error ? <p className="generation-event-error">{result.error}</p> : null}
                          </article>
                        ))}
                  {streamEvents.length === 0 && selectedRunDetail.results.length === 0 ? (
                    <div className="empty-state">No item results have landed for this batch yet.</div>
                  ) : null}
                </div>
              </div>

              {detailError ? <p className="inline-message">{detailError}</p> : null}
            </>
          ) : (
            <GenerationEmptyState
              body={
                runs.length > 0
                  ? "Pick a run from the monitor to inspect counts, stream completions, and control the batch."
                  : runsError || "Launch a batch to start seeing live run activity here."
              }
              title={runs.length > 0 ? "Select a run to inspect" : "Generation activity will show up here"}
            />
          )}
        </section>
      </div>
    </section>
  );
}

