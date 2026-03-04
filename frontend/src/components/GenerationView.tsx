import { useMemo, useState } from "react";
import { ToggleField } from "./ToggleField";

type BatchAction = "pause" | "resume" | "stop" | "restart-failed" | "delete";

const AGENT_OPTIONS = [
  { id: "conversation", label: "Conversation" },
  { id: "instruction_following", label: "Instruction" },
  { id: "style", label: "Style" },
  { id: "qa", label: "Q&A" },
  { id: "adversarial", label: "Adversarial" },
  { id: "domain_specialist", label: "Domain" },
];

const CONVERSATION_LENGTH_OPTIONS = [
  {
    id: "varied",
    label: "Varied",
    description: "Mix short, medium, and longer chats so personality holds up across different conversation lengths.",
  },
  {
    id: "short",
    label: "Short",
    description: "Bias toward quicker 2-4 turn exchanges when you want broader scenario coverage.",
  },
  {
    id: "balanced",
    label: "Balanced",
    description: "Lean toward mid-length back and forth with room for a couple follow-up turns.",
  },
  {
    id: "long",
    label: "Long",
    description: "Bias toward longer threads so the assistant practices staying in character over time.",
  },
] as const;

type BatchRunRow = {
  run_id: string;
  request_group_id?: string | null;
  member_run_ids?: string[];
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "paused" | "stopping";
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

type BatchRunSlotSummary = {
  slot_key: string;
  requested_topic: string;
  selected_agent: string;
  requested_runs: number;
  saved: number;
  failed: number;
};

type BatchRunResult = {
  index: number;
  run_id: string;
  dataset_id: number | null;
  status: "saved" | "failed" | "queued" | "running";
  topic: string | null;
  agent: string | null;
  error: string | null;
};

type BatchRunDetail = {
  batch_run_id: string;
  request_group_id?: string | null;
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
  per_slot_summary?: BatchRunSlotSummary[];
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
  allowTopicVariations: boolean;
  conversationLengthMode: "varied" | "short" | "balanced" | "long";
  amount: string;
  exAmt: string;
  sourceDatasetIds: string;
  personalityInstructions: string;
  sourceMaterialMode: "style_only" | "content_and_style";
  model: string;
  maxConcurrency: string;
};

type ModelOption = {
  id: string;
  name: string;
};

type SourceDatasetOption = {
  id: number;
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
  streamStatus: "idle" | "connecting" | "live" | "offline" | "completed";
  detailError: string;
  actionPending: string;
  modelOptions: ModelOption[];
  availableDatasets: SourceDatasetOption[];
  selectedSourceDatasetIds: number[];
  modelLoading: boolean;
  onOpenModelPicker: () => void;
  onToggleSourceDataset: (datasetId: number) => void;
  onClearSourceDatasets: () => void;
  onOpenDataset: (datasetId: number) => void;
  onSelectRun: (runId: string) => void;
  onRefreshRuns: () => void;
  onClearCompletedRuns: () => void;
  onBatchAction: (action: BatchAction) => void;
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
  if (streamStatus === "completed") {
    return "Complete";
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
  availableDatasets,
  selectedSourceDatasetIds,
  modelLoading,
  onOpenModelPicker,
  onToggleSourceDataset,
  onClearSourceDatasets,
  onOpenDataset,
  onSelectRun,
  onRefreshRuns,
  onClearCompletedRuns,
  onBatchAction,
}: GenerationViewProps) {
  const [isSupportDatasetModalOpen, setIsSupportDatasetModalOpen] = useState(false);
  const [supportDatasetQuery, setSupportDatasetQuery] = useState("");
  const progress = selectedRunDetail
    ? percentage(selectedRunDetail.saved, selectedRunDetail.failed, selectedRunDetail.requested_runs)
    : 0;
  const failureSummary = selectedRunDetail ? summarizeFailures(selectedRunDetail.results) : [];
  const selectedAgentIds = useMemo(
    () =>
      form.agentTypes
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    [form.agentTypes],
  );
  const filteredSupportDatasets = useMemo(() => {
    const query = supportDatasetQuery.trim().toLowerCase();
    if (!query) {
      return availableDatasets;
    }
    return availableDatasets.filter(
      (dataset) => dataset.name.toLowerCase().includes(query) || String(dataset.id).includes(query),
    );
  }, [availableDatasets, supportDatasetQuery]);

  function toggleAgentType(agentId: string) {
    onFormChange((current) => {
      const parsed = current.agentTypes
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      const next = parsed.includes(agentId) ? parsed.filter((value) => value !== agentId) : [...parsed, agentId];
      return { ...current, agentTypes: next.join(", ") };
    });
  }

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
            <span className="field-hint">
              Each non-empty line is a separate topic. When you enter multiple topics, they are used as-is and the total amount is split across those topics and the selected agent types.
            </span>
          </label>

          <div className="field">
            <span className="field-label">Agent Types</span>
            <div className="agent-chip-grid">
              {AGENT_OPTIONS.map((agent) => {
                const isSelected = selectedAgentIds.includes(agent.id);
                return (
                  <button
                    key={agent.id}
                    className={isSelected ? "agent-chip agent-chip-active" : "agent-chip"}
                    onClick={() => toggleAgentType(agent.id)}
                    type="button"
                  >
                    {agent.label}
                  </button>
                );
              })}
            </div>
            <span className="field-hint">
              Pick one or more agent styles. The total amount is split across the selected agents.
            </span>
          </div>

          <div className="field field-toggle-card">
            <div className="field-toggle-head">
              <div>
                <span className="field-label">Generate Topic Variants</span>
                <p className="generation-section-note">
                  Off by default. Turn this on only if you want the system to expand one base topic into related subtopics.
                </p>
              </div>
              <ToggleField
                checked={form.allowTopicVariations}
                compact
                label="Allow variants"
                onChange={(checked) =>
                  onFormChange((current) => ({
                    ...current,
                    allowTopicVariations: checked,
                  }))
                }
              />
            </div>
            <span className="field-hint">
              When this is off, each topic line is used literally and the planner will not invent alternate topic angles.
            </span>
          </div>

          <div className="field field-toggle-card">
            <div className="field-toggle-head">
              <div>
                <span className="field-label">Conversation Length</span>
                <p className="generation-section-note">
                  Applies to the conversation agent. This controls whether generated chats stay short or include more follow-up turns.
                </p>
              </div>
            </div>
            <div className="agent-chip-grid">
              {CONVERSATION_LENGTH_OPTIONS.map((option) => {
                const isSelected = form.conversationLengthMode === option.id;
                return (
                  <button
                    key={option.id}
                    className={isSelected ? "agent-chip agent-chip-active" : "agent-chip"}
                    onClick={() =>
                      onFormChange((current) => ({
                        ...current,
                        conversationLengthMode: option.id,
                      }))
                    }
                    type="button"
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
            <span className="field-hint">
              {
                CONVERSATION_LENGTH_OPTIONS.find((option) => option.id === form.conversationLengthMode)?.description
              }
            </span>
          </div>

          <label className="field">
            <span className="field-label">Amount</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, amount: event.target.value }))}
              type="number"
              value={form.amount}
            />
          </label>

          <label className="field field-tight">
            <span className="field-label field-label-nowrap">Examples per Dataset</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, exAmt: event.target.value }))}
              type="number"
              value={form.exAmt}
            />
          </label>

          <label className="field">
            <span className="field-label">Max Concurrency</span>
            <input
              onChange={(event) => onFormChange((current) => ({ ...current, maxConcurrency: event.target.value }))}
              type="number"
              value={form.maxConcurrency}
            />
          </label>

          <div className="field field-span-2">
            <span className="field-label">Support Datasets</span>
            <div className="support-launcher-card">
              <div>
                <p className="generation-section-note">
                  {selectedSourceDatasetIds.length > 0
                    ? `${selectedSourceDatasetIds.length} support datasets attached`
                    : "Attach example datasets as grounding context without crowding the form"}
                </p>
                {selectedSourceDatasetIds.length > 0 ? (
                  <div className="support-selection-strip">
                    {selectedSourceDatasetIds.slice(0, 6).map((datasetId) => (
                      <span key={datasetId} className="support-selection-pill">
                        #{datasetId}
                      </span>
                    ))}
                    {selectedSourceDatasetIds.length > 6 ? (
                      <span className="support-selection-pill">+{selectedSourceDatasetIds.length - 6} more</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
              <div className="dataset-picker-actions">
                <button
                  className="ghost-button ghost-button-compact"
                  onClick={() => setIsSupportDatasetModalOpen(true)}
                  type="button"
                >
                  Choose Datasets
                </button>
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={selectedSourceDatasetIds.length === 0}
                  onClick={onClearSourceDatasets}
                  type="button"
                >
                  Clear All
                </button>
              </div>
            </div>
            <span className="field-hint">
              Attached dataset examples are expanded into source material before generation starts.
            </span>
          </div>

          <div className="field field-span-2">
            <span className="field-label">Source Material Mode</span>
            <div className="agent-chip-grid">
              <button
                className={
                  form.sourceMaterialMode === "style_only" ? "agent-chip agent-chip-active" : "agent-chip"
                }
                onClick={() =>
                  onFormChange((current) => ({
                    ...current,
                    sourceMaterialMode: "style_only",
                  }))
                }
                type="button"
              >
                Style Only
              </button>
              <button
                className={
                  form.sourceMaterialMode === "content_and_style" ? "agent-chip agent-chip-active" : "agent-chip"
                }
                onClick={() =>
                  onFormChange((current) => ({
                    ...current,
                    sourceMaterialMode: "content_and_style",
                  }))
                }
                type="button"
              >
                Content + Style
              </button>
            </div>
            <span className="field-hint">
              Style Only keeps the topic in charge and uses attached material just for tone, voice, and personality. Content + Style also lets source material influence details and subject matter when it fits.
            </span>
          </div>

          <label className="field field-span-2">
            <span className="field-label">Personality Instructions</span>
            <textarea
              onChange={(event) =>
                onFormChange((current) => ({ ...current, personalityInstructions: event.target.value }))
              }
              placeholder="Optional voice, tone, behavior, and character rules to ground the batch"
              rows={4}
              value={form.personalityInstructions}
            />
            <span className="field-hint">
              Use this for explicit style rules like tone, pacing, favorite phrases, boundaries, and character traits.
            </span>
          </label>

          <label className="field field-span-2">
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
            <div className="dataset-picker-actions">
              <button className="ghost-button ghost-button-compact" onClick={onRefreshRuns} type="button">
                Refresh
              </button>
              <button className="ghost-button ghost-button-compact" onClick={onClearCompletedRuns} type="button">
                Clear Completed
              </button>
            </div>
          </div>

          <div className="run-list run-list-compact">
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
                      <p className="run-id">{run.run_id.slice(0, 8)}</p>
                      <h3 className="run-topic run-topic-compact">{run.topic || "Untitled batch run"}</h3>
                    </div>
                    <span className={`status-tag status-tag-${run.status}`}>{run.status}</span>
                  </div>

                  <div className="run-card-summary">
                    <p className="run-meta">{run.requested_agent || "mixed agents"}</p>
                    <p className="run-meta">{compactDate(run.created_at)}</p>
                  </div>
                  <div className="run-count-strip">
                    <span className="run-count-pill">{run.requested_runs} requested</span>
                    <span className="run-count-pill">{run.saved} saved</span>
                    <span className="run-count-pill">{run.failed} failed</span>
                    {run.running > 0 ? <span className="run-count-pill">{run.running} running</span> : null}
                    {run.queued > 0 ? <span className="run-count-pill">{run.queued} queued</span> : null}
                  </div>

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
                  disabled={
                    actionPending === "resume" ||
                    selectedRunDetail.status !== "paused"
                  }
                  onClick={() => onBatchAction("resume")}
                  type="button"
                >
                  {actionPending === "resume" ? "Resuming..." : "Resume"}
                </button>
                <button
                  className="ghost-button ghost-button-compact"
                  disabled={
                    actionPending === "pause" ||
                    selectedRunDetail.status === "paused" ||
                    selectedRunDetail.status === "completed" ||
                    selectedRunDetail.status === "cancelled" ||
                    selectedRunDetail.status === "stopping"
                  }
                  onClick={() => onBatchAction("pause")}
                  type="button"
                >
                  {actionPending === "pause" ? "Pausing..." : "Pause"}
                </button>
                <button
                  className="danger-button danger-button-slim"
                  disabled={
                    actionPending === "stop" ||
                    selectedRunDetail.status === "cancelled" ||
                    selectedRunDetail.status === "completed" ||
                    selectedRunDetail.status === "stopping"
                  }
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
                <button
                  className="danger-button danger-button-slim"
                  disabled={
                    actionPending === "delete" ||
                    !["completed", "failed", "cancelled", "stopping"].includes(selectedRunDetail.status)
                  }
                  onClick={() => onBatchAction("delete")}
                  type="button"
                >
                  {actionPending === "delete"
                    ? "Removing..."
                    : selectedRunDetail.status === "stopping"
                      ? "Force Remove"
                      : "Remove Run"}
                </button>
              </div>

              {selectedRunDetail.per_slot_summary && selectedRunDetail.per_slot_summary.length > 0 ? (
                <div className="generation-slot-block">
                  <div className="generation-section-head">
                    <p className="card-eyebrow">Grouped slots</p>
                    <p className="generation-section-note">How this batch was split</p>
                  </div>
                  <div className="generation-slot-grid">
                    {selectedRunDetail.per_slot_summary.slice(0, 8).map((slot) => (
                      <article key={slot.slot_key} className="generation-slot-card">
                        <strong>{slot.requested_topic}</strong>
                        <p className="generation-event-meta">
                          {slot.selected_agent} - {slot.saved}/{slot.requested_runs} saved
                        </p>
                      </article>
                    ))}
                  </div>
                </div>
              ) : null}

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

      {isSupportDatasetModalOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="detail-modal model-picker-modal" aria-modal="true" role="dialog">
            <div className="detail-modal-header">
              <div>
                <p className="card-eyebrow">Support datasets</p>
                <h3 className="placeholder-title">Choose grounding datasets</h3>
              </div>
              <button className="ghost-button ghost-button-compact" onClick={() => setIsSupportDatasetModalOpen(false)} type="button">
                Close
              </button>
            </div>

            <label className="field export-picker-search">
              <span className="field-label">Search datasets</span>
              <input
                onChange={(event) => setSupportDatasetQuery(event.target.value)}
                placeholder="Search by name or dataset id"
                type="text"
                value={supportDatasetQuery}
              />
            </label>

            <div className="model-picker-grid">
              {filteredSupportDatasets.map((dataset) => {
                const isSelected = selectedSourceDatasetIds.includes(dataset.id);
                return (
                  <button
                    key={dataset.id}
                    className={isSelected ? "dataset-picker-chip dataset-picker-chip-active" : "dataset-picker-chip"}
                    onClick={() => onToggleSourceDataset(dataset.id)}
                    type="button"
                  >
                    <span className="dataset-picker-id">#{dataset.id}</span>
                    <span className="dataset-picker-name">{dataset.name}</span>
                  </button>
                );
              })}
              {filteredSupportDatasets.length === 0 ? (
                <div className="empty-state">No datasets matched this search.</div>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

