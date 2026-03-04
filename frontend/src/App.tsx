import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "./components/AppShell";
import { DashboardView } from "./components/DashboardView";
import { DatasetsView } from "./components/DatasetsView";
import { GenerationView } from "./components/GenerationView";
import { ModelPickerModal } from "./components/ModelPickerModal";
import { DocumentsView, ExportsView, SettingsView } from "./components/ResourceViews";
import { SearchResults } from "./components/SearchResults";
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

type MergeResponse = {
  success?: boolean;
  message?: string;
  data?: {
    created_dataset_ids?: number[];
    pools_merged?: number;
  };
};

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

type BatchRunResult = {
  index: number;
  run_id: string;
  dataset_id: number | null;
  status: "saved" | "failed" | "queued" | "running";
  topic: string | null;
  agent: string | null;
  error: string | null;
};

type BatchRunSlotSummary = {
  slot_key: string;
  requested_topic: string;
  selected_agent: string;
  requested_runs: number;
  saved: number;
  failed: number;
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

type ExportFormState = {
  datasetIds: string;
  format: "sharegpt" | "chatml" | "alpaca";
  minScore: string;
  maxExamples: string;
  trainValSplit: string;
  dedupePass: boolean;
  shuffle: boolean;
};

type DashboardStats = {
  datasets: number;
  training_examples: number;
  embedding_time: number;
  ingest_time: number;
  grading_time: number;
  api_cost: number;
};

type CreditsSummary = {
  balance: number | null;
  totalCredits: number | null;
  totalUsage: number | null;
};

type ModelOption = {
  id: string;
  name: string;
};

type ModelPickerTarget = "generation" | "default_model" | "grading_model" | "naming_model" | null;

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

type BatchGenerateResponse = {
  success: boolean;
  message: string;
  data?: {
    batch_run_ids?: string[];
    auto_merge?: {
      requested?: boolean;
      status?: "merged" | "no_candidates" | "failed" | "skipped" | "disabled";
      message?: string;
      result?: {
        created_dataset_ids?: number[];
        pools_merged?: number;
      };
    };
  };
};

type HealthState = "checking" | "healthy" | "offline";

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

type ToastState = {
  tone: "success" | "error";
  message: string;
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

type BatchAction = "pause" | "resume" | "stop" | "restart-failed" | "delete";

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

type PersistedUiState = {
  activeView: string;
  selectedDatasetId: number;
  selectedBatchRunId: string;
  selectedDocumentId: number;
  activeCategory: string;
  documentFilter: string;
  generationForm: GenerationFormState;
  exportForm: ExportFormState;
};

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
    description: "Browse the current dataset inventory with the metadata available from the API.",
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
    description: "The backend settings endpoints exist now; this UI just needs its editing surface wired in.",
  },
];

const emptyDashboardCards = [
  { eyebrow: "Balance", title: "--", body: "Loading current OpenRouter balance." },
  { eyebrow: "Datasets", title: "--", body: "Loading total dataset count." },
  { eyebrow: "Examples", title: "--", body: "Loading stored training examples." },
  { eyebrow: "Embedding", title: "--", body: "Loading average embedding time." },
  { eyebrow: "Ingest", title: "--", body: "Loading average ingest time." },
  { eyebrow: "Grading", title: "--", body: "Loading average grading time." },
  { eyebrow: "API Cost", title: "--", body: "Loading total API cost." },
];

const LEGACY_GENERATION_MODEL_DEFAULT = "google/gemini-2.5-flash";

function buildDefaultGenerationForm(defaultModel = ""): GenerationFormState {
  return {
    topics: "Code review and debugging\nSecurity vulnerability identification",
    agentTypes: "qa, instruction_following, adversarial",
    allowTopicVariations: false,
    conversationLengthMode: "varied",
    amount: "120",
    exAmt: "25",
    sourceDatasetIds: "",
    personalityInstructions: "",
    sourceMaterialMode: "content_and_style",
    model: defaultModel,
    maxConcurrency: "25",
  };
}

const defaultGenerationForm = buildDefaultGenerationForm();

const defaultExportForm: ExportFormState = {
  datasetIds: "",
  format: "sharegpt",
  minScore: "",
  maxExamples: "",
  trainValSplit: "",
  dedupePass: false,
  shuffle: false,
};

const UI_STATE_STORAGE_KEY = "pdata.frontend.ui-state.v1";

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

function parseNumericApiValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const normalized = value.replace(/[^0-9.-]/g, "").trim();
    const parsed = Number.parseFloat(normalized);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function formatNullableNumber(value: number | null | undefined, suffix = "", digits = 2): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "--";
}

function dismissSearch(setSearchQuery: (value: string) => void) {
  setSearchQuery("");
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
}

function uniqueSearchItems(items: SearchSection["items"]) {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = `${item.kind ?? item.view}:${item.id}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function summarizeTopics(topics: string[]) {
  if (topics.length === 0) {
    return null;
  }
  if (topics.length === 1) {
    return topics[0];
  }
  return `${topics.length} topics`;
}

function summarizeAgents(agents: string[]) {
  if (agents.length === 0) {
    return null;
  }
  if (agents.length === 1) {
    return agents[0];
  }
  return `${agents.length} agent types`;
}

function aggregateBatchRunStatus(rows: BatchRunRow[]): BatchRunRow["status"] {
  const running = rows.reduce((total, row) => total + row.running, 0);
  const queued = rows.reduce((total, row) => total + row.queued, 0);
  const saved = rows.reduce((total, row) => total + row.saved, 0);
  const failed = rows.reduce((total, row) => total + row.failed, 0);
  const requested = rows.reduce((total, row) => total + row.requested_runs, 0);
  const hasStopping = rows.some((row) => row.status === "stopping");
  const hasCancelled = rows.some((row) => row.status === "cancelled");
  const hasPaused = rows.some((row) => row.status === "paused");

  if (hasStopping || (hasCancelled && running > 0) || (hasCancelled && queued > 0)) {
    return "stopping";
  }
  if (running > 0) {
    return "running";
  }
  if (hasPaused && queued > 0) {
    return "paused";
  }
  if (queued > 0) {
    return saved > 0 || failed > 0 ? "running" : "queued";
  }
  if (hasCancelled) {
    return "cancelled";
  }
  if (failed === requested && requested > 0) {
    return "failed";
  }
  return "completed";
}

function aggregateBatchRunGroup(rows: BatchRunRow[]): BatchRunRow {
  const sorted = [...rows].sort((left, right) => {
    const leftTime = left.created_at ? new Date(left.created_at).getTime() : 0;
    const rightTime = right.created_at ? new Date(right.created_at).getTime() : 0;
    return rightTime - leftTime;
  });
  const uniqueTopics = Array.from(new Set(rows.map((row) => row.topic).filter((value): value is string => Boolean(value))));
  const uniqueAgents = Array.from(
    new Set(rows.map((row) => row.requested_agent).filter((value): value is string => Boolean(value))),
  );
  const representative = sorted[0] ?? rows[0];

  return {
    run_id: representative.request_group_id || representative.run_id,
    request_group_id: representative.request_group_id || representative.run_id,
    member_run_ids: rows.map((row) => row.run_id),
    status: aggregateBatchRunStatus(rows),
    requested_runs: rows.reduce((total, row) => total + row.requested_runs, 0),
    saved: rows.reduce((total, row) => total + row.saved, 0),
    failed: rows.reduce((total, row) => total + row.failed, 0),
    queued: rows.reduce((total, row) => total + row.queued, 0),
    running: rows.reduce((total, row) => total + row.running, 0),
    topic: summarizeTopics(uniqueTopics),
    requested_agent: summarizeAgents(uniqueAgents),
    created_at: rows
      .map((row) => row.created_at)
      .filter((value): value is string => Boolean(value))
      .sort()[0] ?? representative.created_at,
    updated_at: rows
      .map((row) => row.updated_at)
      .filter((value): value is string => Boolean(value))
      .sort()
      .at(-1) ?? representative.updated_at,
    completed_at: rows
      .map((row) => row.completed_at)
      .filter((value): value is string => Boolean(value))
      .sort()
      .at(-1) ?? representative.completed_at,
  };
}

function aggregateBatchRunDetail(details: BatchRunDetail[], selectedId: string): BatchRunDetail | null {
  if (details.length === 0) {
    return null;
  }

  const uniqueTopics = Array.from(
    new Set(details.map((detail) => detail.topic).filter((value): value is string => Boolean(value))),
  );
  const uniqueAgents = Array.from(
    new Set(details.map((detail) => detail.requested_agent).filter((value): value is string => Boolean(value))),
  );
  const allResults = details.flatMap((detail) => detail.results);
  const slotMap = new Map<string, BatchRunSlotSummary>();

  details.flatMap((detail) => detail.per_slot_summary ?? []).forEach((slot) => {
    const current = slotMap.get(slot.slot_key);
    if (current) {
      current.requested_runs += slot.requested_runs;
      current.saved += slot.saved;
      current.failed += slot.failed;
      return;
    }
    slotMap.set(slot.slot_key, { ...slot });
  });

  const startedAtValues = details.map((detail) => detail.started_at).filter((value): value is string => Boolean(value)).sort();
  const createdAtValues = details.map((detail) => detail.created_at).filter((value): value is string => Boolean(value)).sort();
  const updatedAtValues = details.map((detail) => detail.updated_at).filter((value): value is string => Boolean(value)).sort();
  const completedAtValues = details.map((detail) => detail.completed_at).filter((value): value is string => Boolean(value)).sort();
  const requestGroupId = details.find((detail) => detail.request_group_id)?.request_group_id ?? selectedId;

  return {
    batch_run_id: requestGroupId || selectedId,
    request_group_id: requestGroupId,
    status: aggregateBatchRunStatus(
      details.map((detail) => ({
        run_id: detail.batch_run_id,
        request_group_id: detail.request_group_id,
        status: detail.status,
        requested_runs: detail.requested_runs,
        saved: detail.saved,
        failed: detail.failed,
        queued: detail.queued,
        running: detail.running,
        topic: detail.topic,
        requested_agent: detail.requested_agent,
        created_at: detail.created_at,
        updated_at: detail.updated_at,
        completed_at: detail.completed_at,
      })),
    ),
    requested_runs: details.reduce((total, detail) => total + detail.requested_runs, 0),
    saved: details.reduce((total, detail) => total + detail.saved, 0),
    failed: details.reduce((total, detail) => total + detail.failed, 0),
    queued: details.reduce((total, detail) => total + detail.queued, 0),
    running: details.reduce((total, detail) => total + detail.running, 0),
    topic: summarizeTopics(uniqueTopics),
    requested_agent: summarizeAgents(uniqueAgents),
    random_agent: details.some((detail) => detail.random_agent),
    created_at: createdAtValues[0] ?? null,
    updated_at: updatedAtValues.at(-1) ?? null,
    started_at: startedAtValues[0] ?? null,
    completed_at: completedAtValues.at(-1) ?? null,
    per_slot_summary: Array.from(slotMap.values()).sort((left, right) => right.requested_runs - left.requested_runs),
    results: allResults,
  };
}

function App() {
  const [activeView, setActiveView] = useState<string>("dashboard");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number>(0);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats | null>(null);
  const [dashboardError, setDashboardError] = useState<string>("");
  const [dashboardLoading, setDashboardLoading] = useState<boolean>(false);
  const [creditsSummary, setCreditsSummary] = useState<CreditsSummary>({
    balance: null,
    totalCredits: null,
    totalUsage: null,
  });
  const [healthState, setHealthState] = useState<HealthState>("checking");
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [datasetsError, setDatasetsError] = useState<string>("");
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(null);
  const [batchRuns, setBatchRuns] = useState<BatchRunRow[]>([]);
  const [batchRunsError, setBatchRunsError] = useState<string>("");
  const [selectedBatchRunId, setSelectedBatchRunId] = useState<string>("");
  const [selectedBatchRunDetail, setSelectedBatchRunDetail] = useState<BatchRunDetail | null>(null);
  const [batchRunDetailLoading, setBatchRunDetailLoading] = useState<boolean>(false);
  const [batchRunDetailError, setBatchRunDetailError] = useState<string>("");
  const [batchStreamEvents, setBatchStreamEvents] = useState<BatchStreamEvent[]>([]);
  const [batchStreamStatus, setBatchStreamStatus] = useState<"idle" | "connecting" | "live" | "offline" | "completed">("idle");
  const [batchActionPending, setBatchActionPending] = useState<string>("");
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [documentsError, setDocumentsError] = useState<string>("");
  const [documentsLoading, setDocumentsLoading] = useState<boolean>(false);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number>(0);
  const [selectedDocumentDetail, setSelectedDocumentDetail] = useState<DocumentDetail | null>(null);
  const [documentDetailLoading, setDocumentDetailLoading] = useState<boolean>(false);
  const [documentActionPending, setDocumentActionPending] = useState<boolean>(false);
  const [isDocumentViewOpen, setIsDocumentViewOpen] = useState<boolean>(false);
  const [isDocumentDeleteConfirmOpen, setIsDocumentDeleteConfirmOpen] = useState<boolean>(false);
  const [documentUploadFile, setDocumentUploadFile] = useState<File | null>(null);
  const [documentUploadMode, setDocumentUploadMode] = useState<"examples" | "source_material" | "pretraining_data">("source_material");
  const [documentUploadPending, setDocumentUploadPending] = useState<boolean>(false);
  const [documentUploadMessage, setDocumentUploadMessage] = useState<string>("");
  const [documentFilter, setDocumentFilter] = useState<string>("");
  const [documentUploadAdvancedOpen, setDocumentUploadAdvancedOpen] = useState<boolean>(false);
  const [documentUploadChunkSize, setDocumentUploadChunkSize] = useState<string>("2000");
  const [documentUploadChunkOverlap, setDocumentUploadChunkOverlap] = useState<string>("200");
  const [documentChunksExpanded, setDocumentChunksExpanded] = useState<boolean>(false);
  const [scraperText, setScraperText] = useState<string>("");
  const [scraperDatasetName, setScraperDatasetName] = useState<string>("");
  const [scraperPending, setScraperPending] = useState<boolean>(false);
  const [scraperMessage, setScraperMessage] = useState<string>("");
  const [exportsHistory, setExportsHistory] = useState<ExportRow[]>([]);
  const [exportsError, setExportsError] = useState<string>("");
  const [exportsLoading, setExportsLoading] = useState<boolean>(false);
  const [exportActionPendingId, setExportActionPendingId] = useState<number | null>(null);
  const [exportCreatePending, setExportCreatePending] = useState<boolean>(false);
  const [exportMessage, setExportMessage] = useState<string>("");
  const [exportForm, setExportForm] = useState<ExportFormState>(defaultExportForm);
  const [isExportPickerOpen, setIsExportPickerOpen] = useState<boolean>(false);
  const [exportPickerQuery, setExportPickerQuery] = useState<string>("");
  const [generationForm, setGenerationForm] = useState<GenerationFormState>(() => buildDefaultGenerationForm());
  const [generationMessage, setGenerationMessage] = useState<string>("");
  const [generationSubmitting, setGenerationSubmitting] = useState<boolean>(false);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [modelOptionsLoading, setModelOptionsLoading] = useState<boolean>(false);
  const [modelPickerTarget, setModelPickerTarget] = useState<ModelPickerTarget>(null);
  const [modelPickerQuery, setModelPickerQuery] = useState<string>("");
  const [settings, setSettings] = useState<SettingsValues | null>(null);
  const [initialSettings, setInitialSettings] = useState<SettingsValues | null>(null);
  const [settingsError, setSettingsError] = useState<string>("");
  const [settingsMessage, setSettingsMessage] = useState<string>("");
  const [settingsLoading, setSettingsLoading] = useState<boolean>(false);
  const [settingsSaving, setSettingsSaving] = useState<boolean>(false);
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [datasetDeletePending, setDatasetDeletePending] = useState<boolean>(false);
  const [exampleDeletePendingId, setExampleDeletePendingId] = useState<number | null>(null);
  const [exampleSavePendingId, setExampleSavePendingId] = useState<number | null>(null);
  const [isDatasetViewOpen, setIsDatasetViewOpen] = useState<boolean>(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState<boolean>(false);
  const [isTargetedMergeOpen, setIsTargetedMergeOpen] = useState<boolean>(false);
  const [targetedMergeDatasetIds, setTargetedMergeDatasetIds] = useState<number[]>([]);
  const [targetedMergeDeleteOriginals, setTargetedMergeDeleteOriginals] = useState<boolean>(false);
  const [targetedMergeThreshold, setTargetedMergeThreshold] = useState<string>("0.65");
  const [targetedMergePending, setTargetedMergePending] = useState<boolean>(false);
  const [isGlobalMergeConfirmOpen, setIsGlobalMergeConfirmOpen] = useState<boolean>(false);
  const [globalMergeDeleteOriginals, setGlobalMergeDeleteOriginals] = useState<boolean>(true);
  const [globalMergeThreshold, setGlobalMergeThreshold] = useState<string>("0.65");
  const [globalMergePending, setGlobalMergePending] = useState<boolean>(false);
  const [toast, setToast] = useState<ToastState | null>(null);
  const [datasetsRefreshKey, setDatasetsRefreshKey] = useState<number>(0);
  const [examplePreviewOffset, setExamplePreviewOffset] = useState<number>(0);
  const [batchRunsRefreshKey, setBatchRunsRefreshKey] = useState<number>(0);
  const [documentsRefreshKey, setDocumentsRefreshKey] = useState<number>(0);
  const [uiStateHydrated, setUiStateHydrated] = useState<boolean>(false);

  const currentView = useMemo(
    () => navItems.find((item) => item.id === activeView) ?? navItems[0],
    [activeView],
  );

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(UI_STATE_STORAGE_KEY);
      if (!raw) {
        setUiStateHydrated(true);
        return;
      }

      const parsed = JSON.parse(raw) as Partial<PersistedUiState>;

      if (typeof parsed.activeView === "string" && navItems.some((item) => item.id === parsed.activeView)) {
        setActiveView(parsed.activeView);
      }
      if (typeof parsed.selectedDatasetId === "number" && parsed.selectedDatasetId >= 0) {
        setSelectedDatasetId(parsed.selectedDatasetId);
      }
      if (typeof parsed.selectedBatchRunId === "string") {
        setSelectedBatchRunId(parsed.selectedBatchRunId);
      }
      if (typeof parsed.selectedDocumentId === "number" && parsed.selectedDocumentId >= 0) {
        setSelectedDocumentId(parsed.selectedDocumentId);
      }
      if (typeof parsed.activeCategory === "string") {
        setActiveCategory(parsed.activeCategory);
      }
      if (typeof parsed.documentFilter === "string") {
        setDocumentFilter(parsed.documentFilter);
      }
      if (parsed.generationForm) {
        setGenerationForm((current) => ({ ...current, ...parsed.generationForm }));
      }
      if (parsed.exportForm) {
        setExportForm((current) => ({ ...current, ...parsed.exportForm }));
      }
    } catch {
      window.localStorage.removeItem(UI_STATE_STORAGE_KEY);
    } finally {
      setUiStateHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!uiStateHydrated) {
      return;
    }

    const nextState: PersistedUiState = {
      activeView,
      selectedDatasetId,
      selectedBatchRunId,
      selectedDocumentId,
      activeCategory,
      documentFilter,
      generationForm,
      exportForm,
    };

    window.localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(nextState));
  }, [
    activeCategory,
    activeView,
    documentFilter,
    exportForm,
    generationForm,
    selectedBatchRunId,
    selectedDatasetId,
    selectedDocumentId,
    uiStateHydrated,
  ]);

  const trimmedSearch = searchQuery.trim();

  const selectedDataset = useMemo(() => {
    const fromList = datasets.find((dataset) => dataset.id === selectedDatasetId) ?? null;
    if (datasetDetail && fromList && datasetDetail.id === fromList.id) {
      return { ...fromList, ...datasetDetail };
    }
    return fromList;
  }, [datasetDetail, datasets, selectedDatasetId]);

  const selectedDocument = useMemo(
    () => documents.find((document) => document.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  );

  const filteredDocuments = useMemo(() => {
    const query = documentFilter.trim().toLowerCase();
    if (!query) {
      return documents;
    }

    return documents.filter((document) =>
      [document.name, document.file_type, document.source_material_ref]
        .some((value) => value.toLowerCase().includes(query)),
    );
  }, [documentFilter, documents]);

  const selectedExportDatasetIds = useMemo(
    () =>
      exportForm.datasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0),
    [exportForm.datasetIds],
  );

  const selectedGenerationSourceDatasetIds = useMemo(
    () =>
      generationForm.sourceDatasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0),
    [generationForm.sourceDatasetIds],
  );

  const groupedBatchRuns = useMemo(() => {
    const groups = new Map<string, BatchRunRow[]>();
    batchRuns.forEach((run) => {
      const key = run.request_group_id || run.run_id;
      const current = groups.get(key) ?? [];
      current.push(run);
      groups.set(key, current);
    });

    return Array.from(groups.values())
      .map((rows) => aggregateBatchRunGroup(rows))
      .sort((left, right) => {
        const leftTime = left.updated_at ? new Date(left.updated_at).getTime() : 0;
        const rightTime = right.updated_at ? new Date(right.updated_at).getTime() : 0;
        return rightTime - leftTime;
      });
  }, [batchRuns]);

  const selectedBatchMemberRunIds = useMemo(() => {
    const selected = groupedBatchRuns.find((run) => run.run_id === selectedBatchRunId);
    return selected?.member_run_ids ?? [];
  }, [groupedBatchRuns, selectedBatchRunId]);
  const selectedBatchMemberRunKey = useMemo(
    () => selectedBatchMemberRunIds.join(","),
    [selectedBatchMemberRunIds],
  );

  const exportPickerDatasets = useMemo(() => {
    const query = exportPickerQuery.trim().toLowerCase();
    const sorted = [...datasets].sort((left, right) => left.name.localeCompare(right.name));
    if (!query) {
      return sorted.slice(0, 100);
    }

    return sorted
      .filter((dataset) =>
        dataset.name.toLowerCase().includes(query) || String(dataset.id).includes(query),
      )
      .slice(0, 100);
  }, [datasets, exportPickerQuery]);

  const filteredModelOptions = useMemo(() => {
    const query = modelPickerQuery.trim().toLowerCase();
    if (!query) {
      return modelOptions;
    }

    return modelOptions
      .filter((model) => model.id.toLowerCase().includes(query) || model.name.toLowerCase().includes(query));
  }, [modelOptions, modelPickerQuery]);

  const datasetCategories = useMemo(
    () =>
      Array.from(
        new Set(datasets.map((dataset) => dataset.category).filter((category): category is string => Boolean(category))),
      ).sort(),
    [datasets],
  );

  const settingsValidationError = useMemo(() => {
    if (!settings) {
      return "";
    }
    if (settings.threshold < 0 || settings.threshold > 1) {
      return "Duplicate threshold must be between 0 and 1.";
    }
    if (settings.min_save_ratio < 0 || settings.min_save_ratio > 1) {
      return "Min save ratio must be between 0 and 1.";
    }
    if (settings.min_grading_score < 0 || settings.min_grading_score > 10) {
      return "Min grading score must be between 0 and 10.";
    }
    if (settings.min_response_char_length < 1) {
      return "Min response length must be at least 1.";
    }
    if (settings.max_grading_json_retries < 0 || settings.max_naming_json_retries < 0) {
      return "JSON retry counts cannot be negative.";
    }
    if (settings.max_low_quality_retries < 0 || settings.max_generation_retries < 0) {
      return "Retry counts cannot be negative.";
    }
    return "";
  }, [settings]);

  const statCards = useMemo(() => {
    if (!dashboardStats) {
      return emptyDashboardCards.map((card) => ({
        ...card,
        body: dashboardError || card.body,
      }));
    }

    return [
      {
        eyebrow: "Balance",
        title: creditsSummary.balance === null ? "--" : `$${creditsSummary.balance.toFixed(2)}`,
        body:
          creditsSummary.totalCredits === null || creditsSummary.totalUsage === null
            ? "Current OpenRouter balance from the credits API."
            : `$${creditsSummary.totalCredits.toFixed(2)} total credits - $${creditsSummary.totalUsage.toFixed(2)} used`,
      },
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
        title: formatNullableNumber(dashboardStats.embedding_time, "s"),
        body: "Average embedding completion time from benchmark logs.",
      },
      {
        eyebrow: "Ingest",
        title: formatNullableNumber(dashboardStats.ingest_time, "s"),
        body: "Average ingest API duration from benchmark logs.",
      },
      {
        eyebrow: "Grading",
        title: formatNullableNumber(dashboardStats.grading_time, "s"),
        body: "Average grading completion time from benchmark logs.",
      },
      {
        eyebrow: "API Cost",
        title:
          typeof dashboardStats.api_cost === "number" && Number.isFinite(dashboardStats.api_cost)
            ? `$${dashboardStats.api_cost.toFixed(2)}`
            : "--",
        body: "Accumulated API cost recorded in the benchmark summary.",
      },
    ];
  }, [creditsSummary.balance, creditsSummary.totalCredits, creditsSummary.totalUsage, dashboardError, dashboardStats]);

  const quickSearchSections = useMemo<SearchSection[]>(() => {
    if (!trimmedSearch) {
      return [];
    }

    const query = trimmedSearch.toLowerCase();
    const includesQuery = (...values: Array<string | number | null | undefined>) =>
      values.some((value) => String(value ?? "").toLowerCase().includes(query));

    const datasetItems = datasets
      .filter((dataset) =>
        includesQuery(dataset.name, dataset.description, dataset.category, dataset.model, dataset.id),
      )
      .map((dataset) => ({
        id: `dataset-${dataset.id}`,
        title: dataset.name,
        meta: `${dataset.category ?? "uncategorized"} - ${dataset.model ?? "unknown model"}`,
        view: "datasets",
        targetId: dataset.id,
        kind: "dataset" as const,
      }));

    const batchItems = batchRuns
      .filter((run) => includesQuery(run.run_id, run.topic, run.requested_agent, run.status))
      .map((run) => ({
        id: `run-${run.run_id}`,
        title: run.topic || run.run_id,
        meta: `${run.status} - ${run.saved} saved - ${run.failed} failed`,
        view: "generation",
        targetRunId: run.run_id,
        kind: "run" as const,
      }));

    const documentItems = documents
      .filter((document) => includesQuery(document.name, document.file_type, document.source_material_ref))
      .map((document) => ({
        id: `document-${document.id}`,
        title: document.name,
        meta: `${document.file_type} - ${document.chunk_count} chunks`,
        view: "documents",
        targetId: document.id,
        kind: "document" as const,
      }));

    const exportItems = exportsHistory
      .filter((exportRow) =>
        includesQuery(exportRow.output_filename, exportRow.export_format, exportRow.status, exportRow.id),
      )
      .map((exportRow) => ({
        id: `export-${exportRow.id}`,
        title: exportRow.output_filename || `export-${exportRow.id}`,
        meta: `${exportRow.export_format} - ${exportRow.total_examples} examples`,
        view: "exports",
        kind: "export" as const,
      }));

    return [
      { id: "datasets", label: "Datasets", items: uniqueSearchItems(datasetItems).slice(0, 4) },
      { id: "runs", label: "Batch Runs", items: uniqueSearchItems(batchItems).slice(0, 4) },
      { id: "documents", label: "Documents", items: uniqueSearchItems(documentItems).slice(0, 3) },
      { id: "exports", label: "Exports", items: uniqueSearchItems(exportItems).slice(0, 3) },
    ].filter((section) => section.items.length > 0);
  }, [batchRuns, datasets, documents, exportsHistory, trimmedSearch]);

  const dashboardRecentRuns = useMemo(
    () =>
      batchRuns.slice(0, 4).map((run) => ({
        runId: run.run_id,
        topic: run.topic,
        status: run.status,
        saved: run.saved,
        failed: run.failed,
      })),
    [batchRuns],
  );

  const dashboardRecentDatasets = useMemo(
    () =>
      datasets.slice(0, 4).map((dataset) => ({
        id: dataset.id,
        name: dataset.name,
        category: dataset.category,
        exampleCount: dataset.exampleCount,
      })),
    [datasets],
  );

  const dashboardRecentDocuments = useMemo(
    () =>
      documents.slice(0, 4).map((document) => ({
        id: document.id,
        name: document.name,
        fileType: document.file_type,
        chunkCount: document.chunk_count,
      })),
    [documents],
  );

  const dashboardRecentExports = useMemo(
    () =>
      exportsHistory.slice(0, 4).map((exportRow) => ({
        id: exportRow.id,
        name: exportRow.output_filename || `export-${exportRow.id}`,
        format: exportRow.export_format,
        totalExamples: exportRow.total_examples,
      })),
    [exportsHistory],
  );

  const dashboardAttentionItems = useMemo(() => {
    const items: string[] = [];
    const failedRuns = batchRuns.filter((run) => run.failed > 0);
    const runningRuns = batchRuns.filter((run) => run.status === "running" || run.status === "queued");
    const uncategorizedCount = datasets.filter((dataset) => !dataset.category).length;
    const artifactGaps = exportsHistory.filter((exportRow) => !exportRow.has_artifact).length;

    if (failedRuns.length > 0) {
      items.push(`${failedRuns.length} recent batch runs have failures worth reviewing.`);
    }
    if (runningRuns.length > 0) {
      items.push(`${runningRuns.length} batch runs are still active right now.`);
    }
    if (uncategorizedCount > 0) {
      items.push(`${uncategorizedCount} datasets are still uncategorized.`);
    }
    if (artifactGaps > 0) {
      items.push(`${artifactGaps} exports do not currently have a downloadable artifact.`);
    }

    return items.slice(0, 4);
  }, [batchRuns, datasets, exportsHistory]);

  const loadBatchRuns = useCallback(async (signal?: AbortSignal) => {
    setBatchRunsError("");
    const response = await fetch("/api/dataset/batch", { signal });
    if (!response.ok) {
      throw new Error(`Batch request failed with ${response.status}`);
    }

    const payload = (await response.json()) as {
      data?: {
        runs?: BatchRunRow[];
      };
    };
    const rows = payload.data?.runs ?? [];
    setBatchRuns(rows);
    const groupedRows = Array.from(
      rows.reduce((groups, run) => {
        const key = run.request_group_id || run.run_id;
        const current = groups.get(key) ?? [];
        current.push(run);
        groups.set(key, current);
        return groups;
      }, new Map<string, BatchRunRow[]>()),
    ).map(([, grouped]) => aggregateBatchRunGroup(grouped));
    setSelectedBatchRunId((current) =>
      groupedRows.some((run) => run.run_id === current) ? current : groupedRows[0]?.run_id ?? "",
    );
  }, []);

  const loadBatchRunDetail = useCallback(
    async (groupId: string, memberRunIds: string[], signal?: AbortSignal) => {
      if (!groupId) {
        setSelectedBatchRunDetail(null);
        setBatchRunDetailError("");
        return;
      }

      const targetRunIds = memberRunIds.length > 0 ? memberRunIds : [groupId];

      setBatchRunDetailError("");
      const responses = await Promise.all(
        targetRunIds.map(async (runId) => {
          const response = await fetch(`/api/dataset/batch/${runId}`, { signal });
          if (!response.ok) {
            throw new Error(`Batch detail request failed with ${response.status}`);
          }
          const payload = (await response.json()) as {
            data?: BatchRunDetail;
          };
          if (!payload.data) {
            throw new Error("Batch detail response was empty.");
          }
          return payload.data;
        }),
      );
      const aggregatedDetail = aggregateBatchRunDetail(responses, groupId);
      setSelectedBatchRunDetail(aggregatedDetail);
      return aggregatedDetail;
    },
    [],
  );

  const loadDocuments = useCallback(async (signal?: AbortSignal) => {
    setDocumentsLoading(true);
    try {
      setDocumentsError("");
      const response = await fetch("/api/dataset/documents", { signal });
      if (!response.ok) {
        throw new Error(`Document request failed with ${response.status}`);
      }

      const payload = (await response.json()) as {
        data?: {
          documents?: DocumentRow[];
        };
      };
      const rows = payload.data?.documents ?? [];
      setDocuments(rows);
      setSelectedDocumentId((current) => (rows.some((document) => document.id === current) ? current : rows[0]?.id ?? 0));
    } finally {
      if (!signal?.aborted) {
        setDocumentsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboardStats() {
      try {
        setDashboardLoading(true);
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
      } finally {
        if (!controller.signal.aborted) {
          setDashboardLoading(false);
        }
      }
    }

    void loadDashboardStats();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!toast) {
      return;
    }

    const timeout = window.setTimeout(() => setToast(null), 2800);
    return () => window.clearTimeout(timeout);
  }, [toast]);

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

    async function loadCredits() {
      try {
        const [balanceResponse, totalCreditsResponse, totalUsageResponse] = await Promise.all([
          fetch("/api/utils/credits?index=0", { signal: controller.signal }),
          fetch("/api/utils/credits?index=1", { signal: controller.signal }),
          fetch("/api/utils/credits?index=2", { signal: controller.signal }),
        ]);

        if (!balanceResponse.ok || !totalCreditsResponse.ok || !totalUsageResponse.ok) {
          throw new Error("Credits request failed.");
        }

        const [balanceText, totalCreditsText, totalUsageText] = await Promise.all([
          balanceResponse.json(),
          totalCreditsResponse.json(),
          totalUsageResponse.json(),
        ]);

        setCreditsSummary({
          balance: parseNumericApiValue(balanceText),
          totalCredits: parseNumericApiValue(totalCreditsText),
          totalUsage: parseNumericApiValue(totalUsageText),
        });
      } catch {
        if (!controller.signal.aborted) {
          setCreditsSummary({
            balance: null,
            totalCredits: null,
            totalUsage: null,
          });
        }
      }
    }

    void loadCredits();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadModels() {
      try {
        setModelOptionsLoading(true);
        const response = await fetch("/api/utils/models?query=&limit=0", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Model request failed with ${response.status}`);
        }

        const payload = (await response.json()) as Array<string | { id: string; name?: string }>;
        const nextOptions = payload
          .map((entry) =>
            typeof entry === "string"
              ? { id: entry, name: entry }
              : { id: entry.id, name: entry.name ?? entry.id },
          )
          .filter((entry) => Boolean(entry.id));

        setModelOptions(nextOptions);
      } catch {
        if (!controller.signal.aborted) {
          setModelOptions([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setModelOptionsLoading(false);
        }
      }
    }

    void loadModels();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      void loadDatasets();
    }, trimmedSearch ? 180 : 0);

    async function loadDatasets() {
      try {
        setDatasetsError("");
        const params = new URLSearchParams({ limit: "100" });
        if (trimmedSearch) {
          params.set("q", trimmedSearch);
        }
        if (activeCategory !== "all") {
          params.set("category", activeCategory);
        }

        const response = await fetch(`/api/dataset?${params.toString()}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Dataset request failed with ${response.status}`);
        }

        const payload = (await response.json()) as {
          data?: {
            datasets?: Array<{
              id: number;
              name: string;
              description: string | null;
              category: string | null;
              model: string | null;
              example_count: number;
              generation_cost: number;
              grading_cost: number;
              total_cost: number;
            }>;
          };
        };

        const rows = (payload.data?.datasets ?? []).map((dataset) => ({
          id: dataset.id,
          name: dataset.name,
          description: dataset.description ?? "",
          category: dataset.category,
          model: dataset.model,
          exampleCount: dataset.example_count,
          generationCost: dataset.generation_cost,
          gradingCost: dataset.grading_cost,
          totalCost: dataset.total_cost,
        }));

        setDatasets(rows);
        setSelectedDatasetId((current) => (rows.some((dataset) => dataset.id === current) ? current : 0));
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setDatasetsError(error instanceof Error ? error.message : "Unable to load datasets.");
      }
    }

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [activeCategory, datasetsRefreshKey, trimmedSearch]);

  useEffect(() => {
    setExamplePreviewOffset(0);
  }, [selectedDatasetId]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDatasetDetail() {
      if (!selectedDatasetId) {
        return;
      }
      try {
        const params = new URLSearchParams({
          example_offset: String(examplePreviewOffset),
          example_limit: "3",
        });
        const response = await fetch(`/api/dataset/${selectedDatasetId}?${params.toString()}`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Dataset detail request failed with ${response.status}`);
        }
        const payload = (await response.json()) as {
          data?: {
            dataset?: {
              id: number;
              name: string;
              description: string | null;
              category: string | null;
              model: string | null;
              example_count: number;
              generation_cost: number;
              grading_cost: number;
              total_cost: number;
              examples_preview?: Array<{
                id: number | null;
                instruction: string;
                response: string;
              }>;
              examples_preview_offset?: number;
              examples_preview_limit?: number;
            };
          };
        };
        const dataset = payload.data?.dataset;
        if (!dataset) {
          return;
        }
        setDatasetDetail({
          id: dataset.id,
          name: dataset.name,
          description: dataset.description ?? "",
          category: dataset.category,
          model: dataset.model,
          exampleCount: dataset.example_count,
          generationCost: dataset.generation_cost,
          gradingCost: dataset.grading_cost,
          totalCost: dataset.total_cost,
          examplesPreview: dataset.examples_preview ?? [],
          examplesPreviewOffset: dataset.examples_preview_offset ?? 0,
          examplesPreviewLimit: dataset.examples_preview_limit ?? 3,
        });
      } catch {
        if (!controller.signal.aborted) {
          setDatasetDetail(null);
        }
      }
    }

    void loadDatasetDetail();
    return () => controller.abort();
  }, [datasetsRefreshKey, examplePreviewOffset, selectedDatasetId]);

  useEffect(() => {
    const controller = new AbortController();

    async function run() {
      try {
        await loadBatchRuns(controller.signal);
      } catch (error) {
        if (!controller.signal.aborted) {
          setBatchRunsError(error instanceof Error ? error.message : "Unable to load batch runs.");
        }
      }
    }

    void run();
    return () => controller.abort();
  }, [batchRunsRefreshKey, loadBatchRuns]);

  useEffect(() => {
    const controller = new AbortController();

    async function run() {
      try {
        setSelectedBatchRunDetail(null);
        setBatchRunDetailLoading(Boolean(selectedBatchRunId));
        await loadBatchRunDetail(selectedBatchRunId, selectedBatchMemberRunIds, controller.signal);
      } catch (error) {
        if (!controller.signal.aborted) {
          setSelectedBatchRunDetail(null);
          setBatchRunDetailError(error instanceof Error ? error.message : "Unable to load batch detail.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setBatchRunDetailLoading(false);
        }
      }
    }

    setBatchStreamEvents([]);
    setBatchStreamStatus(selectedBatchRunId ? "connecting" : "idle");
    void run();
    return () => controller.abort();
  }, [loadBatchRunDetail, selectedBatchMemberRunKey, selectedBatchRunId]);

  useEffect(() => {
    if (!selectedBatchRunId) {
      setBatchStreamStatus("idle");
      setBatchStreamEvents([]);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setBatchStreamStatus("connecting");
    setBatchStreamEvents([]);

    let intervalId = 0;

    async function refreshSelectedBatch() {
      try {
        const [, detail] = await Promise.all([
          loadBatchRuns(controller.signal),
          loadBatchRunDetail(selectedBatchRunId, selectedBatchMemberRunIds, controller.signal),
        ]);
        if (!cancelled) {
          const terminal = detail && ["completed", "failed", "cancelled"].includes(detail.status);
          setBatchStreamStatus(terminal ? "completed" : "live");
          if (terminal && intervalId) {
            window.clearInterval(intervalId);
          }
        }
      } catch {
        if (!cancelled && !controller.signal.aborted) {
          setBatchStreamStatus("offline");
        }
      }
    }

    void refreshSelectedBatch();
    intervalId = window.setInterval(() => {
      void refreshSelectedBatch();
    }, 3000);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [loadBatchRunDetail, loadBatchRuns, selectedBatchMemberRunKey, selectedBatchRunId]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadSettings() {
      try {
        setSettingsLoading(true);
        setSettingsError("");
        const response = await fetch("/api/dashboard/settings", { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Settings request failed with ${response.status}`);
        }
        const payload = (await response.json()) as SettingsValues;
        setSettings(payload);
        setInitialSettings(payload);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setSettingsError(error instanceof Error ? error.message : "Unable to load settings.");
      } finally {
        if (!controller.signal.aborted) {
          setSettingsLoading(false);
        }
      }
    }

    void loadSettings();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!settings?.default_model) {
      return;
    }

    setGenerationForm((current) => {
      const currentModel = current.model.trim();
      if (currentModel && currentModel !== LEGACY_GENERATION_MODEL_DEFAULT) {
        return current;
      }
      return { ...current, model: settings.default_model };
    });
  }, [settings?.default_model]);

  useEffect(() => {
    const controller = new AbortController();

    async function run() {
      try {
        await loadDocuments(controller.signal);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setDocumentsError(error instanceof Error ? error.message : "Unable to load source documents.");
      }
    }

    void run();
    return () => controller.abort();
  }, [documentsRefreshKey, loadDocuments]);

  useEffect(() => {
    const controller = new AbortController();

    async function run() {
      if (!selectedDocumentId) {
        setSelectedDocumentDetail(null);
        return;
      }

      try {
        setDocumentDetailLoading(true);
        const response = await fetch(`/api/dataset/documents/${selectedDocumentId}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Document detail request failed with ${response.status}`);
        }
        const payload = (await response.json()) as {
          data?: DocumentDetail;
        };
        setSelectedDocumentDetail(payload.data ?? null);
      } catch {
        if (!controller.signal.aborted) {
          setSelectedDocumentDetail(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setDocumentDetailLoading(false);
        }
      }
    }

    void run();
    return () => controller.abort();
  }, [selectedDocumentId]);

  useEffect(() => {
    setDocumentChunksExpanded(false);
  }, [selectedDocumentId]);

  useEffect(() => {
    const controller = new AbortController();

    async function loadExports() {
      try {
        setExportsLoading(true);
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
      } finally {
        if (!controller.signal.aborted) {
          setExportsLoading(false);
        }
      }
    }

    void loadExports();
    return () => controller.abort();
  }, []);

  async function handleBatchLaunch() {
    try {
      setGenerationSubmitting(true);
      setGenerationMessage("");
      const requestGroupId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `batch-${Date.now()}`;

      const topics = generationForm.topics
        .split("\n")
        .map((value) => value.trim())
        .filter(Boolean);
      const agentTypes = generationForm.agentTypes
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
      const sourceDatasetIds = generationForm.sourceDatasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0);
      const personalityInstructions = generationForm.personalityInstructions.trim();
      const sourceMaterial =
        sourceDatasetIds.length > 0 && personalityInstructions
          ? [...sourceDatasetIds, personalityInstructions]
          : sourceDatasetIds.length > 0
            ? sourceDatasetIds
            : personalityInstructions || null;

      const response = await fetch("/api/dataset/batch/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          amount: Number(generationForm.amount),
          request_group_id: requestGroupId,
          topics,
          agent_types: agentTypes,
          allow_topic_variations: generationForm.allowTopicVariations,
          conversation_length_mode: generationForm.conversationLengthMode,
          ex_amt: Number(generationForm.exAmt),
          auto_merge_related: false,
          auto_merge_similarity_threshold: Number(globalMergeThreshold),
          source_material: sourceMaterial,
          source_material_mode: generationForm.sourceMaterialMode,
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
      const autoMerge = payload.data?.auto_merge;
      const autoMergeSuffix =
        autoMerge?.requested && autoMerge.message
          ? ` Auto-merge: ${autoMerge.message}${
              autoMerge.status === "merged" && autoMerge.result?.created_dataset_ids?.length
                ? ` New dataset IDs: ${autoMerge.result.created_dataset_ids.join(", ")}.`
                : ""
            }`
          : "";
      setGenerationMessage(
        batchIds.length > 0
          ? `Batch queued successfully. Batch group: ${requestGroupId}.${autoMergeSuffix}`
          : `${payload.message}${autoMergeSuffix}`,
      );
      if (batchIds.length > 0) {
        setSelectedBatchRunId(requestGroupId);
      }
      setBatchRunsRefreshKey((current) => current + 1);
    } catch (error) {
      setGenerationMessage(error instanceof Error ? error.message : "Unable to launch batch.");
    } finally {
      setGenerationSubmitting(false);
    }
  }

  function handleQuickSearchSelect(view: string, targetId?: number, targetRunId?: string) {
    setActiveView(view);
    if (view === "datasets" && targetId) {
      setSelectedDatasetId(targetId);
    }
    if (view === "documents" && targetId) {
      setSelectedDocumentId(targetId);
    }
    if (view === "generation" && targetRunId) {
      setSelectedBatchRunId(targetRunId);
    }
    dismissSearch(setSearchQuery);
  }

  function handleOpenRunFromDashboard(runId: string) {
    setActiveView("generation");
    setSelectedBatchRunId(runId);
  }

  function handleOpenDatasetFromDashboard(datasetId: number) {
    setDatasetsRefreshKey((current) => current + 1);
    setActiveView("datasets");
    setSelectedDatasetId(datasetId);
  }

  function handleOpenDocumentFromDashboard(documentId: number) {
    setActiveView("documents");
    setSelectedDocumentId(documentId);
  }

  function handleOpenExportsFromDashboard() {
    setActiveView("exports");
  }

  function handleSearchDatasetAction(action: "view" | "export" | "delete", datasetId: number) {
    setActiveView("datasets");
    setSelectedDatasetId(datasetId);
    dismissSearch(setSearchQuery);

    if (action === "view") {
      setIsDatasetViewOpen(true);
      return;
    }

    if (action === "export") {
      window.open(`/api/dataset/${datasetId}/export`, "_blank", "noopener,noreferrer");
      setToast({ tone: "success", message: "Export opened in a new tab." });
      return;
    }

    setIsDeleteConfirmOpen(true);
  }

  function handleRefreshDatasets() {
      setDatasetsRefreshKey((current) => current + 1);
      setToast({ tone: "success", message: "Refreshing dataset library." });
    }

  function handleRefreshBatchRuns() {
    setBatchRunsRefreshKey((current) => current + 1);
    setToast({ tone: "success", message: "Refreshing batch monitor." });
  }

  async function handleClearCompletedBatchRuns() {
    try {
      const response = await fetch("/api/dataset/batch", { method: "DELETE" });
      const payload = (await response.json()) as {
        success?: boolean;
        message?: string;
        data?: { deleted_run_ids?: string[]; deleted_count?: number };
      };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Batch clear request failed with ${response.status}`);
      }

      const deletedRunIds = new Set(payload.data?.deleted_run_ids ?? []);
      const selectedGroupedRun = groupedBatchRuns.find((run) => run.run_id === selectedBatchRunId);
      const selectedGroupFullyDeleted = Boolean(
        selectedGroupedRun &&
          (selectedGroupedRun.member_run_ids ?? []).length > 0 &&
          (selectedGroupedRun.member_run_ids ?? []).every((runId) => deletedRunIds.has(runId)),
      );
      setBatchRuns((current) => current.filter((run) => !deletedRunIds.has(run.run_id)));
      setSelectedBatchRunId((current) => {
        if (!current) {
          return current;
        }
        return deletedRunIds.has(current) || selectedGroupFullyDeleted ? "" : current;
      });
      if (selectedBatchRunId && (deletedRunIds.has(selectedBatchRunId) || selectedGroupFullyDeleted)) {
        setSelectedBatchRunDetail(null);
        setBatchRunDetailError("");
        setBatchStreamEvents([]);
        setBatchStreamStatus("idle");
      }
      setBatchRunsRefreshKey((current) => current + 1);
      setToast({ tone: "success", message: payload.message || "Completed runs cleared." });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to clear completed runs.",
      });
    }
  }

  function handleRefreshDocuments() {
    setDocumentsRefreshKey((current) => current + 1);
    setToast({ tone: "success", message: "Refreshing document library." });
  }

  async function handleUploadDocument() {
    if (!documentUploadFile) {
      return;
    }

    try {
      setDocumentUploadPending(true);
      setDocumentUploadMessage("");
      const formData = new FormData();
      formData.append("file", documentUploadFile);
      formData.append("intake_mode", documentUploadMode);
      formData.append("chunk_char_size", documentUploadChunkSize || "2000");
      formData.append("chunk_overlap", documentUploadChunkOverlap || "200");

      const response = await fetch("/api/dataset/intake/upload", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as {
        success?: boolean;
        message?: string;
        data?: {
          document_id?: number;
          dataset_id?: number;
        };
      };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Upload failed with ${response.status}`);
      }

      setDocumentUploadMessage(payload.message || "Upload completed.");
      setDocumentUploadFile(null);
      setDocumentUploadAdvancedOpen(false);
      setDocumentChunksExpanded(false);
      setDocumentsRefreshKey((current) => current + 1);
      if (documentUploadMode !== "examples" && payload.data?.document_id) {
        setSelectedDocumentId(payload.data.document_id);
      }
      if (documentUploadMode === "examples" && payload.data?.dataset_id) {
        setDocumentUploadMessage(
          `${payload.message || "Upload completed."} Dataset ID ${payload.data.dataset_id} was created.`,
        );
      }
      setToast({ tone: "success", message: payload.message || "Document uploaded." });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to upload file.";
      setDocumentUploadMessage(message);
      setToast({ tone: "error", message });
    } finally {
      setDocumentUploadPending(false);
    }
  }

  async function handleScraperImport() {
    if (!scraperText.trim()) {
      return;
    }

    try {
      setScraperPending(true);
      setScraperMessage("");
      const response = await fetch("/api/dataset/intake/scraper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_name: scraperDatasetName || undefined,
          records: [{ text: scraperText }],
        }),
      });
      const payload = (await response.json()) as {
        success?: boolean;
        message?: string;
        data?: {
          dataset_id?: number;
        };
      };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Scraper intake failed with ${response.status}`);
      }

      setScraperMessage(payload.message || "Text imported.");
      if (payload.data?.dataset_id) {
        setScraperMessage(`${payload.message || "Text imported."} Dataset ID ${payload.data.dataset_id} was created.`);
      }
      setScraperText("");
      setScraperDatasetName("");
      setDocumentChunksExpanded(false);
      setDatasetsRefreshKey((current) => current + 1);
      setToast({ tone: "success", message: payload.message || "Scraper text imported." });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to import scraper text.";
      setScraperMessage(message);
      setToast({ tone: "error", message });
    } finally {
      setScraperPending(false);
    }
  }

  function handleExampleShift(step: number) {
    if (!selectedDataset) {
      return;
    }
    const limit = datasetDetail?.examplesPreviewLimit ?? 3;
    const maxOffset = Math.max(0, selectedDataset.exampleCount - limit);
    setExamplePreviewOffset((current) => {
      const baseOffset = datasetDetail?.examplesPreviewOffset ?? current;
      return Math.min(maxOffset, Math.max(0, baseOffset + step));
      });
    }

  async function handleRemoveExample(exampleId: number) {
    if (!selectedDataset) {
      return;
    }

    try {
      setExampleDeletePendingId(exampleId);
      const response = await fetch(`/api/dataset/examples/${exampleId}`, { method: "DELETE" });
      const payload = (await response.json()) as {
        success?: boolean;
        message?: string;
        data?: { remaining_examples?: number };
      };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Example delete request failed with ${response.status}`);
      }

      const remainingExamples = payload.data?.remaining_examples ?? Math.max(0, selectedDataset.exampleCount - 1);
      setDatasetDetail((current) => {
        if (!current || current.id !== selectedDataset.id) {
          return current;
        }

        const nextPreview = current.examplesPreview.filter((example) => example.id !== exampleId);
        const nextLimit = current.examplesPreviewLimit || 3;
        const nextOffset =
          remainingExamples > 0 && nextPreview.length === 0
            ? Math.max(0, Math.min(current.examplesPreviewOffset - nextLimit, remainingExamples - 1))
            : current.examplesPreviewOffset;
        if (nextOffset !== current.examplesPreviewOffset) {
          setExamplePreviewOffset(nextOffset);
        }

        return {
          ...current,
          exampleCount: remainingExamples,
          examplesPreview: nextOffset === current.examplesPreviewOffset ? nextPreview : current.examplesPreview,
        };
      });
      setDatasets((current) =>
        current.map((dataset) =>
          dataset.id === selectedDataset.id
            ? {
                ...dataset,
                exampleCount: remainingExamples,
              }
            : dataset,
        ),
      );
      setToast({ tone: "success", message: payload.message || "Example removed." });
      if (datasetDetail && datasetDetail.examplesPreview.length <= 1) {
        setDatasetsRefreshKey((current) => current + 1);
      }
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to remove example.",
      });
    } finally {
      setExampleDeletePendingId(null);
    }
  }

  async function handleUpdateExample(exampleId: number, instruction: string, response: string) {
    if (!selectedDataset) {
      return;
    }

    try {
      setExampleSavePendingId(exampleId);
      const payload = {
        instruction: instruction.trim(),
        response: response.trim(),
      };
      const responseResult = await fetch(`/api/dataset/examples/${exampleId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const responsePayload = (await responseResult.json()) as {
        success?: boolean;
        message?: string;
        data?: {
          example?: ExamplePreview & { dataset_id?: number };
        };
      };
      if (!responseResult.ok || responsePayload.success === false) {
        throw new Error(responsePayload.message || `Example update request failed with ${responseResult.status}`);
      }

      const updatedExample = responsePayload.data?.example;
      if (!updatedExample) {
        throw new Error("Example update response did not include the saved example.");
      }

      setDatasetDetail((current) => {
        if (!current || current.id !== selectedDataset.id) {
          return current;
        }
        return {
          ...current,
          examplesPreview: current.examplesPreview.map((example) =>
            example.id === exampleId
              ? {
                  ...example,
                  instruction: updatedExample.instruction,
                  response: updatedExample.response,
                }
              : example,
          ),
        };
      });
      setToast({ tone: "success", message: responsePayload.message || "Example updated." });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to update example.",
      });
      throw error;
    } finally {
      setExampleSavePendingId(null);
    }
  }

  function handleExportDataset() {
    if (!selectedDataset) {
      return;
    }
    window.open(`/api/dataset/${selectedDataset.id}/export`, "_blank", "noopener,noreferrer");
    setToast({ tone: "success", message: "Export opened in a new tab." });
  }

  async function handleDeleteDataset() {
    if (!selectedDataset) {
      return;
    }
    try {
      setDatasetDeletePending(true);
      const response = await fetch(`/api/dataset/remove/${selectedDataset.id}`, { method: "DELETE" });
      const payload = (await response.json()) as { message?: string; success?: boolean };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Delete request failed with ${response.status}`);
      }

      const currentIndex = datasets.findIndex((dataset) => dataset.id === selectedDataset.id);
      const nextDataset =
        datasets[currentIndex + 1] ?? datasets[currentIndex - 1] ?? null;

      setDatasets((current) => current.filter((dataset) => dataset.id !== selectedDataset.id));
      setDatasetDetail(null);
      setSelectedDatasetId(nextDataset?.id ?? 0);
      setIsDatasetViewOpen(false);
      setIsDeleteConfirmOpen(false);
      setToast({ tone: "success", message: payload.message || "Dataset deleted." });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to delete dataset.",
      });
    } finally {
      setDatasetDeletePending(false);
    }
  }

  async function handleCopyDatasetId() {
    if (!selectedDataset) {
      return;
    }
    try {
      await navigator.clipboard.writeText(String(selectedDataset.id));
      setToast({ tone: "success", message: `Copied dataset ID ${selectedDataset.id}.` });
    } catch {
      setToast({ tone: "error", message: "Unable to copy dataset ID." });
    }
  }

  async function handleCopyDocumentRef() {
    if (!selectedDocument) {
      return;
    }
    try {
      await navigator.clipboard.writeText(selectedDocument.source_material_ref);
      setToast({ tone: "success", message: `Copied ${selectedDocument.source_material_ref}.` });
    } catch {
      setToast({ tone: "error", message: "Unable to copy source reference." });
    }
  }

  async function handleCopyChunkContent(content: string) {
    try {
      await navigator.clipboard.writeText(content);
      setToast({ tone: "success", message: "Chunk copied to clipboard." });
    } catch {
      setToast({ tone: "error", message: "Unable to copy chunk content." });
    }
  }

  function handleSettingsFieldChange(field: keyof SettingsValues, value: string) {
    setSettings((current) => {
      if (!current) {
        return current;
      }
      const numericFields = new Set<keyof SettingsValues>([
        "threshold",
        "min_grading_score",
        "min_response_char_length",
        "max_grading_json_retries",
        "max_naming_json_retries",
        "max_low_quality_retries",
        "max_generation_retries",
        "min_save_ratio",
      ]);
      return {
        ...current,
        [field]: numericFields.has(field) ? Number(value) : value,
      };
    });
  }

  function handleOpenModelPicker(target: Exclude<ModelPickerTarget, null>) {
    setModelPickerTarget(target);
    setModelPickerQuery("");
  }

  function handleCloseModelPicker() {
    setModelPickerTarget(null);
    setModelPickerQuery("");
  }

  function handleSelectModel(modelId: string) {
    if (modelPickerTarget === "generation") {
      setGenerationForm((current) => ({ ...current, model: modelId }));
    } else if (modelPickerTarget) {
      handleSettingsFieldChange(modelPickerTarget, modelId);
    }
    handleCloseModelPicker();
  }

  function handleResetSettings() {
    if (!initialSettings) {
      return;
    }
    setSettings(initialSettings);
    setSettingsError("");
    setSettingsMessage("Reverted local edits to the last saved settings.");
  }

  async function handleSaveSettings() {
    if (!settings) {
      return;
    }
    if (settingsValidationError) {
      setSettingsError(settingsValidationError);
      return;
    }
    try {
      setSettingsSaving(true);
      setSettingsError("");
      setSettingsMessage("");
      const response = await fetch("/api/dashboard/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (!response.ok) {
        throw new Error(`Settings update failed with ${response.status}`);
      }
      const payload = (await response.json()) as SettingsValues;
      setSettings(payload);
      setInitialSettings(payload);
      setSettingsMessage("Settings saved.");
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "Unable to save settings.");
    } finally {
      setSettingsSaving(false);
    }
  }

  async function handleBatchAction(action: BatchAction) {
    if (!selectedBatchRunId) {
      return;
    }

    try {
      setBatchActionPending(action);
      const targetRunIds = selectedBatchMemberRunIds.length > 0 ? selectedBatchMemberRunIds : [selectedBatchRunId];
      const results = await Promise.all(
        targetRunIds.map(async (runId) => {
          const response = await fetch(
            action === "delete" ? `/api/dataset/batch/${runId}` : `/api/dataset/batch/${runId}/${action}`,
            {
              method: action === "delete" ? "DELETE" : "POST",
            },
          );
          const payload = (await response.json()) as {
            success?: boolean;
            message?: string;
            data?: BatchRunDetail | { current_summary?: BatchRunDetail };
          };
          if (!response.ok || payload.success === false) {
            throw new Error(payload.message || `Batch action failed with ${response.status}`);
          }
          return payload;
        }),
      );
      const payload = results[0];

      if (action === "delete") {
        const currentIndex = groupedBatchRuns.findIndex((run) => run.run_id === selectedBatchRunId);
        const nextRun = groupedBatchRuns[currentIndex + 1] ?? groupedBatchRuns[currentIndex - 1] ?? null;

        setBatchRuns((current) =>
          current.filter((run) => (run.request_group_id || run.run_id) !== selectedBatchRunId),
        );
        setSelectedBatchRunId(nextRun?.run_id ?? "");
        setSelectedBatchRunDetail(null);
        setBatchRunDetailError("");
        setBatchStreamEvents([]);
        setBatchStreamStatus(nextRun ? "connecting" : "idle");
        setToast({ tone: "success", message: payload.message || "Batch run removed." });
        return;
      }

      const nextDetails = results
        .map((entry) =>
          entry.data && "current_summary" in entry.data
            ? entry.data.current_summary ?? null
            : (entry.data as BatchRunDetail | undefined) ?? null,
        )
        .filter((entry): entry is BatchRunDetail => Boolean(entry));
      const nextDetail = aggregateBatchRunDetail(nextDetails, selectedBatchRunId);
      if (nextDetail) {
        setSelectedBatchRunDetail(nextDetail);
      }
      setBatchRunsRefreshKey((current) => current + 1);
      setToast({ tone: "success", message: payload.message || "Batch action completed." });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to update batch state.",
      });
    } finally {
      setBatchActionPending("");
    }
  }

  function handleOpenTargetedMerge() {
    if (selectedDatasetId > 0) {
      setTargetedMergeDatasetIds((current) =>
        current.length > 0 ? current : [selectedDatasetId],
      );
    }
    setTargetedMergeThreshold(globalMergeThreshold);
    setIsTargetedMergeOpen(true);
  }

  function handleToggleMergeDataset(datasetId: number) {
    setTargetedMergeDatasetIds((current) =>
      current.includes(datasetId) ? current.filter((value) => value !== datasetId) : [...current, datasetId],
    );
  }

  async function handleSubmitTargetedMerge() {
    if (targetedMergeDatasetIds.length < 2) {
      return;
    }

    try {
      setTargetedMergePending(true);
      const response = await fetch("/api/dataset/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_ids: targetedMergeDatasetIds,
          dataset_similarity_threshold: Number(targetedMergeThreshold),
          delete_originals: targetedMergeDeleteOriginals,
        }),
      });
      const payload = (await response.json()) as MergeResponse;
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Merge request failed with ${response.status}`);
      }

      const createdIds = payload.data?.created_dataset_ids ?? [];
      setIsTargetedMergeOpen(false);
      setTargetedMergeDatasetIds([]);
      setTargetedMergeDeleteOriginals(false);
      setDatasetsRefreshKey((current) => current + 1);
      if (createdIds.length > 0) {
        setSelectedDatasetId(createdIds[0]);
      }
      setToast({
        tone: "success",
        message:
          payload.message ||
          (createdIds.length > 0
            ? `Merge completed. New dataset ID ${createdIds[0]}.`
            : "Selected datasets merged."),
      });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to merge selected datasets.",
      });
    } finally {
      setTargetedMergePending(false);
    }
  }

  async function handleConfirmGlobalMerge() {
    try {
      setGlobalMergePending(true);
      const response = await fetch("/api/dataset/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_similarity_threshold: Number(globalMergeThreshold),
          delete_originals: globalMergeDeleteOriginals,
        }),
      });
      const payload = (await response.json()) as MergeResponse;
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Merge request failed with ${response.status}`);
      }

      const createdIds = payload.data?.created_dataset_ids ?? [];
      setIsGlobalMergeConfirmOpen(false);
      setDatasetsRefreshKey((current) => current + 1);
      if (createdIds.length > 0) {
        setSelectedDatasetId(createdIds[0]);
      }
      setToast({
        tone: "success",
        message:
          payload.message ||
          (createdIds.length > 0
            ? `Library merge completed. New dataset ID ${createdIds[0]}.`
            : "Library merge completed."),
      });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to merge related datasets.",
      });
    } finally {
      setGlobalMergePending(false);
    }
  }

  async function handleDeleteDocument() {
    if (!selectedDocument) {
      return;
    }

    try {
      setDocumentActionPending(true);
      const response = await fetch(`/api/dataset/documents/${selectedDocument.id}`, {
        method: "DELETE",
      });
      const payload = (await response.json()) as { success?: boolean; message?: string };
      if (!response.ok || payload.success === false) {
        throw new Error(payload.message || `Document delete failed with ${response.status}`);
      }

      const currentIndex = documents.findIndex((document) => document.id === selectedDocument.id);
      const nextDocument = documents[currentIndex + 1] ?? documents[currentIndex - 1] ?? null;

      setDocuments((current) => current.filter((document) => document.id !== selectedDocument.id));
      setSelectedDocumentId(nextDocument?.id ?? 0);
      setSelectedDocumentDetail(null);
      setIsDocumentViewOpen(false);
      setIsDocumentDeleteConfirmOpen(false);
      setToast({ tone: "success", message: payload.message || "Document deleted." });
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to delete document.",
      });
    } finally {
      setDocumentActionPending(false);
    }
  }

  function handleExportFieldChange(
    field: "datasetIds" | "format" | "minScore" | "maxExamples" | "trainValSplit" | "dedupePass" | "shuffle",
    value: string | boolean,
  ) {
    setExportForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function handleToggleExportDataset(datasetId: number) {
    setExportForm((current) => {
      const parsedIds = current.datasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0);

      const nextIds = parsedIds.includes(datasetId)
        ? parsedIds.filter((value) => value !== datasetId)
        : [...parsedIds, datasetId];

      return {
        ...current,
        datasetIds: nextIds.join(", "),
      };
    });
  }

  function handleToggleGenerationSourceDataset(datasetId: number) {
    setGenerationForm((current) => {
      const parsedIds = current.sourceDatasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0);

      const nextIds = parsedIds.includes(datasetId)
        ? parsedIds.filter((value) => value !== datasetId)
        : [...parsedIds, datasetId];

      return {
        ...current,
        sourceDatasetIds: nextIds.join(", "),
      };
    });
  }

  function handleClearGenerationSourceDatasets() {
    setGenerationForm((current) => ({
      ...current,
      sourceDatasetIds: "",
    }));
  }

  function handleClearExportDatasets() {
    setExportForm((current) => ({
      ...current,
      datasetIds: "",
    }));
  }

  function handleDownloadExport(exportId: number) {
    window.open(`/api/dataset/exports/${exportId}/download`, "_blank", "noopener,noreferrer");
    setToast({ tone: "success", message: "Export download opened." });
  }

  async function handleRerunExport(exportId: number) {
    try {
      setExportActionPendingId(exportId);
      const response = await fetch(`/api/dataset/exports/${exportId}/rerun`, {
        method: "POST",
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { message?: string } | null;
        throw new Error(payload?.message || `Export rerun failed with ${response.status}`);
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      window.open(objectUrl, "_blank", "noopener,noreferrer");
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000);

      setToast({ tone: "success", message: "Export rerun completed and opened." });

      const refreshedExports = await fetch("/api/dataset/exports/history");
      if (refreshedExports.ok) {
        const refreshedPayload = (await refreshedExports.json()) as {
          data?: {
            exports?: ExportRow[];
          };
        };
        setExportsHistory(refreshedPayload.data?.exports ?? []);
      }
    } catch (error) {
      setToast({
        tone: "error",
        message: error instanceof Error ? error.message : "Unable to rerun export.",
      });
    } finally {
      setExportActionPendingId(null);
    }
  }

  async function handleCreateExport() {
    try {
      setExportCreatePending(true);
      setExportMessage("");
      const datasetIds = exportForm.datasetIds
        .split(/[\s,]+/)
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value > 0);

      if (datasetIds.length === 0) {
        throw new Error("Enter at least one dataset ID to export.");
      }

      const response = await fetch("/api/dataset/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_ids: datasetIds,
          export_format: exportForm.format,
          min_score: exportForm.minScore ? Number(exportForm.minScore) : null,
          max_examples: exportForm.maxExamples ? Number(exportForm.maxExamples) : null,
          train_val_split: exportForm.trainValSplit ? Number(exportForm.trainValSplit) : null,
          dedupe_pass: exportForm.dedupePass,
          shuffle: exportForm.shuffle,
        }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { message?: string } | null;
        throw new Error(payload?.message || `Export creation failed with ${response.status}`);
      }

      const historyId = response.headers.get("X-Export-History-Id");
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      window.open(objectUrl, "_blank", "noopener,noreferrer");
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000);

      setExportMessage(historyId ? `Export created. History ID ${historyId}.` : "Export created.");
      setExportForm(defaultExportForm);
      setToast({ tone: "success", message: "Export created and opened." });

      const refreshedExports = await fetch("/api/dataset/exports/history");
      if (refreshedExports.ok) {
        const refreshedPayload = (await refreshedExports.json()) as {
          data?: {
            exports?: ExportRow[];
          };
        };
        setExportsHistory(refreshedPayload.data?.exports ?? []);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to create export.";
      setExportMessage(message);
      setToast({ tone: "error", message });
    } finally {
      setExportCreatePending(false);
    }
  }

  return (
    <AppShell
      activeView={activeView}
      description={currentView.description}
      eyebrow={currentView.eyebrow}
      healthState={healthState}
      navItems={navItems.map(({ id, label }) => ({ id, label }))}
      onNavigate={setActiveView}
      onNewBatch={() => setActiveView("generation")}
      onSearchChange={setSearchQuery}
      searchPanel={
        <SearchResults
          onDatasetAction={handleSearchDatasetAction}
          onSelect={handleQuickSearchSelect}
          sections={quickSearchSections}
          trimmedSearch={trimmedSearch}
        />
      }
      searchQuery={searchQuery}
      title={currentView.title}
    >
      {activeView === "dashboard" ? (
        <DashboardView
          attentionItems={dashboardAttentionItems}
          cards={statCards}
          error={dashboardError}
          loading={dashboardLoading}
          onOpenDataset={handleOpenDatasetFromDashboard}
          onOpenDocument={handleOpenDocumentFromDashboard}
          onOpenExports={handleOpenExportsFromDashboard}
          onOpenRun={handleOpenRunFromDashboard}
          recentDatasets={dashboardRecentDatasets}
          recentDocuments={dashboardRecentDocuments}
          recentExports={dashboardRecentExports}
          recentRuns={dashboardRecentRuns}
        />
      ) : null}
      {activeView === "datasets" ? (
        <DatasetsView
          activeCategory={activeCategory}
          categories={datasetCategories}
          deletePending={datasetDeletePending}
          datasets={datasets}
          datasetsError={datasetsError}
          exampleDeletePendingId={exampleDeletePendingId}
          exampleSavePendingId={exampleSavePendingId}
          globalMergeDeleteOriginals={globalMergeDeleteOriginals}
          globalMergePending={globalMergePending}
          globalMergeThreshold={globalMergeThreshold}
          isGlobalMergeConfirmOpen={isGlobalMergeConfirmOpen}
          isTargetedMergeOpen={isTargetedMergeOpen}
          onCategoryChange={setActiveCategory}
          onClearMergeDatasets={() => setTargetedMergeDatasetIds([])}
          onCloseView={() => setIsDatasetViewOpen(false)}
          onCloseDeleteConfirm={() => setIsDeleteConfirmOpen(false)}
          onCloseGlobalMergeConfirm={() => setIsGlobalMergeConfirmOpen(false)}
          onCloseTargetedMerge={() => setIsTargetedMergeOpen(false)}
          onConfirmGlobalMerge={() => void handleConfirmGlobalMerge()}
          onCopyId={() => void handleCopyDatasetId()}
          onDelete={() => void handleDeleteDataset()}
          onExampleShift={handleExampleShift}
          onGlobalMergeDeleteOriginalsChange={setGlobalMergeDeleteOriginals}
          onGlobalMergeThresholdChange={setGlobalMergeThreshold}
          onOpenGlobalMergeConfirm={() => setIsGlobalMergeConfirmOpen(true)}
          onOpenTargetedMerge={handleOpenTargetedMerge}
          onRemoveExample={(exampleId) => void handleRemoveExample(exampleId)}
          onUpdateExample={(exampleId, instruction, response) => handleUpdateExample(exampleId, instruction, response)}
          onExport={handleExportDataset}
          onOpenDeleteConfirm={() => setIsDeleteConfirmOpen(true)}
          onRefresh={handleRefreshDatasets}
          onSelectDataset={setSelectedDatasetId}
          onSubmitTargetedMerge={() => void handleSubmitTargetedMerge()}
          onTargetedMergeDeleteOriginalsChange={setTargetedMergeDeleteOriginals}
          onTargetedMergeThresholdChange={setTargetedMergeThreshold}
          onToggleMergeDataset={handleToggleMergeDataset}
          onView={() => setIsDatasetViewOpen(true)}
          isDeleteConfirmOpen={isDeleteConfirmOpen}
          isViewOpen={isDatasetViewOpen}
          mergePending={targetedMergePending}
          selectedDataset={selectedDataset}
          targetedMergeDatasetIds={targetedMergeDatasetIds}
          targetedMergeDeleteOriginals={targetedMergeDeleteOriginals}
          targetedMergeThreshold={targetedMergeThreshold}
          trimmedSearch={trimmedSearch}
        />
      ) : null}
      {activeView === "generation" ? (
        <GenerationView
          actionPending={batchActionPending}
          availableDatasets={datasets.map((dataset) => ({ id: dataset.id, name: dataset.name }))}
          detailLoading={batchRunDetailLoading}
          detailError={batchRunDetailError}
          form={generationForm}
          message={generationMessage}
          modelLoading={modelOptionsLoading}
          modelOptions={modelOptions}
          onBatchAction={(action) => void handleBatchAction(action)}
          onClearSourceDatasets={handleClearGenerationSourceDatasets}
          onFormChange={(updater) => setGenerationForm((current) => updater(current))}
          onOpenDataset={handleOpenDatasetFromDashboard}
          onOpenModelPicker={() => handleOpenModelPicker("generation")}
          onClearCompletedRuns={() => void handleClearCompletedBatchRuns()}
          onRefreshRuns={handleRefreshBatchRuns}
          onReset={() =>
            setGenerationForm(buildDefaultGenerationForm(settings?.default_model ?? defaultGenerationForm.model))
          }
          onSelectRun={setSelectedBatchRunId}
          onSubmit={() => void handleBatchLaunch()}
          onToggleSourceDataset={handleToggleGenerationSourceDataset}
          runs={batchRuns}
          runsError={batchRunsError}
          selectedSourceDatasetIds={selectedGenerationSourceDatasetIds}
          selectedRunDetail={selectedBatchRunDetail}
          selectedRunId={selectedBatchRunId}
          streamEvents={batchStreamEvents}
          streamStatus={batchStreamStatus}
          submitting={generationSubmitting}
        />
      ) : null}
      {activeView === "documents" ? (
        <DocumentsView
          documentActionPending={documentActionPending}
          documentChunksExpanded={documentChunksExpanded}
          uploadAdvancedOpen={documentUploadAdvancedOpen}
          uploadChunkOverlap={documentUploadChunkOverlap}
          uploadChunkSize={documentUploadChunkSize}
          documentDetailLoading={documentDetailLoading}
          documentFilter={documentFilter}
          documents={filteredDocuments}
          documentsError={documentsError}
          documentsLoading={documentsLoading}
          isDocumentDeleteConfirmOpen={isDocumentDeleteConfirmOpen}
          isDocumentViewOpen={isDocumentViewOpen}
          onCloseDeleteConfirm={() => setIsDocumentDeleteConfirmOpen(false)}
          onCloseView={() => setIsDocumentViewOpen(false)}
          onCopyChunkContent={(content) => void handleCopyChunkContent(content)}
          onCopyRef={() => void handleCopyDocumentRef()}
          onDelete={() => void handleDeleteDocument()}
          onDocumentFilterChange={setDocumentFilter}
          onOpenDeleteConfirm={() => setIsDocumentDeleteConfirmOpen(true)}
          onRefresh={handleRefreshDocuments}
          onScraperDatasetNameChange={setScraperDatasetName}
          onScraperSubmit={() => void handleScraperImport()}
          onScraperTextChange={setScraperText}
          onSelectDocument={setSelectedDocumentId}
          onUploadAdvancedToggle={() => setDocumentUploadAdvancedOpen((current) => !current)}
          onUploadChunkOverlapChange={setDocumentUploadChunkOverlap}
          onUploadChunkSizeChange={setDocumentUploadChunkSize}
          onUploadFileChange={setDocumentUploadFile}
          onUploadModeChange={setDocumentUploadMode}
          onUploadSubmit={() => void handleUploadDocument()}
          onToggleChunkExpansion={() => setDocumentChunksExpanded((current) => !current)}
          onView={() => setIsDocumentViewOpen(true)}
          scraperDatasetName={scraperDatasetName}
          scraperMessage={scraperMessage}
          scraperPending={scraperPending}
          scraperText={scraperText}
          selectedDocumentDetail={selectedDocumentDetail}
          selectedDocumentId={selectedDocumentId}
          uploadFileName={documentUploadFile?.name ?? ""}
          uploadMessage={documentUploadMessage}
          uploadMode={documentUploadMode}
          uploadPending={documentUploadPending}
        />
      ) : null}
      {activeView === "exports" ? (
        <ExportsView
          availableDatasets={exportPickerDatasets.map((dataset) => ({ id: dataset.id, name: dataset.name }))}
          exportCreatePending={exportCreatePending}
          exportActionPendingId={exportActionPendingId}
          exportDatasetIds={exportForm.datasetIds}
          exportsError={exportsError}
          exportsHistory={exportsHistory}
          exportsLoading={exportsLoading}
          exportFormat={exportForm.format}
          exportMaxExamples={exportForm.maxExamples}
          exportMessage={exportMessage}
          exportMinScore={exportForm.minScore}
          exportTrainValSplit={exportForm.trainValSplit}
          exportDedupePass={exportForm.dedupePass}
          exportShuffle={exportForm.shuffle}
          exportPickerQuery={exportPickerQuery}
          formatDate={formatDate}
          isExportPickerOpen={isExportPickerOpen}
          onExportFieldChange={handleExportFieldChange}
          onExportPickerQueryChange={setExportPickerQuery}
          onOpenExportPicker={() => setIsExportPickerOpen(true)}
          onCloseExportPicker={() => setIsExportPickerOpen(false)}
          onExportSubmit={() => void handleCreateExport()}
          onToggleExportDataset={handleToggleExportDataset}
          onClearExportDatasets={handleClearExportDatasets}
          onDownload={handleDownloadExport}
          onRerun={(exportId) => void handleRerunExport(exportId)}
          selectedExportDatasetIds={selectedExportDatasetIds}
        />
      ) : null}
      {activeView === "settings" ? (
        <SettingsView
          error={settingsError}
          initialSettings={initialSettings}
          loading={settingsLoading}
          message={settingsMessage}
          onFieldChange={handleSettingsFieldChange}
          onOpenModelPicker={handleOpenModelPicker}
          onReset={handleResetSettings}
          onSave={() => void handleSaveSettings()}
          saving={settingsSaving}
          settings={settings}
          validationError={settingsValidationError}
        />
      ) : null}
      <ModelPickerModal
        isOpen={modelPickerTarget !== null}
        onClose={handleCloseModelPicker}
        onQueryChange={setModelPickerQuery}
        onSelect={handleSelectModel}
        options={filteredModelOptions}
        query={modelPickerQuery}
        targetLabel={
          modelPickerTarget === "generation"
            ? "generation"
            : modelPickerTarget === "default_model"
              ? "default model"
              : modelPickerTarget === "grading_model"
                ? "grading model"
                : modelPickerTarget === "naming_model"
                  ? "naming model"
                  : "model selection"
        }
      />
      {toast ? (
        <div className={`toast toast-${toast.tone}`} aria-live="polite" role="status">
          {toast.message}
        </div>
      ) : null}
    </AppShell>
  );
}

export default App;
