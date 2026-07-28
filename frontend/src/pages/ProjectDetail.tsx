import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  api,
  type AnalysisRun,
  type DatasetSummary,
  type Profile,
  type Project,
} from "../api";
import { Chat } from "../components/Chat";
import { RunProgress } from "../components/RunProgress";
import { SkeletonCard, SkeletonLines } from "../components/Skeleton";
import { Button, Empty, ErrorNote, Metric, Panel } from "../components/shell";

const SEVERITY = {
  error: "text-failed",
  warning: "text-weak",
  info: "text-ink-faint",
} as const;

export function ProjectDetail() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);

  const [project, setProject] = useState<Project | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [goal, setGoal] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const dataset = datasets[0] ?? null;

  useEffect(() => {
    Promise.all([
      api.getProject(projectId),
      api.listDatasets(projectId),
      api.listRuns(projectId),
    ])
      .then(([p, d, r]) => {
        setProject(p);
        setDatasets(d);
        setRuns(r);
        if (d[0]) return api.getProfile(projectId, d[0].id).then(setProfile);
      })
      .catch((e) => setError(e.message));
  }, [projectId]);

  async function upload(file: File) {
    setBusy("Reading and profiling the table…");
    setError(null);
    try {
      const res = await api.uploadDataset(projectId, file);
      setDatasets((d) => [res.dataset, ...d.filter((x) => x.id !== res.dataset.id)]);
      setProfile(await api.getProfile(projectId, res.dataset.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function run() {
    if (!dataset) return;
    setRunning(true);
    setError(null);
    try {
      const created = await api.startAnalysis(projectId, dataset.id, goal || undefined);
      navigate(`/runs/${created.id}`);
    } catch (e) {
      setError((e as Error).message);
      setRunning(false);
    }
  }

  if (error && !project) return <div className="mx-auto max-w-3xl p-6"><ErrorNote>{error}</ErrorNote></div>;
  if (!project) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-8">
        <div className="mb-6 h-6 w-56 animate-pulse rounded bg-rule" />
        <div className="grid gap-5 lg:grid-cols-[1.15fr_1fr]">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <Link to="/" className="tabular text-[12px] text-ink-faint hover:text-ink">
        ← all projects
      </Link>
      <h1 className="mt-2 text-[22px] font-bold tracking-tight">{project.name}</h1>

      {error && <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>}

      <div className="mt-6 grid gap-5 lg:grid-cols-[1.15fr_1fr]">
        <div className="rise rise-1 space-y-5">
          <Panel
            title="Table"
            aside={
              dataset && (
                <button
                  onClick={() => fileInput.current?.click()}
                  className="text-[12px] text-ink-faint hover:text-ink"
                >
                  replace
                </button>
              )
            }
          >
            <input
              ref={fileInput}
              type="file"
              accept=".csv,.tsv,.txt"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
            {!dataset ? (
              <Empty
                title="No table yet. CSV or TSV, up to 100 MB."
                action={
                  <Button onClick={() => fileInput.current?.click()} disabled={!!busy}>
                    Choose a file
                  </Button>
                }
              />
            ) : (
              <div>
                <div className="tabular mb-3 truncate text-[13px]">
                  {dataset.original_filename}
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <Metric label="Rows" value={dataset.n_rows?.toLocaleString() ?? "—"} />
                  <Metric label="Columns" value={dataset.n_columns ?? "—"} />
                  <Metric
                    label="Size"
                    value={`${(dataset.size_bytes / 1024).toFixed(0)} KB`}
                  />
                </div>
              </div>
            )}
          </Panel>

          {dataset && !profile && (
            <Panel title="What the profiler found">
              <SkeletonLines rows={4} />
            </Panel>
          )}

          {profile && (
            <Panel title="What the profiler found">
              <div className="mb-4 grid grid-cols-2 gap-4">
                <Metric label="Duplicate rows" value={profile.duplicate_rows} />
                <Metric
                  label="Suggested target"
                  value={profile.target_candidates[0]?.column ?? "none"}
                />
              </div>
              {profile.warnings.length > 0 && (
                <ul className="space-y-1.5 border-t border-rule pt-3">
                  {profile.warnings.slice(0, 8).map((w, i) => (
                    <li key={i} className="flex gap-2 text-[13px] leading-snug">
                      <span className={`tabular shrink-0 text-[11px] ${SEVERITY[w.severity]}`}>
                        {w.severity}
                      </span>
                      <span className="text-ink-soft">
                        {w.column && <span className="tabular">{w.column}: </span>}
                        {w.message}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          )}
        </div>

        <div className="rise rise-2 space-y-5">
          <Panel title="Run an analysis">
            <label className="mb-1.5 block text-[13px] text-ink-soft">
              What do you want to know? Optional.
            </label>
            <input
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="work out who is going to leave"
              className="mb-3 w-full rounded border border-rule-strong px-2.5 py-1.5 text-[14px] placeholder:text-ink-faint"
            />
            {running ? (
              <RunProgress />
            ) : (
              <>
                <Button onClick={run} disabled={!dataset || !!busy}>
                  {busy ? "Working…" : "Run analysis"}
                </Button>
                {busy && (
                  <p className="mt-2.5 text-[12px] leading-snug text-ink-faint">{busy}</p>
                )}
              </>
            )}
          </Panel>

          <Chat projectId={projectId} hasDataset={!!dataset} />

          <Panel title={`Runs (${runs.length})`}>
            {runs.length === 0 ? (
              <p className="text-[13px] text-ink-faint">Nothing run yet.</p>
            ) : (
              <ul className="divide-y divide-rule">
                {runs.map((r) => {
                  const q = r.output_payload?.quality;
                  return (
                    <li key={r.id}>
                      <Link
                        to={`/runs/${r.id}`}
                        className="-mx-2 flex items-baseline justify-between gap-3 rounded px-2 py-2 hover:bg-paper"
                      >
                        <span className="tabular truncate text-[13px]">
                          {r.output_payload?.plan?.target_column ?? r.status}
                        </span>
                        <span
                          className={`tabular shrink-0 text-[12px] ${
                            r.status === "failed" ? "text-failed" : "text-ink-faint"
                          }`}
                        >
                          {q ? `${q.gate_metric} ${q.gate_value.toFixed(3)}` : r.status}
                        </span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
