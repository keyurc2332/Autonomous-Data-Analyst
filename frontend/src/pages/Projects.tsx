import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LLMCheck, type ProjectSummary } from "../api";
import { SkeletonCard } from "../components/Skeleton";
import { Button, ErrorNote } from "../components/shell";

const VERDICT = {
  strong: { label: "Strong", cls: "bg-verified-wash text-verified border-verified/30" },
  acceptable: { label: "Acceptable", cls: "bg-verified-wash text-verified border-verified/30" },
  weak: { label: "Weak", cls: "bg-weak-wash text-weak border-weak/30" },
} as const;

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div>
      <div className="tabular text-[22px] font-semibold leading-none text-ink">{value}</div>
      <div className="mt-1 text-[11px] uppercase tracking-wider text-ink-faint">{label}</div>
    </div>
  );
}

function ProjectCard({ project }: { project: ProjectSummary }) {
  const verdict = project.last_verdict ? VERDICT[project.last_verdict] : null;
  const flags: string[] = [];
  if (project.leaked_count) flags.push(`${project.leaked_count} leaked column removed`);
  if (project.derived_target) flags.push("target derived from its own columns");

  return (
    <Link
      to={`/projects/${project.id}`}
      className="rise group flex flex-col rounded-md border border-rule bg-surface p-4 transition-all duration-200 hover:-translate-y-[2px] hover:border-rule-strong hover:shadow-[0_2px_12px_rgba(20,33,58,0.06)]"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-[15px] font-semibold leading-tight">{project.name}</h3>
        {verdict && (
          <span
            className={`shrink-0 rounded border px-1.5 py-[2px] text-[10px] font-semibold uppercase tracking-wider ${verdict.cls}`}
          >
            {verdict.label}
          </span>
        )}
      </div>

      {project.target_column ? (
        <p className="tabular mt-1 text-[12px] text-ink-faint">
          predicting {project.target_column} · {project.task_type}
        </p>
      ) : (
        <p className="mt-1 text-[12px] text-ink-faint">
          {project.dataset_count ? "not analysed yet" : "no table uploaded"}
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-baseline gap-x-5 gap-y-2 border-t border-rule pt-3">
        {project.row_count != null && (
          <Stat value={project.row_count.toLocaleString()} label="rows" />
        )}
        {project.last_value != null && (
          <Stat value={project.last_value.toFixed(3)} label={project.last_metric ?? "score"} />
        )}
        <Stat value={project.run_count} label={project.run_count === 1 ? "run" : "runs"} />
      </div>

      {flags.length > 0 && (
        <ul className="mt-3 space-y-1">
          {flags.map((f) => (
            <li key={f} className="flex gap-1.5 text-[11px] leading-snug text-weak">
              <span aria-hidden className="mt-[5px] h-[5px] w-[5px] shrink-0 rounded-full bg-weak" />
              {f}
            </li>
          ))}
        </ul>
      )}
    </Link>
  );
}

export function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [llm, setLlm] = useState<LLMCheck | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => {
        // An unreachable API previously rendered as "No projects yet", which
        // reads as data loss. It is not the same thing and must not look it.
        setLoadFailed(true);
        setProjects([]);
        setError(e.message);
      });
    api.llmCheck().then(setLlm).catch(() => setLlm(null));
  }, []);

  async function create() {
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const created = await api.createProject(name.trim());
      setProjects((p) => [
        {
          ...created,
          dataset_count: 0, run_count: 0, row_count: null,
          last_run_id: null, last_verdict: null, last_metric: null,
          last_value: null, last_run_at: null,
          leaked_count: 0, derived_target: false,
        },
        ...(p ?? []),
      ]);
      setName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreating(false);
    }
  }

  const analysed = (projects ?? []).filter((p) => p.last_verdict);
  const totalRuns = (projects ?? []).reduce((n, p) => n + p.run_count, 0);
  const totalRows = (projects ?? []).reduce((n, p) => n + (p.row_count ?? 0), 0);
  const caught =
    (projects ?? []).filter((p) => p.leaked_count || p.derived_target).length;

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <header className="rise max-w-2xl">
        <h1 className="text-[30px] font-bold leading-[1.15] tracking-tight">
          Autonomous Data Analyst
        </h1>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-soft">
          Upload a table. It cleans the data, decides what to predict, trains and
          ranks models, measures what actually drove the predictions — and when
          the data can't support the question, it says so instead of returning a
          confident number.
        </p>
        {llm && (
          <p className="tabular mt-4 text-[12px] text-ink-faint">
            <span
              className={`mr-1.5 inline-block h-[7px] w-[7px] rounded-full align-middle ${
                llm.ok ? "bg-verified" : "bg-failed"
              }`}
            />
            {llm.provider} · {llm.model}
            {llm.ok ? "" : ` · unreachable (${llm.error_type})`}
          </p>
        )}
      </header>

      {analysed.length > 0 && (
        <div className="rise rise-1 mt-9 flex flex-wrap gap-x-12 gap-y-5 border-y border-rule py-5">
          <Stat value={projects!.length} label="projects" />
          <Stat value={totalRuns} label="analyses run" />
          <Stat value={totalRows.toLocaleString()} label="rows examined" />
          <Stat value={caught} label="datasets with leakage caught" />
        </div>
      )}

      <div className="rise rise-2 mt-9 flex flex-wrap items-center gap-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && create()}
          placeholder="New project — e.g. customer churn"
          className="min-w-[240px] flex-1 rounded border border-rule-strong bg-surface px-3 py-2 text-[14px] placeholder:text-ink-faint"
        />
        <Button onClick={create} disabled={creating || !name.trim()}>
          Create project
        </Button>
      </div>
      {error && <div className="mt-3"><ErrorNote>{error}</ErrorNote></div>}

      <div className="mt-6">
        {projects === null ? (
          <div className="grid gap-4 sm:grid-cols-2">
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : loadFailed ? (
          <ErrorNote>
            Could not reach the API, so your projects could not be listed. They
            have not been deleted. If a large analysis is still running it may be
            holding the server.
          </ErrorNote>
        ) : projects.length === 0 ? (
          <div className="rounded-md border border-dashed border-rule-strong px-6 py-12 text-center">
            <p className="text-[15px] text-ink">Nothing here yet.</p>
            <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-ink-soft">
              Create a project, upload a CSV, and it will tell you what's
              predictable in it — or why nothing is. There's a sample table in{" "}
              <span className="tabular">sample_data/</span> to start with.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            {projects.map((p) => (
              <ProjectCard key={p.id} project={p} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
