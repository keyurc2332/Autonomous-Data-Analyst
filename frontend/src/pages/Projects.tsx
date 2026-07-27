import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LLMCheck, type Project } from "../api";
import { Button, Empty, ErrorNote, Panel } from "../components/shell";

export function Projects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [llm, setLlm] = useState<LLMCheck | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => {
        // An unreachable API previously rendered as "No projects yet", which
        // reads as data loss. It is not the same thing and must not look it.
        setLoadFailed(true);
        setError(e.message);
      });
    api.llmCheck().then(setLlm).catch(() => setLlm(null));
  }, []);

  async function create() {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(name.trim());
      setProjects((p) => [project, ...p]);
      setName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-[26px] font-bold leading-tight tracking-tight">
          Autonomous Data Analyst
        </h1>
        <p className="mt-1.5 max-w-lg text-[14px] leading-relaxed text-ink-soft">
          Upload a table. It profiles the data, picks a target, trains and ranks models,
          measures what drove the predictions, and tells you plainly when the data cannot
          answer the question.
        </p>
        {llm && (
          <p className="tabular mt-3 text-[12px] text-ink-faint">
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

      <Panel title="New project">
        <div className="flex gap-2">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && create()}
            placeholder="Customer churn"
            className="flex-1 rounded border border-rule-strong bg-surface px-2.5 py-1.5 text-[14px] placeholder:text-ink-faint"
          />
          <Button onClick={create} disabled={busy || !name.trim()}>
            Create project
          </Button>
        </div>
        {error && <div className="mt-3"><ErrorNote>{error}</ErrorNote></div>}
      </Panel>

      <div className="mt-6">
        {loadFailed ? (
          <ErrorNote>
            Could not reach the API, so your projects could not be listed. They
            have not been deleted. If a large analysis is still running it may
            be holding the server; check <span className="tabular">docker compose logs api</span>.
          </ErrorNote>
        ) : projects.length === 0 ? (
          <Empty title="No projects yet. Create one to upload your first table." />
        ) : (
          <ul className="divide-y divide-rule overflow-hidden rounded-md border border-rule bg-surface">
            {projects.map((p) => (
              <li key={p.id}>
                <Link
                  to={`/projects/${p.id}`}
                  className="flex items-baseline justify-between gap-4 px-4 py-3 hover:bg-paper"
                >
                  <div className="min-w-0">
                    <div className="truncate text-[14px] font-medium">{p.name}</div>
                    {p.target_column && (
                      <div className="tabular text-[12px] text-ink-faint">
                        target {p.target_column} · {p.task_type}
                      </div>
                    )}
                  </div>
                  <time className="tabular shrink-0 text-[12px] text-ink-faint">
                    {new Date(p.created_at).toLocaleDateString()}
                  </time>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
