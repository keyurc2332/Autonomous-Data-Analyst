import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type AnalysisRun } from "../api";
import { ImportanceChart } from "../components/ImportanceChart";
import { RunTrace } from "../components/RunTrace";
import { QualityChecks, VerdictBadge } from "../components/Verdict";
import { ErrorNote, Metric, Panel } from "../components/shell";

const CLEANING_TITLES: Record<string, string> = {
  drop_duplicate_rows: "Removed duplicate rows",
  normalise_missing: "Converted placeholder text to nulls",
  trim_whitespace: "Trimmed whitespace",
  drop_empty_columns: "Removed empty columns",
};

const ACTION_COPY: Record<string, string> = {
  retry: "Tried again with changes",
  accept: "Stopped here",
  abandon: "Gave up on this question",
};

export function RunDetail() {
  const { runId = "" } = useParams();
  const [run, setRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getRun(runId).then(setRun).catch((e) => setError(e.message));
  }, [runId]);

  if (error) return <div className="mx-auto max-w-3xl p-6"><ErrorNote>{error}</ErrorNote></div>;
  if (!run) return <div className="mx-auto max-w-3xl p-6 text-ink-faint">Loading…</div>;

  const out = run.output_payload;
  const quality = out?.quality;
  const attempts = out?.attempts ?? [];

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <Link
        to={`/projects/${run.project_id}`}
        className="tabular text-[12px] text-ink-faint hover:text-ink"
      >
        ← back to project
      </Link>

      <div className="mt-2 flex flex-wrap items-center gap-3">
        <h1 className="text-[22px] font-bold tracking-tight">
          {out?.plan?.target_column ?? "Analysis"}
        </h1>
        {quality && <VerdictBadge verdict={quality.verdict} />}
        {out?.plan && (
          <span className="tabular text-[12px] text-ink-faint">{out.plan.task_type}</span>
        )}
        {run.status === "succeeded" && (
          <a
            href={api.reportUrl(run.id)}
            className="ml-auto rounded border border-rule-strong px-3 py-1.5 text-[13px] font-medium text-ink-soft hover:bg-paper"
          >
            Download report
          </a>
        )}
      </div>

      {run.status === "failed" && (
        <div className="mt-4"><ErrorNote>{run.error ?? "The run did not complete."}</ErrorNote></div>
      )}

      {!!out?.training?.leaked_features?.length && (
        <div className="mt-5 rounded-md border border-weak/40 bg-weak-wash px-4 py-3">
          <p className="text-[13px] font-semibold text-weak">
            Columns removed for restating the target
          </p>
          <ul className="mt-1.5 space-y-1">
            {out.training.leaked_features.map((f) => (
              <li key={f.column} className="text-[13px] leading-snug text-ink-soft">
                <span className="tabular font-medium">{f.column}</span> — {f.reason}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[12px] leading-snug text-ink-soft">
            Scores below come from a model trained without them. Left in, they
            would have produced a perfect score and learned nothing.
          </p>
        </div>
      )}

      {out?.training?.additive_leakage && (
        <div className="mt-5 rounded-md border border-failed/40 bg-failed-wash px-4 py-3">
          <p className="text-[13px] font-semibold text-failed">
            The target is derived from its own columns
          </p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-soft">
            {out.training.additive_leakage.reason}
          </p>
          <p className="tabular mt-2 text-[12px] text-ink-soft">
            {out.training.additive_leakage.contributors
              .map((c) => `${c.column} ×${c.coefficient.toFixed(3)}`)
              .join("  ·  ")}
          </p>
          <p className="mt-2 text-[12px] leading-snug text-ink-faint">
            Coefficients near 1.0 mean the target is a sum. The score below is
            real but meaningless — remove the component columns, or predict
            something not derived from them.
          </p>
        </div>
      )}

      {out?.training?.sampled_from && (
        <p className="tabular mt-4 text-[12px] text-ink-faint">
          Sampled {out.training.n_train + out.training.n_test} rows from{" "}
          {out.training.sampled_from.toLocaleString()} to keep the run interactive.
        </p>
      )}

      {out?.summary && (
        <p className="mt-5 max-w-3xl border-l-2 border-verified pl-4 text-[15px] leading-relaxed text-ink">
          {out.summary}
        </p>
      )}

      <div className="mt-7 grid gap-5 lg:grid-cols-[250px_1fr]">
        <div className="space-y-5">
          <Panel title="How it ran">
            <RunTrace steps={out?.steps ?? []} />
          </Panel>

          {out?.cleaning?.changed && (
            <Panel
              title="Cleaned first"
              aside={
                <span className="tabular text-[12px] text-ink-faint">
                  {out.cleaning.rows_before.toLocaleString()} →{" "}
                  {out.cleaning.rows_after.toLocaleString()} rows
                </span>
              }
            >
              <ul className="space-y-2.5">
                {out.cleaning.actions.map((a) => (
                  <li key={a.action}>
                    <div className="text-[13px] font-medium">
                      {CLEANING_TITLES[a.action] ?? a.action}
                      {a.rows_affected > 0 && (
                        <span className="tabular ml-1.5 text-ink-faint">
                          {a.rows_affected.toLocaleString()}
                        </span>
                      )}
                    </div>
                    <p className="text-[12px] leading-snug text-ink-faint">{a.detail}</p>
                    {a.columns.length > 0 && (
                      <p className="tabular text-[12px] text-ink-soft">
                        {a.columns.join(", ")}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {out?.training && (
            <Panel title="Data used">
              <div className="grid grid-cols-2 gap-4">
                <Metric label="Train rows" value={out.training.n_train.toLocaleString()} />
                <Metric label="Test rows" value={out.training.n_test.toLocaleString()} />
                <Metric label="Features" value={out.training.features_used.length} />
                <Metric label="Dropped" value={out.training.features_dropped.length} />
              </div>
              {out.training.features_dropped.length > 0 && (
                <ul className="mt-3 space-y-1 border-t border-rule pt-3">
                  {out.training.features_dropped.map((d) => (
                    <li key={d.column} className="text-[12px] leading-snug text-ink-faint">
                      <span className="tabular text-ink-soft">{d.column}</span> — {d.reason}
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          )}
        </div>

        <div className="space-y-5">
          {out?.explanation && (
            <Panel title="What drove the predictions">
              <ImportanceChart explanation={out.explanation} />
            </Panel>
          )}

          {quality && (
            <Panel
              title="Quality checks"
              aside={
                <span className="tabular text-[12px] text-ink-faint">
                  {quality.gate_metric} {quality.gate_value.toFixed(4)}
                </span>
              }
            >
              <QualityChecks quality={quality} />
              {quality.dead_features.length > 0 && (
                <p className="mt-3 border-t border-rule pt-3 text-[12px] leading-snug text-ink-faint">
                  Contributing almost nothing:{" "}
                  <span className="tabular">{quality.dead_features.join(", ")}</span>
                </p>
              )}
            </Panel>
          )}

          {out?.reflection && (
            <Panel
              title="Decision"
              aside={
                <span className="tabular text-[12px] text-ink-faint">
                  {ACTION_COPY[out.reflection.action] ?? out.reflection.action}
                </span>
              }
            >
              <p className="text-[14px] leading-relaxed text-ink-soft">
                {out.reflection.reasoning}
              </p>
            </Panel>
          )}

          {attempts.length > 1 && (
            <Panel
              title="Attempts"
              aside={
                <span className="text-[11px] text-ink-faint">
                  retries are judged on the gate metric
                </span>
              }
            >
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="text-left text-[11px] uppercase tracking-wider text-ink-faint">
                    <th className="pb-1.5 font-medium">Round</th>
                    <th className="pb-1.5 font-medium">Model</th>
                    <th className="pb-1.5 font-medium">Excluded</th>
                    <th className="pb-1.5 text-right font-medium">Decided on</th>
                    <th className="pb-1.5 text-right font-medium">Headline</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {attempts.map((a) => (
                    <tr key={a.round} className="border-t border-rule">
                      <td className="py-1.5">{a.round}</td>
                      <td className="py-1.5">{a.best_model}</td>
                      <td className="py-1.5 text-ink-faint">
                        {a.excluded_features.length ? a.excluded_features.join(", ") : "—"}
                      </td>
                      <td className="py-1.5 text-right">
                        {a.gate_value !== undefined ? (
                          <>
                            <span className="text-ink-faint">{a.gate_metric} </span>
                            {a.gate_value.toFixed(4)}
                          </>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="py-1.5 text-right text-ink-faint">
                        {a.primary_metric} {a.primary_metric_value.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}

          {run.experiments.length > 0 && (
            <Panel title="Models compared">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="text-left text-[11px] uppercase tracking-wider text-ink-faint">
                    <th className="pb-1.5 font-medium">Model</th>
                    <th className="pb-1.5 text-right font-medium">Metric</th>
                    <th className="pb-1.5 text-right font-medium">Seconds</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {run.experiments.map((e) => (
                    <tr key={e.model_name} className="border-t border-rule">
                      <td className="py-1.5">
                        {e.model_name}
                        {e.is_selected && (
                          <span className="ml-2 rounded bg-verified-wash px-1.5 py-px text-[10px] font-semibold uppercase tracking-wider text-verified">
                            chosen
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 text-right">
                        {e.primary_metric} {e.primary_metric_value?.toFixed(4)}
                      </td>
                      <td className="py-1.5 text-right text-ink-faint">
                        {e.train_seconds?.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}

          {out?.plan?.rationale && (
            <Panel title="Why this target">
              <p className="text-[14px] leading-relaxed text-ink-soft">{out.plan.rationale}</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
