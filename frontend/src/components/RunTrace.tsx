/**
 * The run trace.
 *
 * This is the most characteristic artifact the system produces:
 *   planner:ok -> training:ok(r0) -> explain:shap -> reflect:retry(r1) -> ...
 *
 * The retry is a genuine loop in the graph, so it is drawn as a loop.
 * Flattening it into a straight list would misrepresent the execution.
 * Rounds are numbered because the order carries information the reader
 * needs, not for decoration.
 */

interface Step {
  raw: string;
  node: string;
  outcome: string;
  round: number;
}

const NODE_LABEL: Record<string, string> = {
  clean: "Clean",
  planner: "Plan",
  training: "Train",
  explain: "Explain",
  reflect: "Reflect",
  summary: "Report",
};

/**
 * Rounds are derived by walking the list, not read off each label.
 *
 * Only `training:ok(rN)` and `reflect:retry(rN)` carry a round marker;
 * `explain:*`, `reflect:limit` and `summary:ok` do not. Parsing each label in
 * isolation reported round 0 for every one of them, so a three-round run
 * looked like it had happened entirely in round 0.
 *
 * A training step opens a round. Everything after it belongs to that round
 * until the next one opens -- including the reflection that closed it.
 */
function parseAll(raws: string[]): Step[] {
  let current = 0;
  return raws.map((raw) => {
    const [node, rest = ""] = raw.split(":");
    const outcome = rest.replace(/\(r\d+\)/, "");
    const marker = rest.match(/\(r(\d+)\)/);
    if (node === "training" && marker) current = Number(marker[1]);
    return { raw, node, outcome, round: current };
  });
}

function tone(step: Step): { dot: string; text: string; ring: string } {
  const bad = ["failed", "unavailable", "skipped"];
  const notable = ["fallback", "corrected", "retry", "limit", "no_improvement", "abandon"];
  if (bad.includes(step.outcome)) {
    return { dot: "bg-failed", text: "text-failed", ring: "ring-failed/25" };
  }
  if (notable.includes(step.outcome)) {
    return { dot: "bg-weak", text: "text-weak", ring: "ring-weak/25" };
  }
  return { dot: "bg-verified", text: "text-verified", ring: "ring-verified/25" };
}

export function RunTrace({ steps }: { steps: string[] }) {
  if (!steps.length) {
    return <p className="text-sm text-ink-faint">No steps recorded.</p>;
  }

  const parsed = parseAll(steps);
  const maxRound = Math.max(...parsed.map((s) => s.round));

  return (
    <ol className="relative">
      {parsed.map((step, i) => {
        const t = tone(step);
        const indented = step.round > 0;
        const isLast = i === parsed.length - 1;
        const opensLoop = step.node === "reflect" && step.outcome === "retry";

        return (
          <li key={i} className="relative flex gap-3" style={{ paddingLeft: indented ? 28 : 0 }}>
            {/* connector */}
            {!isLast && (
              <span
                aria-hidden
                className="absolute top-5 w-px bg-rule-strong"
                style={{ left: (indented ? 28 : 0) + 7, bottom: -4 }}
              />
            )}
            {/* the loop bracket: retry returns to training */}
            {opensLoop && (
              <span
                aria-hidden
                className="absolute rounded-bl-md border-b border-l border-weak"
                style={{ left: 7, top: 14, width: 21, height: 34 }}
              />
            )}

            <span className="flex h-5 w-[15px] shrink-0 items-center justify-center">
              <span className={`h-[7px] w-[7px] rounded-full ring-4 ${t.dot} ${t.ring}`} />
            </span>

            <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 pb-5">
              <span className="text-[13px] font-semibold tracking-tight">
                {NODE_LABEL[step.node] ?? step.node}
              </span>
              <span className={`tabular text-[12px] ${t.text}`}>{step.outcome}</span>
              {maxRound > 0 && (
                <span className="tabular text-[11px] text-ink-faint">round {step.round}</span>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
