import type { Quality } from "../api";

const TONE = {
  strong: { label: "Strong", cls: "bg-verified-wash text-verified border-verified/30" },
  acceptable: { label: "Acceptable", cls: "bg-verified-wash text-verified border-verified/30" },
  weak: { label: "Weak", cls: "bg-weak-wash text-weak border-weak/30" },
} as const;

export function VerdictBadge({ verdict }: { verdict: Quality["verdict"] }) {
  const t = TONE[verdict] ?? TONE.weak;
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-[3px] text-[11px] font-semibold uppercase tracking-wider ${t.cls}`}
    >
      {t.label}
    </span>
  );
}

/**
 * Checks are shown passed *and* failed, deliberately.
 *
 * The reflection agent kept inventing causes ("only 482 rows, too small")
 * for checks that had actually passed. Stating what is fine, not only what
 * is broken, is what stopped that -- and a reader deserves the same context.
 */
export function QualityChecks({ quality }: { quality: Quality }) {
  return (
    <ul className="space-y-2">
      {quality.checks.map((c) => {
        // Rose means the verdict was driven down. Amber means something was
        // found and dealt with -- a good outcome, not a failure.
        const tone = c.passed
          ? "bg-verified"
          : c.gating === false
            ? "bg-weak"
            : "bg-failed";
        const label = c.passed ? null : c.gating === false ? "found" : "failed";
        return (
        <li key={c.name} className="flex gap-2.5">
          <span aria-hidden className={`mt-[6px] h-[7px] w-[7px] shrink-0 rounded-full ${tone}`} />
          <div className="min-w-0">
            <span className="tabular text-[11px] uppercase tracking-wider text-ink-faint">
              {c.name.replace(/_/g, " ")}
              {label && (
                <span className={c.gating === false ? "ml-1.5 text-weak" : "ml-1.5 text-failed"}>
                  {label}
                </span>
              )}
            </span>
            <p className="text-[13px] leading-snug text-ink-soft">{c.detail}</p>
          </div>
        </li>
        );
      })}
    </ul>
  );
}
