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
      {quality.checks.map((c) => (
        <li key={c.name} className="flex gap-2.5">
          <span
            aria-hidden
            className={`mt-[6px] h-[7px] w-[7px] shrink-0 rounded-full ${
              c.passed ? "bg-verified" : "bg-failed"
            }`}
          />
          <div className="min-w-0">
            <span className="tabular text-[11px] uppercase tracking-wider text-ink-faint">
              {c.name.replace(/_/g, " ")}
            </span>
            <p className="text-[13px] leading-snug text-ink-soft">{c.detail}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
