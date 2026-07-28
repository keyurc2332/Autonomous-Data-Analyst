import { useEffect, useState } from "react";

/**
 * Shown while an analysis runs.
 *
 * The request is synchronous, so the client cannot know which stage is
 * executing. Rather than invent a progress bar, this shows the real pipeline,
 * a live elapsed count, and an honest expectation. Fake progress is worse than
 * none: it teaches the reader that the interface guesses.
 */
const STAGES = [
  ["Clean", "duplicates, placeholders, whitespace"],
  ["Profile", "types, nulls, outliers, correlations"],
  ["Plan", "choose a target and task"],
  ["Train", "fit and rank candidate models"],
  ["Explain", "measure what drove the predictions"],
  ["Reflect", "judge the result, retry if worthwhile"],
  ["Report", "write it up"],
] as const;

export function RunProgress() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rise">
      <div className="mb-3.5 flex items-baseline justify-between">
        <span className="text-[13px] font-medium text-ink">Running</span>
        <span className="tabular text-[12px] text-ink-faint">{seconds}s</span>
      </div>

      <ol className="space-y-2">
        {STAGES.map(([name, detail], i) => (
          <li key={name} className="flex items-baseline gap-2.5">
            <span
              aria-hidden
              className="sweep mt-[6px] h-[6px] w-[6px] shrink-0 rounded-full"
              style={{ animationDelay: `${i * 120}ms` }}
            />
            <span className="text-[12px] font-medium text-ink-soft">{name}</span>
            <span className="text-[11px] leading-snug text-ink-faint">{detail}</span>
          </li>
        ))}
      </ol>

      <p className="mt-3.5 border-t border-rule pt-3 text-[12px] leading-snug text-ink-faint">
        Usually 10–30 seconds. Larger tables are sampled to keep it interactive.
        {seconds > 45 && (
          <span className="block pt-1 text-weak">
            Taking longer than usual — a big table can hold the server while it
            trains.
          </span>
        )}
      </p>
    </div>
  );
}
