/** Loading placeholders that match the shape of what's coming.
 *  A spinner says "wait"; a skeleton says "here is what you'll get". */
export function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-md border border-rule bg-surface p-4">
      <div className="h-4 w-1/2 rounded bg-rule" />
      <div className="mt-3 h-3 w-3/4 rounded bg-rule/70" />
      <div className="mt-5 flex gap-4">
        <div className="h-8 w-16 rounded bg-rule/60" />
        <div className="h-8 w-16 rounded bg-rule/60" />
        <div className="h-8 w-16 rounded bg-rule/60" />
      </div>
    </div>
  );
}

export function SkeletonLines({ rows = 3 }: { rows?: number }) {
  return (
    <div className="animate-pulse space-y-2.5">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-3 rounded bg-rule"
          style={{ width: `${90 - i * 12}%` }}
        />
      ))}
    </div>
  );
}
