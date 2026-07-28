import type { ReactNode } from "react";

export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-md border border-rule bg-surface transition-colors duration-200 ${className}`}
    >
      {title && (
        <header className="flex items-baseline justify-between gap-3 border-b border-rule px-4 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
            {title}
          </h2>
          {aside}
        </header>
      )}
      <div className="px-4 py-3.5">{children}</div>
    </section>
  );
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-ink-faint">{label}</div>
      <div className="tabular text-[15px] font-medium text-ink">{value}</div>
    </div>
  );
}

export function Empty({ title, action }: { title: string; action?: ReactNode }) {
  return (
    <div className="rounded-md border border-dashed border-rule-strong px-5 py-9 text-center">
      <p className="text-[14px] text-ink-soft">{title}</p>
      {action && <div className="mt-3.5">{action}</div>}
    </div>
  );
}

export function Button({
  children,
  variant = "primary",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "quiet" }) {
  const base =
    "inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-[13px] font-medium " +
    "transition-all duration-150 active:scale-[0.97] " +
    "disabled:opacity-45 disabled:cursor-not-allowed disabled:active:scale-100";
  const styles =
    variant === "primary"
      ? "bg-verified text-white hover:bg-[#0b5f58]"
      : "border border-rule-strong text-ink-soft hover:bg-paper";
  return (
    <button className={`${base} ${styles}`} {...rest}>
      {children}
    </button>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded border border-failed/30 bg-failed-wash px-3 py-2 text-[13px] text-failed">
      {children}
    </div>
  );
}
