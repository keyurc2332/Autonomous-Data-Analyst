import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { Explanation } from "../api";

const METHOD_NOTE: Record<string, string> = {
  shap: "Mean absolute SHAP value. One-hot columns summed back to their source column.",
  permutation: "Drop in score when each column is shuffled, averaged over repeats.",
  unavailable: "Importances could not be computed for this model.",
};

export function ImportanceChart({ explanation }: { explanation: Explanation }) {
  if (explanation.method === "unavailable" || !explanation.features.length) {
    return (
      <p className="text-[13px] text-ink-faint">
        {explanation.note ?? METHOD_NOTE.unavailable}
      </p>
    );
  }

  // Missingness indicators make names much longer ("mass (was missing)").
  // Without more room the axis clipped them into "mass (wasmissing)".
  const data = explanation.features.slice(0, 10).map((f) => ({
    name: f.feature.length > 26 ? `${f.feature.slice(0, 25)}…` : f.feature,
    full: f.feature,
    value: f.importance,
    parts: f.encoded_parts,
  }));
  const max = Math.max(...data.map((d) => d.value));

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3">
        <span className="tabular rounded bg-verified-wash px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider text-verified">
          {explanation.method}
        </span>
        <span className="tabular text-[11px] text-ink-faint">
          {explanation.rows_explained} rows · {explanation.model_name}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={Math.max(180, data.length * 30)}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 48, top: 0, bottom: 0 }}>
          <XAxis type="number" hide domain={[0, max * 1.02]} />
          <YAxis
            type="category"
            dataKey="name"
            width={172}
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 11, fill: "#4a5773", fontFamily: "IBM Plex Mono" }}
            interval={0}
          />
          <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={13} isAnimationActive={false}>
            {data.map((d, i) => (
              // Contribution fades with rank: the eye should land on what mattered.
              <Cell key={i} fill="#0f766e" fillOpacity={Math.max(0.28, d.value / max)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="mt-2 text-[12px] leading-snug text-ink-faint">
        {METHOD_NOTE[explanation.method]}
      </p>
    </div>
  );
}
