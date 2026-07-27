"""Turning a full profile into something worth spending tokens on.

A raw profile for a 60-column dataset runs to tens of thousands of tokens.
Sending it whole is slow, expensive, and on a rate-limited free tier it is
the difference between working and hitting 429 on every call. This module
compresses it to the facts a planner actually needs.
"""
from __future__ import annotations

from typing import Any

MAX_COLUMNS_IN_PROMPT = 60
MAX_WARNINGS_IN_PROMPT = 15


def summarise_profile(profile: dict[str, Any]) -> str:
    """Render a compact, LLM-readable digest of a dataset profile."""
    shape = profile.get("shape", {})
    lines = [
        f"Rows: {shape.get('rows')}  Columns: {shape.get('columns')}",
        f"Fully duplicated rows: {profile.get('duplicate_rows', 0)}",
        "",
        "COLUMNS (name | type | null% | distinct):",
    ]

    columns = profile.get("columns", [])
    for col in columns[:MAX_COLUMNS_IN_PROMPT]:
        detail = ""
        stats = col.get("stats") or {}
        if "min" in stats and "max" in stats:
            detail = f" range={stats['min']}..{stats['max']}"
        elif col.get("top_values"):
            vals = ", ".join(str(v["value"]) for v in col["top_values"][:4])
            detail = f" values=[{vals}]"
        lines.append(
            f"  {col['name']} | {col['semantic_type']} | "
            f"{col['null_pct']:.1f}% | {col['unique_count']}{detail}"
        )
    if len(columns) > MAX_COLUMNS_IN_PROMPT:
        lines.append(f"  ... and {len(columns) - MAX_COLUMNS_IN_PROMPT} more columns")

    warnings = profile.get("warnings", [])
    if warnings:
        lines += ["", "DATA QUALITY FINDINGS:"]
        for w in warnings[:MAX_WARNINGS_IN_PROMPT]:
            scope = f"[{w['column']}] " if w.get("column") else ""
            lines.append(f"  ({w['severity']}) {scope}{w['message']}")

    candidates = profile.get("target_candidates", [])
    if candidates:
        lines += ["", "SUGGESTED TARGETS (heuristic, not authoritative):"]
        for c in candidates:
            lines.append(
                f"  {c['column']} -> {c['task_type']} "
                f"(confidence {c['confidence']}) -- {c['reason']}"
            )

    corr = profile.get("correlations", {}).get("pairs", [])
    if corr:
        lines += ["", "STRONGEST CORRELATIONS:"]
        for p in corr[:8]:
            lines.append(f"  {p['a']} ~ {p['b']}: r={p['r']}")

    return "\n".join(lines)


PLANNER_SYSTEM = """You are the planning agent in an automated data analysis \
pipeline. You are given a statistical profile of a dataset that was computed \
deterministically -- it is ground truth, not a guess.

Your job is to choose ONE target column and the task type, then justify it.

Rules:
- The target MUST be one of the column names listed in the profile. Never \
invent a column.
- Choose classification for categorical or binary targets, regression for \
continuous numeric ones.
- Never choose a column whose type is identifier, constant, empty, or text.
- The suggested targets are heuristics. Prefer them when they are sensible, \
but override them if the user's stated goal points elsewhere.
- Keep the rationale to two or three sentences, and ground it in specific \
numbers from the profile.
"""
