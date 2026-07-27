"""The chat agent.

This is the one place in the system where the model genuinely *chooses* what
to do. Everywhere else the graph fixes the sequence; here the model decides
which tools to call, in what order, and when it has enough to answer.

The loop is bounded, every tool call is schema-validated, and tool errors are
returned to the model as readable text so it can correct itself rather than
failing the turn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.llm import get_llm, message_text
from app.core.logging import get_logger
from app.services.chat_tools import TOOL_SCHEMAS, execute, load

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 4
MAX_HISTORY_MESSAGES = 12

SYSTEM = """You answer questions about a dataset a user has uploaded.

You have tools that compute real values from the data. Use them. Never state a
number you have not obtained from a tool, and never estimate.

Rules:
- If you are unsure what the data contains, call DatasetOverview first.
- Answer in one or two short paragraphs of plain language. No headings.
- Quote the actual figures the tools returned, and say what they are of \
(counts, percentages, averages).
- If a tool returns an error, read it and try a different approach. The error \
usually lists the valid column names.
- If the data cannot answer the question, say so plainly and say what would be \
needed. Do not substitute a related answer for the one asked.
- Correlation is not causation. Do not describe an association as a cause.
- Keep to what was measured. Do not speculate about why the data looks as it \
does unless the user asks for interpretation, and label it clearly if you do.
"""


def _context_block(
    profile: dict[str, Any] | None, latest_run: dict[str, Any] | None
) -> str:
    """Static facts worth giving the model up front, cheaply."""
    lines: list[str] = []
    if profile:
        shape = profile.get("shape", {})
        lines.append(f"Table: {shape.get('rows')} rows, {shape.get('columns')} columns.")
        names = [c["name"] for c in profile.get("columns", [])]
        lines.append("Columns: " + ", ".join(names))
    if latest_run:
        plan = latest_run.get("plan") or {}
        quality = latest_run.get("quality") or {}
        explanation = latest_run.get("explanation") or {}
        lines.append(
            f"A model was trained to predict '{plan.get('target_column')}' "
            f"({plan.get('task_type')}). "
            f"{quality.get('gate_metric')}={quality.get('gate_value')}, "
            f"verdict {quality.get('verdict')}."
        )
        if explanation.get("features"):
            top = ", ".join(f["feature"] for f in explanation["features"][:5])
            lines.append(f"Strongest measured drivers: {top}.")
    return "\n".join(lines)


def _latest_analysis_payload(latest_run: dict[str, Any] | None) -> dict[str, Any]:
    if not latest_run:
        return {"error": "No analysis has been run on this dataset yet."}
    quality = latest_run.get("quality") or {}
    explanation = latest_run.get("explanation") or {}
    training = latest_run.get("training") or {}
    return {
        "target": training.get("target_column"),
        "task_type": training.get("task_type"),
        "best_model": training.get("best_model"),
        "verdict": quality.get("verdict"),
        "gate_metric": quality.get("gate_metric"),
        "gate_value": quality.get("gate_value"),
        "failed_checks": quality.get("reasons", []),
        "feature_importance": [
            {"feature": f["feature"], "importance": f["importance"]}
            for f in explanation.get("features", [])[:10]
        ],
        "importance_method": explanation.get("method"),
        "summary": latest_run.get("summary"),
    }


async def answer(
    question: str,
    dataset_path: Path,
    profile: dict[str, Any] | None = None,
    latest_run: dict[str, Any] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Answer one question. Returns (reply, tool calls made)."""
    import anyio

    df: pd.DataFrame = await anyio.to_thread.run_sync(load, dataset_path)

    messages: list[Any] = [SystemMessage(content=SYSTEM)]
    context = _context_block(profile, latest_run)
    if context:
        messages.append(SystemMessage(content=f"Context:\n{context}"))

    for role, content in (history or [])[-MAX_HISTORY_MESSAGES:]:
        messages.append(
            HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        )
    messages.append(HumanMessage(content=question))

    llm = get_llm().bind_tools(TOOL_SCHEMAS)
    used: list[dict[str, Any]] = []

    for round_no in range(MAX_TOOL_ROUNDS):
        response = await llm.ainvoke(messages)
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            return message_text(response), used

        messages.append(response)
        for call in calls:
            name, args = call["name"], call.get("args") or {}
            if name == "LatestAnalysis":
                result = _latest_analysis_payload(latest_run)
            else:
                result = await anyio.to_thread.run_sync(execute, df, name, args)
            used.append({"tool": name, "arguments": args, "round": round_no})
            messages.append(ToolMessage(
                content=str(result), tool_call_id=call.get("id", name)
            ))

    # Out of rounds: ask for an answer from what has been gathered rather than
    # returning nothing.
    messages.append(HumanMessage(
        content="Answer now using only the tool results above. Do not call more tools."
    ))
    final = await get_llm().ainvoke(messages)
    logger.info("Chat hit the tool-round limit", extra={"calls": len(used)})
    return message_text(final), used
