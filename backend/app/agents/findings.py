"""The single registry of things the report must mention.

Three separate bugs shared one shape: a signal was added to training or
quality, wired carefully into the database, the UI and the PDF, and never
given to the summary node. Missingness indicators, then additive leakage, then
removed leaked columns. Each was invisible until someone read the output and
noticed the prose disagreed with the panel beside it.

The cause was that `summary_node` hand-assembled its facts. Adding a finding
meant remembering to edit a function three files away, and remembering is not
a mechanism.

Now there is one list. `summary_node` iterates it, and `test_findings.py`
asserts that every finding-shaped field on the training outcome and the
quality report appears here. Adding a signal without registering it fails the
suite rather than silently producing a report that omits it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Finding:
    """One thing that, when present, the report must state."""

    name: str
    # Where the value lives: ("training", "additive_leakage") reads
    # state["training"]["additive_leakage"].
    path: tuple[str, ...]
    # Critical findings outrank the metrics and must lead the summary.
    critical: bool
    render: Callable[[Any], str]


def _render_additive(value: dict[str, Any]) -> str:
    return (
        "CRITICAL: " + value["reason"]
        + " The scores are arithmetic, not prediction. Lead with this, state "
        "plainly that the result is meaningless, and say which columns must go."
    )


def _render_leaked(value: list[dict[str, Any]]) -> str:
    return (
        "Columns removed before training for restating the target: "
        + ", ".join(f["column"] for f in value)
        + ". Every score comes from a model trained without them."
    )


def _render_sampled(value: int) -> str:
    return (
        f"The table was sampled down from {value:,} rows to keep the run "
        "interactive. Mention this."
    )


def _render_dead_features(value: list[str]) -> str:
    return (
        "Measured as contributing almost nothing: " + ", ".join(value)
        + ". Do not describe these as drivers."
    )


def _render_failed_checks(value: list[str]) -> str:
    return "Quality checks that failed: " + "; ".join(value)


FINDINGS: tuple[Finding, ...] = (
    Finding("additive_leakage", ("training", "additive_leakage"), True, _render_additive),
    Finding("leaked_features", ("training", "leaked_features"), True, _render_leaked),
    Finding("sampled_from", ("training", "sampled_from"), False, _render_sampled),
    Finding("dead_features", ("quality", "dead_features"), False, _render_dead_features),
    Finding("failed_checks", ("quality", "reasons"), False, _render_failed_checks),
)


def _lookup(state: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = state
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def collect(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (critical facts, ordinary facts) for whatever is present."""
    critical: list[str] = []
    ordinary: list[str] = []
    for finding in FINDINGS:
        value = _lookup(state, finding.path)
        if not value:            # absent, empty list, empty dict, 0, None
            continue
        (critical if finding.critical else ordinary).append(finding.render(value))
    return critical, ordinary
