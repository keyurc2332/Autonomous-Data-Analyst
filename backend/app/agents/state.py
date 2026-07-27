"""LangGraph state.

Rule for this file: **no DataFrames, ever.** LangGraph serialises state on
every node transition. Putting a dataset in state means re-serialising the
whole thing repeatedly, and it makes checkpoints unusable. State carries
identifiers and paths; nodes load what they need and discard it.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class AnalysisPlan(TypedDict):
    target_column: str
    task_type: Literal["classification", "regression"]
    rationale: str
    excluded_columns: list[str]
    data_quality_concerns: list[str]


class AnalysisState(TypedDict, total=False):
    # --- inputs, set once ---
    run_id: str
    project_id: str
    dataset_id: str
    storage_path: str          # relative to STORAGE_DIR
    profile: dict[str, Any]    # deterministic, from Phase 2
    user_goal: str | None

    # --- produced by nodes ---
    # Path to the cleaned copy, relative to STORAGE_DIR. Training reads this
    # in preference to the raw upload, so the original is never modified.
    cleaned_path: str | None
    cleaning: dict[str, Any] | None

    plan: AnalysisPlan | None
    training: dict[str, Any] | None
    # Path to the best fitted pipeline, relative to STORAGE_DIR. A path, not
    # the model itself -- see the rule at the top of this file.
    model_path: str | None
    explanation: dict[str, Any] | None
    quality: dict[str, Any] | None
    reflection: dict[str, Any] | None
    summary: str | None

    # --- reflection loop control ---
    round: int                      # 0 for the first attempt
    excluded_features: list[str]    # accumulated across rounds
    attempts: list[dict[str, Any]]  # one record per training round
    best_score: float | None

    # --- control ---
    error: str | None
    failed_node: str | None
    steps: list[str]
