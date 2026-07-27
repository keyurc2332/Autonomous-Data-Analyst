"""Graph nodes.

Each node takes state and returns a partial update. Nodes never raise: a
failure sets `error` and `failed_node` so the graph can route to a clean
terminal state and the run row records what happened.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.prompting import PLANNER_SYSTEM, summarise_profile
from app.agents.state import AnalysisState
from app.core.llm import get_llm, message_text
from app.core.logging import get_logger
from app.services import artifacts, storage
from app.services.cleaning import clean_file
from app.services.explain import explain
from app.services.quality import assess, is_improvement
from app.services.training import TrainingError, prepare_split, train

logger = get_logger(__name__)


class PlanOutput(BaseModel):
    """Structured output schema for the planner."""

    target_column: str = Field(description="Exact column name to predict.")
    task_type: Literal["classification", "regression"]
    rationale: str = Field(description="Two or three sentences citing profile numbers.")
    excluded_columns: list[str] = Field(
        default_factory=list, description="Columns to exclude from features, with no explanation."
    )
    data_quality_concerns: list[str] = Field(
        default_factory=list, description="Issues that could undermine the result."
    )


def _fallback_plan(profile: dict[str, Any], reason: str) -> dict[str, Any] | None:
    """Use the deterministic top candidate when the LLM cannot be trusted.

    The heuristic shortlist already exists and is decent. Falling back to it
    beats failing the whole run because a model returned a bad column name.
    """
    candidates = profile.get("target_candidates") or []
    if not candidates:
        return None
    top = candidates[0]
    return {
        "target_column": top["column"],
        "task_type": top["task_type"],
        "rationale": f"Fell back to the highest-confidence heuristic target ({reason}).",
        "excluded_columns": [],
        "data_quality_concerns": [f"Plan was not produced by the planner: {reason}"],
    }


async def planner_node(state: AnalysisState) -> dict[str, Any]:
    profile = state["profile"]
    valid_columns = {c["name"] for c in profile.get("columns", [])}

    prompt = summarise_profile(profile)
    if state.get("user_goal"):
        prompt += f"\n\nUSER GOAL: {state['user_goal']}"

    try:
        llm = get_llm().with_structured_output(PlanOutput)
        result: PlanOutput = await llm.ainvoke(
            [("system", PLANNER_SYSTEM), ("human", prompt)]
        )
        plan = result.model_dump()
    except Exception as exc:
        logger.warning("Planner LLM call failed", extra={"error": str(exc)})
        fallback = _fallback_plan(profile, f"{type(exc).__name__}")
        if fallback is None:
            return {
                "error": f"Planner failed and no fallback target exists: {exc}",
                "failed_node": "planner",
                "steps": [*state.get("steps", []), "planner:failed"],
            }
        return {"plan": fallback, "steps": [*state.get("steps", []), "planner:fallback"]}

    # Never trust a model-supplied column name. This is the guard that stops
    # a hallucinated target from reaching sklearn as a KeyError.
    if plan["target_column"] not in valid_columns:
        logger.warning(
            "Planner hallucinated a column", extra={"proposed": plan["target_column"]}
        )
        fallback = _fallback_plan(
            profile, f"proposed unknown column '{plan['target_column']}'"
        )
        if fallback is None:
            return {
                "error": f"Planner chose '{plan['target_column']}', which does not exist.",
                "failed_node": "planner",
                "steps": [*state.get("steps", []), "planner:failed"],
            }
        return {"plan": fallback, "steps": [*state.get("steps", []), "planner:corrected"]}

    logger.info("Plan ready", extra={"target": plan["target_column"], "task": plan["task_type"]})
    return {"plan": plan, "steps": [*state.get("steps", []), "planner:ok"]}


async def cleaning_node(state: AnalysisState) -> dict[str, Any]:
    """Apply deterministic cleaning and write a cleaned copy.

    Runs before planning so the planner sees the state of the data that will
    actually be modelled. The original upload is never modified.
    """
    import anyio

    from app.core.config import settings

    source = storage.resolve(state["storage_path"])
    relative = f"cleaned/{state['run_id']}.csv"
    destination = settings.STORAGE_DIR / relative

    try:
        report = await anyio.to_thread.run_sync(clean_file, source, destination)
    except Exception as exc:
        # Cleaning is an improvement, not a prerequisite. Fall back to the raw
        # file rather than failing a run over it.
        logger.warning("Cleaning failed", extra={"error": str(exc)})
        return {
            "cleaned_path": None,
            "cleaning": None,
            "steps": [*state.get("steps", []), "clean:skipped"],
        }

    return {
        "cleaned_path": relative,
        "cleaning": report.to_dict(),
        "steps": [
            *state.get("steps", []),
            f"clean:ok({len(report.actions)})" if report.changed else "clean:ok(0)",
        ],
    }


async def training_node(state: AnalysisState) -> dict[str, Any]:
    import anyio

    plan = state.get("plan")
    if not plan:
        return {"error": "No plan to execute.", "failed_node": "training"}

    path: Path = storage.resolve(state.get("cleaned_path") or state["storage_path"])
    try:
        outcome = await anyio.to_thread.run_sync(
            train, path, state["profile"], plan["target_column"], plan["task_type"],
            state.get("excluded_features") or [],
        )
    except TrainingError as exc:
        return {
            "error": str(exc),
            "failed_node": "training",
            "steps": [*state.get("steps", []), "training:failed"],
        }
    except Exception as exc:
        logger.exception("Unexpected training failure")
        return {
            "error": f"Training failed unexpectedly: {type(exc).__name__}",
            "failed_node": "training",
            "steps": [*state.get("steps", []), "training:failed"],
        }

    model_path = None
    try:
        best_pipeline = outcome.fitted_pipelines[outcome.best_model]
        model_path = await anyio.to_thread.run_sync(
            artifacts.save_model, best_pipeline, state["run_id"], outcome.best_model
        )
    except Exception as exc:
        # A missing artifact costs us the explanation, not the run.
        logger.warning("Could not persist model", extra={"error": str(exc)})

    round_no = state.get("round", 0)
    best = max(outcome.experiments, key=lambda e: e.primary_metric_value)
    attempt = {
        "round": round_no,
        "target_column": plan["target_column"],
        "task_type": plan["task_type"],
        "excluded_features": list(state.get("excluded_features") or []),
        "best_model": outcome.best_model,
        "primary_metric": best.primary_metric,
        "primary_metric_value": best.primary_metric_value,
    }

    return {
        "training": outcome.to_dict(),
        "model_path": model_path,
        "attempts": [*state.get("attempts", []), attempt],
        "steps": [*state.get("steps", []), f"training:ok(r{round_no})"],
    }


async def explain_node(state: AnalysisState) -> dict[str, Any]:
    """Compute feature importances for the winning model.

    Deterministic and LLM-free. The split is reconstructed rather than carried
    through state: `prepare_split` is seeded, so it returns the identical test
    set the model was scored on.
    """
    import anyio

    training, plan = state.get("training"), state.get("plan")
    model_path = state.get("model_path")
    if not training or not plan or not model_path:
        return {
            "explanation": None,
            "steps": [*state.get("steps", []), "explain:skipped"],
        }

    def _work() -> dict[str, Any]:
        pipeline = artifacts.load_model(model_path)
        split = prepare_split(
            storage.resolve(state.get("cleaned_path") or state["storage_path"]),
            state["profile"],
            plan["target_column"],
            plan["task_type"],
        )
        result = explain(
            pipeline, split.X_test, split.y_test, training["best_model"],
            split.numeric, split.categorical,
        )
        return result.to_dict()

    try:
        explanation = await anyio.to_thread.run_sync(_work)
    except Exception as exc:
        logger.warning("Explainability failed", extra={"error": str(exc)})
        return {
            "explanation": None,
            "steps": [*state.get("steps", []), "explain:failed"],
        }

    return {
        "explanation": explanation,
        "steps": [*state.get("steps", []), f"explain:{explanation['method']}"],
    }


SUMMARY_SYSTEM = """You are the reporting agent. Write a short, factual summary \
of a completed modelling run for a non-technical stakeholder.

Rules:
- Three to five sentences. No headings, no bullet points.
- Quote the actual metric values you are given. Never invent numbers.
- State plainly if the result is weak. A model that barely beats chance must \
be described as such, not dressed up.
- If feature importances are provided, name the columns that actually drove \
the predictions. These are measured, not guessed. Quote feature names exactly \
as given, including any qualifier in brackets; never shorten them.
- Do NOT speculate about WHY a result is weak, and do not claim any column \
harmed or helped performance unless the importance figures show it. A column \
having missing values does not mean it damaged the model. State what was \
measured; leave causes to the reader.
- Mention a data-quality caveat only as an observation, never as an explanation.
- If more than one attempt was made, say so in one clause: what changed and \
whether it helped. Do not dramatise it.
"""


async def summary_node(state: AnalysisState) -> dict[str, Any]:
    training = state.get("training")
    plan = state.get("plan")
    if not training or not plan:
        return {"summary": None}

    best = max(training["experiments"], key=lambda e: e["primary_metric_value"])
    facts = [
        f"Target: {training['target_column']} ({training['task_type']})",
        f"Rows: {training['n_train']} train / {training['n_test']} test",
        f"Features used: {len(training['features_used'])}",
        f"Dropped: {', '.join(d['column'] for d in training['features_dropped']) or 'none'}",
        f"Best model: {best['model_name']}",
        "Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in best["metrics"].items()),
        "All models: " + "; ".join(
            f"{e['model_name']} {e['primary_metric']}={e['primary_metric_value']:.4f}"
            for e in training["experiments"]
        ),
        f"Planner rationale: {plan['rationale']}",
    ]
    explanation = state.get("explanation")
    if explanation and explanation.get("features"):
        method = explanation["method"]
        # Quote the names. Formatting them as `mass (was missing) (0.0334)`
        # gave two parenthetical groups, and the model read the first as the
        # column name -- reporting that "mass" drove predictions when the
        # driver was mass's ABSENCE. That is the exact misattribution the
        # indicator exists to prevent.
        top = "; ".join(
            f'"{f["feature"]}" = {f["importance"]:.4f}'
            for f in explanation["features"][:5]
        )
        facts.append(f"Measured feature importance ({method}), strongest first: {top}")
        facts.append(
            "Use these names EXACTLY as quoted. Any column not listed did not "
            "meaningfully drive predictions."
        )
        if any(f["feature"].endswith("(was missing)")
               for f in explanation["features"][:5]):
            facts.append(
                'IMPORTANT: a name ending in "(was missing)" means predictions '
                "were driven by WHETHER that value was recorded, not by the "
                'value itself. Never shorten it to the bare column name -- '
                '"mass (was missing)" and "mass" are different findings.'
            )

    cleaning = state.get("cleaning")
    if cleaning and cleaning.get("changed"):
        facts.append("Cleaning applied before modelling: " + "; ".join(
            f"{a['action']}"
            + (f" ({a['rows_affected']} rows)" if a["rows_affected"] else "")
            + (f" on {', '.join(a['columns'])}" if a["columns"] else "")
            for a in cleaning["actions"]
        ))

    attempts = state.get("attempts") or []
    if len(attempts) > 1:
        facts.append("Attempts: " + " | ".join(
            f"round {a['round']}: {a['target_column']}, "
            f"{a['primary_metric']}={a['primary_metric_value']:.4f}"
            + (f", excluded {len(a['excluded_features'])} feature(s)"
               if a["excluded_features"] else "")
            for a in attempts
        ))
    reflection = state.get("reflection") or {}
    if reflection.get("action") in {"abandon", "accept"} and reflection.get("reasoning"):
        facts.append(f"Reflection verdict ({reflection['action']}): {reflection['reasoning']}")

    if plan.get("data_quality_concerns"):
        facts.append("Observed data-quality notes (not proven causes): "
                     + "; ".join(plan["data_quality_concerns"]))

    try:
        response = await get_llm().ainvoke(
            [("system", SUMMARY_SYSTEM), ("human", "\n".join(facts))]
        )
        text = message_text(response)
        return {"summary": text, "steps": [*state.get("steps", []), "summary:ok"]}
    except Exception as exc:
        logger.warning("Summary LLM call failed", extra={"error": str(exc)})
        # A failed summary must not fail the run -- the metrics are the value.
        return {
            "summary": (
                f"Best model {best['model_name']} scored "
                f"{best['primary_metric']}={best['primary_metric_value']:.4f} "
                f"predicting {training['target_column']}. "
                f"(Narrative unavailable: {type(exc).__name__}.)"
            ),
            "steps": [*state.get("steps", []), "summary:fallback"],
        }


# --------------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------------
MAX_REFLECTION_ROUNDS = 2

REFLECTION_SYSTEM = """You are the reflection agent. A model has been trained \
and scored, and a deterministic quality check has judged the result weak. \
Decide what to do next.

You have three options:

- retry: change something concrete and train again. Only worth doing if you \
have a specific, evidence-backed change.
- abandon: the data cannot answer this question well. Say so.
- accept: the result is weak but is the best available and still informative.

Rules:
- Base every decision on the measured numbers you are given. Do not speculate.
- You are given the COMPLETE list of failed checks and the list of checks that \
passed. Never cite a passed check as a reason for anything. In particular, do \
not call a dataset too small unless the sample-size check actually failed.
- To exclude features, name only columns that appear in the feature list. \
Excluding features with near-zero measured importance is the safest change; \
it removes noise without losing signal.
- Changing the target is a big move. Only do it if the current target is \
clearly unsuitable and a listed alternative is clearly better.
- Prefer abandon over a retry you do not expect to help. A weak result \
honestly reported is more useful than three rounds of churn.
- If the problem is too few rows, no modelling change will fix it. Abandon.
- Keep reasoning to two or three sentences.
"""


class ReflectionOutput(BaseModel):
    """Structured output schema for the reflection agent."""

    action: Literal["retry", "abandon", "accept"]
    reasoning: str = Field(description="Two or three sentences citing measured values.")
    exclude_features: list[str] = Field(
        default_factory=list, description="Feature columns to drop on the retry."
    )
    new_target: str | None = Field(
        default=None, description="Only if the current target is unsuitable."
    )
    new_task_type: Literal["classification", "regression"] | None = None


def _validate_revision(
    revision: dict[str, Any],
    state: AnalysisState,
    valid_columns: set[str],
    feature_columns: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Sanity-check a proposed revision. Same principle as the planner guard:
    a model's suggestion is a proposal, never an instruction."""
    notes: list[str] = []

    unknown = [c for c in revision["exclude_features"] if c not in feature_columns]
    if unknown:
        notes.append(f"ignored unknown columns: {', '.join(unknown)}")
        revision["exclude_features"] = [
            c for c in revision["exclude_features"] if c in feature_columns
        ]

    already = set(state.get("excluded_features") or ())
    combined = already | set(revision["exclude_features"])
    if combined >= feature_columns:
        notes.append("refused to exclude every remaining feature")
        revision["action"] = "abandon"
        return revision, notes

    if revision["new_target"] and revision["new_target"] not in valid_columns:
        notes.append(f"ignored unknown target '{revision['new_target']}'")
        revision["new_target"] = None

    # A retry that changes nothing would loop forever at the same score.
    if revision["action"] == "retry":
        changes_something = (
            set(revision["exclude_features"]) - already
        ) or revision["new_target"]
        if not changes_something:
            notes.append("retry proposed no actual change; treating as accept")
            revision["action"] = "accept"

    return revision, notes


async def reflection_node(state: AnalysisState) -> dict[str, Any]:
    """Judge the round, and decide whether another is worth running.

    The *decision to reflect* is deterministic -- services/quality.assess
    applies thresholds. The LLM is consulted only about what to do when the
    result is genuinely weak, so a good run costs no extra tokens.
    """
    training, plan = state.get("training"), state.get("plan")
    if not training or not plan:
        return {"reflection": None, "steps": [*state.get("steps", []), "reflect:skipped"]}

    report = assess(training, state.get("explanation"))
    round_no = state.get("round", 0)
    # Progress is judged on the gating metric, not the headline one.
    current = report.gate_value
    previous_best = state.get("best_score")

    # Stamp the gate metric onto the attempt that just finished. Without it a
    # reader sees F1 in the attempts table while every retry decision was made
    # on ROC-AUC, which makes the loop's behaviour look arbitrary.
    attempts = list(state.get("attempts") or ())
    if attempts:
        attempts[-1] = {
            **attempts[-1],
            "gate_metric": report.gate_metric,
            "gate_value": current,
            "verdict": report.verdict,
        }

    base = {
        "quality": report.to_dict(),
        "best_score": max(current, previous_best or current),
        "attempts": attempts,
    }

    # --- deterministic stop conditions, checked before spending a token ---
    if not report.needs_reflection:
        return {**base, "reflection": {"action": "accept", "reasoning":
                f"Quality check passed ({report.gate_metric}={current:.4f})."},
                "steps": [*state.get("steps", []), f"reflect:accept({report.verdict})"]}

    if round_no >= MAX_REFLECTION_ROUNDS:
        return {**base, "reflection": {"action": "accept", "reasoning":
                f"Reached the {MAX_REFLECTION_ROUNDS}-round retry limit."},
                "steps": [*state.get("steps", []), "reflect:limit"]}

    if previous_best is not None and not is_improvement(current, previous_best):
        return {**base, "reflection": {"action": "accept", "reasoning":
                f"Retry scored {report.gate_metric}={current:.4f} against a previous "
                f"best of {previous_best:.4f}; no meaningful improvement, so stopping."},
                "steps": [*state.get("steps", []), "reflect:no_improvement"]}

    # --- only now is the LLM worth consulting ---
    facts = [
        f"Target: {training['target_column']} ({training['task_type']})",
        f"Best model: {training['best_model']}, "
        f"{report.primary_metric}={report.primary_value:.4f}"
        + (f", {report.gate_metric}={current:.4f}"
           if report.gate_metric != report.primary_metric else ""),
        f"Rows: {training['n_train']} train / {training['n_test']} test",
        f"Features in use: {', '.join(training['features_used'])}",
        f"Already excluded: {', '.join(state.get('excluded_features') or []) or 'none'}",
        f"Attempt {round_no + 1} of {MAX_REFLECTION_ROUNDS + 1}.",
        "",
        "Why the quality check failed (this list is COMPLETE):",
        *(f"  - {r}" for r in report.reasons),
        "",
        "Checks that PASSED. These are NOT problems and must not be cited as reasons:",
        *(f"  - {c['detail']}" for c in report.checks if c["passed"]),
    ]
    if report.suggestions:
        facts += ["", "Observations:", *(f"  - {s}" for s in report.suggestions)]
    explanation = state.get("explanation")
    if explanation and explanation.get("features"):
        facts += ["", "Measured feature importance:"]
        facts += [
            f"  {f['feature']}: {f['importance']:.5f}"
            for f in explanation["features"]
        ]

    try:
        llm = get_llm().with_structured_output(ReflectionOutput)
        result: ReflectionOutput = await llm.ainvoke(
            [("system", REFLECTION_SYSTEM), ("human", "\n".join(facts))]
        )
        revision = result.model_dump()
    except Exception as exc:
        logger.warning("Reflection LLM failed", extra={"error": str(exc)})
        return {**base, "reflection": {"action": "accept", "reasoning":
                f"Reflection unavailable ({type(exc).__name__}); "
                "reporting the current result."},
                "steps": [*state.get("steps", []), "reflect:unavailable"]}

    valid_columns = {c["name"] for c in state["profile"].get("columns", [])}
    feature_columns = set(training["features_used"])
    revision, notes = _validate_revision(revision, state, valid_columns, feature_columns)
    if notes:
        revision["reasoning"] += f" (Guard: {'; '.join(notes)}.)"

    if revision["action"] != "retry":
        return {**base, "reflection": revision,
                "steps": [*state.get("steps", []), f"reflect:{revision['action']}"]}

    new_plan = dict(plan)
    if revision["new_target"]:
        new_plan["target_column"] = revision["new_target"]
        if revision["new_task_type"]:
            new_plan["task_type"] = revision["new_task_type"]
        new_plan["rationale"] = f"Revised by reflection: {revision['reasoning']}"

    excluded = sorted(
        set(state.get("excluded_features") or ()) | set(revision["exclude_features"])
    )
    logger.info("Reflection retry", extra={"round": round_no + 1, "excluded": len(excluded)})
    return {
        **base,
        "reflection": revision,
        "plan": new_plan,
        "excluded_features": excluded,
        "round": round_no + 1,
        "steps": [*state.get("steps", []), f"reflect:retry(r{round_no + 1})"],
    }
