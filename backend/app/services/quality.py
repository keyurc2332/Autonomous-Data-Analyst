"""Deterministic assessment of whether a modelling result is good enough.

Deliberately not an LLM call. Whether a result is weak is a threshold
question with a defensible answer; asking a model would make it
non-reproducible and cost tokens on every run, including good ones. The LLM
is consulted only about what to *do* when this module says something is wrong.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Below these, a model is not usable for decisions.
MIN_ROC_AUC = 0.65
MIN_F1 = 0.60
MIN_R2 = 0.30
MIN_TRAINING_ROWS = 200
# A perfect score on real data is nearly always a defect, not an achievement.
# Lowered from 0.999 after taxis scored R2 0.9969 on pure arithmetic and
# slipped through. Diamonds at 0.978 is the highest legitimate score observed.
IMPLAUSIBLE_SCORE = 0.98

# A retry must beat the incumbent by at least this much to count as progress.
MIN_IMPROVEMENT = 0.02

# Features contributing less than this share of total importance are inert.
DEAD_FEATURE_SHARE = 0.02


@dataclass
class QualityReport:
    verdict: Literal["strong", "acceptable", "weak"]
    primary_metric: str
    primary_value: float
    # The metric the verdict actually hinges on. For classification that is
    # ROC-AUC when available, because class separation is what the gate tests.
    # Progress between rounds must be measured on the SAME metric that raised
    # the concern -- otherwise a retry can fix the stated problem and still be
    # rejected for a dip in an unrelated number.
    gate_metric: str = ""
    gate_value: float = 0.0
    reasons: list[str] = field(default_factory=list)
    dead_features: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    # Every check run, with its outcome. Passed checks are stated explicitly
    # so the reflection agent cannot cite one as a problem: given only a raw
    # number like "482 rows" it will invent "too small" as a justification.
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed_checks(self) -> list[str]:
        return [c["name"] for c in self.checks if c["passed"]]

    @property
    def needs_reflection(self) -> bool:
        return self.verdict == "weak"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dead_features(explanation: dict[str, Any] | None) -> list[str]:
    """Features whose measured contribution is negligible."""
    if not explanation or not explanation.get("features"):
        return []
    features = explanation["features"]
    total = sum(abs(f["importance"]) for f in features)
    if total <= 0:
        return [f["feature"] for f in features]
    return [
        f["feature"] for f in features
        if abs(f["importance"]) / total < DEAD_FEATURE_SHARE
    ]


def assess(
    training: dict[str, Any], explanation: dict[str, Any] | None = None
) -> QualityReport:
    """Judge a completed training round."""
    experiments = training.get("experiments") or []
    if not experiments:
        return QualityReport(
            verdict="weak", primary_metric="none", primary_value=0.0,
            reasons=["No models trained successfully."], gate_metric="none",
        )

    best = max(experiments, key=lambda e: e["primary_metric_value"])
    metric = best["primary_metric"]
    value = float(best["primary_metric_value"])
    metrics = best.get("metrics") or {}
    task = training.get("task_type")

    reasons: list[str] = []
    suggestions: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str, gating: bool = True) -> None:
        """gating=False records a finding without forcing a 'weak' verdict.

        Leakage that has already been removed is worth showing the reader, but
        it does not make the resulting honest model bad.
        """
        checks.append({"name": name, "passed": passed, "detail": detail,
                       "gating": gating})
        if not passed and gating:
            reasons.append(detail)

    gate_metric, gate_value = metric, value

    if task == "classification":
        auc = metrics.get("roc_auc")
        if auc is not None:
            gate_metric, gate_value = "roc_auc", float(auc)
            record("class_separation", auc >= MIN_ROC_AUC,
                   f"ROC-AUC {auc:.4f} is below {MIN_ROC_AUC}; barely separates classes."
                   if auc < MIN_ROC_AUC else
                   f"ROC-AUC {auc:.4f} meets the {MIN_ROC_AUC} threshold.")
        record("f1_threshold", value >= MIN_F1,
               f"F1 {value:.4f} is below {MIN_F1}." if value < MIN_F1
               else f"F1 {value:.4f} meets the {MIN_F1} threshold.")
    elif task == "regression":
        record("variance_explained", value >= MIN_R2,
               f"R2 {value:.4f} is below {MIN_R2}; the model explains little variance."
               if value < MIN_R2 else f"R2 {value:.4f} meets the {MIN_R2} threshold.")

    leaked = training.get("leaked_features") or []
    record("target_leakage", not leaked,
           "No feature restates the target." if not leaked else
           "Removed before training because they restate the target: "
           + "; ".join(f"{f['column']} (mutual information {f['score']})" for f in leaked),
           gating=False)
    if leaked:
        suggestions.append(
            f"{len(leaked)} leaked feature(s) were excluded. Scores below are "
            "from the model trained WITHOUT them."
        )

    if value >= IMPLAUSIBLE_SCORE:
        record("plausibility", False,
               f"{metric} of {value:.4f} is essentially perfect. On real data "
               "this usually means leakage or a duplicated target, not skill.",
               gating=False)
        suggestions.append("Verify no remaining column encodes the answer.")

    additive = training.get("additive_leakage")
    record("target_is_derived", not additive,
           "The target is not a combination of its own features."
           if not additive else additive["reason"])

    # Only SHAP exposes missingness indicators: it scores encoded features,
    # while permutation importance shuffles the original column and folds the
    # missingness effect into it. So this check fires for tree models and not
    # for linear ones -- a known limitation, not a silent failure.
    missingness_drivers = [
        f["feature"] for f in (explanation or {}).get("features", [])[:5]
        if f["feature"].endswith("(was missing)")
    ]
    if missingness_drivers:
        record("informative_missingness", False,
               "Predictions are driven by whether a value was recorded, not by "
               "the value itself: " + ", ".join(missingness_drivers)
               + ". This is real signal only if the same columns will be missing "
                 "the same way in future data.",
               gating=False)
        suggestions.append(
            "Check why those columns are missing. If missingness reflects how "
            "the data was collected rather than the subject, the model will not "
            "transfer."
        )

    dead = _dead_features(explanation)
    if dead:
        suggestions.append(
            f"{len(dead)} feature(s) contribute almost nothing: {', '.join(dead[:5])}. "
            "Removing them reduces noise without losing signal."
        )

    n_train = training.get("n_train") or 0
    record("sample_size", n_train >= MIN_TRAINING_ROWS,
           f"Only {n_train} training rows; results are unstable."
           if n_train < MIN_TRAINING_ROWS else
           f"{n_train} training rows is sufficient; sample size is NOT a problem.")
    if n_train < MIN_TRAINING_ROWS:
        suggestions.append("More data would help more than any model change.")

    if not reasons:
        verdict = "strong" if (
            (task == "classification" and value >= 0.75)
            or (task == "regression" and value >= 0.60)
        ) else "acceptable"
    else:
        verdict = "weak"

    return QualityReport(
        verdict=verdict, primary_metric=metric, primary_value=value,
        reasons=reasons, dead_features=dead, suggestions=suggestions, checks=checks,
        gate_metric=gate_metric, gate_value=gate_value,
    )


def is_improvement(new_value: float, previous_best: float) -> bool:
    """Whether a retry genuinely beat the incumbent.

    Requiring a margin rather than any increase stops the loop from churning
    on noise-level differences between runs.
    """
    return new_value >= previous_best + MIN_IMPROVEMENT
