"""Model explainability. Deterministic, no LLM.

Two tiers, chosen automatically:

  1. SHAP TreeExplainer for tree ensembles -- exact, fast, and the method the
     blueprint calls for.
  2. Permutation importance for everything else -- model-agnostic, sklearn-only,
     and measured on held-out data rather than training impurity.

Encoded feature names are aggregated back to source columns. A stakeholder
wants to know that "contract" matters, not that
"cat__contract_Month-to-month" does.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from app.core.logging import get_logger

logger = get_logger(__name__)

# SHAP cost is driven by TREE SIZE, not row count: TreeExplainer scales with
# leaves squared. On the diamonds dataset a 3.7M-node forest took 88 seconds
# for 50 rows, which would have been ~15 minutes at the old flat 500-row cap.
# Because that work is CPU-bound it starves the event loop and the whole API
# stops responding -- so the cost has to be predicted, not discovered.
#
# Empirically ~1.9e-7 seconds per (row x node). Budget the product instead of
# fixing the rows.
MAX_SHAP_ROWS = 500
MIN_SHAP_ROWS = 60
SHAP_ROW_NODE_BUDGET = 60_000_000        # ~11s of work
SHAP_MAX_MODEL_NODES = 2_500_000         # beyond this, do not attempt SHAP
PERMUTATION_REPEATS = 5
TOP_FEATURES_REPORTED = 15

TREE_MODELS = ("RandomForest", "GradientBoosting", "ExtraTrees", "DecisionTree",
               "HistGradientBoosting", "XGB", "LGBM", "CatBoost")


@dataclass
class FeatureImportance:
    feature: str
    importance: float
    rank: int
    encoded_parts: int = 1   # how many one-hot columns were folded in


@dataclass
class Explanation:
    method: Literal["shap", "permutation", "unavailable"]
    model_name: str
    rows_explained: int
    features: list[FeatureImportance] = field(default_factory=list)
    encoded_features: list[FeatureImportance] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = [asdict(f) for f in self.features]
        payload["encoded_features"] = [asdict(f) for f in self.encoded_features]
        return payload

    def top_feature_names(self, n: int = 5) -> list[str]:
        return [f.feature for f in self.features[:n]]


# --------------------------------------------------------------------------
# Name mapping
# --------------------------------------------------------------------------
def _source_column(encoded: str, numeric: list[str], categorical: list[str]) -> str:
    """Map an encoded feature name back to the original dataset column.

    ColumnTransformer emits 'num__age' and 'cat__region_North'. The latter is
    ambiguous without the source list, since a column could legitimately be
    called 'region_North', so we match against known names longest-first.
    """
    name = encoded
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Missingness indicators must stay separate from the column they describe:
    # "the value of mass" and "whether mass was recorded" are different facts,
    # and merging them hides exactly what we added the indicator to reveal.
    if name.startswith("missingindicator_"):
        return f"{name[len('missingindicator_'):]} (was missing)"

    if name in numeric or name in categorical:
        return name
    # One-hot: 'region_North' -> 'region'. Longest match wins so that
    # 'plan_type_A' resolves to 'plan_type', not 'plan'.
    for col in sorted(categorical, key=len, reverse=True):
        if name.startswith(f"{col}_"):
            return col
    return name


def _aggregate(
    encoded_names: list[str],
    scores: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> tuple[list[FeatureImportance], list[FeatureImportance]]:
    """Fold one-hot columns back into their source column by summing."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, score in zip(encoded_names, scores, strict=False):
        source = _source_column(name, numeric, categorical)
        totals[source] = totals.get(source, 0.0) + float(score)
        counts[source] = counts.get(source, 0) + 1

    aggregated = [
        FeatureImportance(feature=k, importance=round(v, 6), rank=0, encoded_parts=counts[k])
        for k, v in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    for i, f in enumerate(aggregated, start=1):
        f.rank = i

    encoded = [
        FeatureImportance(feature=n, importance=round(float(s), 6), rank=0)
        for n, s in sorted(
            zip(encoded_names, scores, strict=False), key=lambda p: p[1], reverse=True
        )
    ]
    for i, f in enumerate(encoded, start=1):
        f.rank = i

    return aggregated[:TOP_FEATURES_REPORTED], encoded[:TOP_FEATURES_REPORTED]


def _is_tree_model(estimator: Any) -> bool:
    return any(t in type(estimator).__name__ for t in TREE_MODELS)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def _model_nodes(model: Any) -> int:
    """Total decision nodes across the ensemble, or 0 if not a forest."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return 0
    total = 0
    for est in np.ravel(estimators):
        tree = getattr(est, "tree_", None)
        if tree is not None:
            total += int(tree.node_count)
    return total


class ShapTooExpensive(RuntimeError):
    """The model is large enough that SHAP would block for minutes."""


def _shap_row_budget(nodes: int) -> int:
    """How many rows can be explained within the time budget."""
    if nodes <= 0:
        return MAX_SHAP_ROWS
    return int(np.clip(SHAP_ROW_NODE_BUDGET // nodes, MIN_SHAP_ROWS, MAX_SHAP_ROWS))


def _shap_importances(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[list[str], np.ndarray, int]:
    """Mean absolute SHAP value per encoded feature."""
    import shap  # imported lazily; absence must not break the app

    prep, model = pipeline["prep"], pipeline["model"]

    nodes = _model_nodes(model)
    if nodes > SHAP_MAX_MODEL_NODES:
        raise ShapTooExpensive(
            f"{nodes:,} nodes exceeds the SHAP budget; using permutation importance."
        )
    rows = _shap_row_budget(nodes)
    logger.info("SHAP budget", extra={"nodes": nodes, "rows": rows})

    sample = X.head(rows)
    encoded = prep.transform(sample)
    names = list(prep.get_feature_names_out())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(model).shap_values(encoded)

    arr = np.asarray(values)
    # Binary/multiclass classifiers return (rows, features, classes) in
    # shap 0.5x. Collapse the class axis; for binary take the positive class.
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] == 2 else np.abs(arr).mean(axis=2)

    return names, np.abs(arr).mean(axis=0), len(sample)


# --------------------------------------------------------------------------
# Permutation importance
# --------------------------------------------------------------------------
def _permutation_importances(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[list[str], np.ndarray, int]:
    """Drop in score when each ORIGINAL column is shuffled.

    Applied to the whole pipeline, so results are already per source column --
    no one-hot aggregation needed, and preprocessing is included in the effect.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            pipeline, X, y,
            n_repeats=PERMUTATION_REPEATS, random_state=42, n_jobs=1,
        )
    # Negative means shuffling helped, i.e. noise. Clamp to zero.
    return list(X.columns), np.clip(result.importances_mean, 0, None), len(X)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def explain(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    numeric: list[str],
    categorical: list[str],
) -> Explanation:
    """Explain a fitted pipeline. Blocking -- run in a thread."""
    # Probing the estimator must not itself be able to raise: explainability
    # is an enhancement, and nothing here may take down a completed run.
    try:
        is_tree = _is_tree_model(pipeline["model"])
    except Exception:
        is_tree = False

    if is_tree:
        try:
            names, scores, n = _shap_importances(pipeline, X)
            aggregated, encoded = _aggregate(names, scores, numeric, categorical)
            return Explanation(
                method="shap", model_name=model_name, rows_explained=n,
                features=aggregated, encoded_features=encoded,
                note="Mean absolute SHAP value per feature, summed across one-hot columns.",
            )
        except ShapTooExpensive as exc:
            logger.info("Skipping SHAP", extra={"reason": str(exc)})
        except ImportError:
            logger.info("shap not installed; using permutation importance")
        except Exception as exc:
            logger.warning("SHAP failed, falling back", extra={"error": str(exc)})

    try:
        names, scores, n = _permutation_importances(pipeline, X, y)
        # Already per source column, so aggregation is a no-op pass-through.
        features = [
            FeatureImportance(feature=nm, importance=round(float(s), 6), rank=i)
            for i, (nm, s) in enumerate(
                sorted(zip(names, scores, strict=False), key=lambda p: p[1], reverse=True),
                start=1,
            )
        ][:TOP_FEATURES_REPORTED]
        return Explanation(
            method="permutation", model_name=model_name, rows_explained=n,
            features=features,
            note="Mean decrease in score when each column is shuffled, over "
                 f"{PERMUTATION_REPEATS} repeats.",
        )
    except Exception as exc:
        logger.warning("Explainability unavailable", extra={"error": str(exc)})
        return Explanation(
            method="unavailable", model_name=model_name, rows_explained=0,
            note=f"Could not compute importances: {type(exc).__name__}",
        )
