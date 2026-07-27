"""Model training. Deterministic, no LLM.

The Planner decides *what* to train; this module decides *how* and does it.
Keeping the split sharp means training is unit-testable without API quota,
and a hallucinated plan cannot produce a corrupt pipeline -- it produces a
validation error instead.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.logging import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Semantic types the profiler flags as unusable as features.
EXCLUDED_SEMANTIC_TYPES = {"identifier", "constant", "empty", "text"}
MAX_TRAINING_ROWS = 100_000
# Above this we sample. Training and explaining 50k+ rows inline can take
# minutes, and because the work is CPU-bound it starves the event loop --
# the whole API becomes unresponsive, not just the one request. Sampling is
# the stopgap; moving this to a background worker is the real fix.
INTERACTIVE_ROW_LIMIT = 20_000


class TrainingError(RuntimeError):
    """Training could not proceed. The message is safe to show a user."""


@dataclass
class ExperimentResult:
    model_name: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    primary_metric: str
    primary_metric_value: float
    train_seconds: float
    feature_count: int


@dataclass
class TrainingOutcome:
    task_type: Literal["classification", "regression"]
    target_column: str
    n_train: int
    n_test: int
    features_used: list[str] = field(default_factory=list)
    features_dropped: list[dict[str, str]] = field(default_factory=list)
    leaked_features: list[dict[str, Any]] = field(default_factory=list)
    sampled_from: int | None = None
    additive_leakage: dict[str, Any] | None = None
    experiments: list[ExperimentResult] = field(default_factory=list)
    best_model: str | None = None

    # NOTE: `fitted_pipelines`, `X_test`, `y_test`, `numeric_columns` and
    # `categorical_columns` are attached as PLAIN ATTRIBUTES after
    # construction, deliberately not declared as dataclass fields. asdict()
    # would otherwise try to deep-copy a fitted sklearn Pipeline, which is
    # expensive and can fail. Explainability reads them; JSON never sees them.

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiments"] = [asdict(e) for e in self.experiments]
        return payload


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
def _is_numeric_dtype_name(dtype: str) -> bool:
    """Whether a pandas dtype string denotes numbers.

    Needed because the semantic type alone is not enough: "binary" covers
    both a 0/1 integer column and a two-value string column like "A"/"B".
    Routing the latter into the numeric pipeline hands strings to a median
    imputer, and every model fails to fit.
    """
    return dtype.startswith(("int", "uint", "float", "bool")) or dtype in {
        "Int64", "Int32", "Float64", "Float32", "boolean",
    }


def select_features(
    profile: dict[str, Any],
    target_column: str,
    exclude: list[str] | None = None,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Split columns into (numeric, categorical, dropped-with-reason).

    Driven entirely by the profile, so the same decisions are reproducible
    and explainable -- the report can state why each column was excluded.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[dict[str, str]] = []
    excluded = set(exclude or ())

    for col in profile.get("columns", []):
        name, semantic = col["name"], col["semantic_type"]
        if name == target_column:
            continue
        if name in excluded:
            dropped.append({"column": name, "reason": "excluded by reflection"})
            continue
        if semantic in EXCLUDED_SEMANTIC_TYPES:
            dropped.append({"column": name, "reason": f"semantic type '{semantic}'"})
        elif col["null_pct"] >= 80:
            dropped.append({"column": name, "reason": f"{col['null_pct']:.0f}% missing"})
        elif semantic in {"numeric", "categorical_numeric", "boolean"}:
            # Booleans were previously unhandled and silently dropped, which
            # threw away perfectly good features (Titanic's adult_male, alone).
            numeric.append(name)
        elif semantic == "binary":
            # Route by actual dtype, not by semantic label.
            if _is_numeric_dtype_name(col["dtype"]):
                numeric.append(name)
            else:
                categorical.append(name)
        elif semantic in {"categorical", "high_cardinality_categorical"}:
            if col["unique_count"] > 100:
                dropped.append(
                    {"column": name, "reason": f"{col['unique_count']} categories"}
                )
            else:
                categorical.append(name)
        else:
            # datetime / datetime_string: real feature engineering is Phase 4.
            dropped.append({"column": name, "reason": f"unhandled type '{semantic}'"})

    return numeric, categorical, dropped


def _build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                # add_indicator surfaces informative missingness instead of
                # smuggling it in. In the planets dataset 'mass' is 99.7%
                # missing for Transit detections and 7.8% for Radial Velocity,
                # so median imputation hands the model a near-perfect class
                # signal that then gets attributed to the column's *value*.
                # An explicit indicator makes the effect visible to SHAP.
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # handle_unknown='ignore' stops unseen test-set categories
                # from raising at predict time.
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    if not transformers:
        raise TrainingError("No usable feature columns remain after filtering.")
    return ColumnTransformer(transformers, remainder="drop")


# An unbounded forest on 16k rows grows ~3.7M nodes at depth 39. That is not
# better -- it overfits -- and it makes SHAP unusable, because TreeExplainer's
# cost scales with leaves squared. Measured on the diamonds dataset:
#
#   depth      nodes      R2     SHAP (100 rows)
#   None   3,690,590   0.9793   > 10 minutes
#     12     653,372   0.9780        12s
#
# 0.0013 of R2 for a 5.6x smaller model is an obvious trade.
MAX_TREE_DEPTH = 12
LARGE_DATA_ROWS = 10_000


def _forest_size(n_rows: int) -> int:
    """Fewer trees on larger tables. Accuracy plateaus; cost does not."""
    return 120 if n_rows > LARGE_DATA_ROWS else 200


def _candidate_models(task_type: str, n_rows: int = 0) -> dict[str, Any]:
    trees = _forest_size(n_rows)
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=trees, max_depth=MAX_TREE_DEPTH,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
        }
    return {
        "ridge": Ridge(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=trees, max_depth=MAX_TREE_DEPTH,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    # 'weighted' averaging keeps this correct for multiclass without
    # branching, and matches binary results when there are two classes.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # Raises when the test split happens to contain a single class.
        with contextlib.suppress(ValueError):
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
# Association above this with the target means a feature restates the answer.
LEAKAGE_NMI_THRESHOLD = 0.95
# A linear model that reconstructs a numeric target almost exactly means the
# target is a combination of its own features. Measured separation:
#   taxis (total = fare + tip + tolls)  0.994   <- leakage
#   diamonds (price)                    0.854   legitimate
#   mpg                                 0.824   legitimate
#   tips                                0.481   legitimate
ADDITIVE_LEAKAGE_R2 = 0.99
LEAKAGE_SAMPLE_ROWS = 20_000
_LEAK_BINS = 12


def _discretise(series: pd.Series) -> pd.Series:
    """Bin a column so mutual information can be computed on mixed types."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > _LEAK_BINS:
        try:
            return pd.qcut(series, _LEAK_BINS, duplicates="drop", labels=False)
        except (ValueError, TypeError):
            return pd.cut(series, _LEAK_BINS, labels=False)
    return series.astype("string")


def detect_leakage(X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    """Find features that essentially restate the target.

    Titanic is the canonical case: `alive` is `survived` as yes/no, so a model
    using it scores a perfect 1.000 and has learned nothing. Correlation alone
    misses this -- both columns are categorical, and the profiler's Pearson
    check only covers numeric pairs.

    Normalised mutual information is type-agnostic: it measures how much
    knowing the feature tells you about the target, regardless of whether
    either is a number, a string or a boolean.
    """
    from sklearn.metrics import normalized_mutual_info_score

    frame = X.join(y.rename("__target__"))
    if len(frame) > LEAKAGE_SAMPLE_ROWS:
        frame = frame.sample(LEAKAGE_SAMPLE_ROWS, random_state=RANDOM_STATE)
    frame = frame.dropna(subset=["__target__"])
    target = _discretise(frame["__target__"]).astype(str)

    found: list[dict[str, Any]] = []
    for column in X.columns:
        col = frame[column]
        if col.isna().all():
            continue
        try:
            score = float(normalized_mutual_info_score(
                target, _discretise(col).astype(str)
            ))
        except (ValueError, TypeError):
            continue
        if score >= LEAKAGE_NMI_THRESHOLD:
            found.append({
                "column": column,
                "score": round(score, 4),
                "reason": (
                    f"knowing '{column}' determines the target "
                    f"(mutual information {score:.3f}); it restates the answer"
                ),
            })

    found.sort(key=lambda f: f["score"], reverse=True)
    return found


def detect_additive_leakage(
    X: pd.DataFrame, y: pd.Series, numeric: list[str]
) -> dict[str, Any] | None:
    """Detect a target that is a combination of its own features.

    Per-feature mutual information cannot see this: in the taxis dataset
    `total` equals fare + tip + tolls, and no single component restates it, so
    every individual check passes while the model scores R2 0.997 by learning
    addition.

    Fitting a linear model on all numeric features catches it directly. If the
    features reconstruct the target almost exactly, the relationship is
    arithmetic rather than predictive.
    """
    if len(numeric) < 2 or len(X) < 20:
        return None

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    frame = X[numeric]
    try:
        model = make_pipeline(
            SimpleImputer(strategy="median"), LinearRegression()
        ).fit(frame, y)
        r2 = float(model.score(frame, y))
    except Exception:
        return None

    if r2 < ADDITIVE_LEAKAGE_R2:
        return None

    coefficients = sorted(
        zip(numeric, model[-1].coef_, strict=False),
        key=lambda pair: abs(pair[1]), reverse=True,
    )
    contributors = [
        {"column": name, "coefficient": round(float(coef), 4)}
        for name, coef in coefficients[:5]
        if abs(coef) > 1e-6
    ]
    return {
        "r2": round(r2, 5),
        "contributors": contributors,
        "reason": (
            f"A linear model reconstructs '{y.name}' from its own features with "
            f"R2 {r2:.4f}. The target appears to be a combination of "
            + ", ".join(c["column"] for c in contributors[:3])
            + ". Any model will score highly by learning arithmetic, not by "
            "predicting anything."
        ),
    }


@dataclass
class SplitData:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    numeric: list[str]
    categorical: list[str]
    dropped: list[dict[str, str]]
    leakage: list[dict[str, Any]] = field(default_factory=list)
    sampled_from: int | None = None
    additive_leakage: dict[str, Any] | None = None


def prepare_split(
    path: Path,
    profile: dict[str, Any],
    target_column: str,
    task_type: Literal["classification", "regression"],
    exclude: list[str] | None = None,
) -> SplitData:
    """Load, validate and split. Deterministic: RANDOM_STATE is fixed, so
    calling this twice with the same inputs yields byte-identical splits.

    That determinism is what lets the explain step reconstruct the exact test
    set instead of carrying DataFrames through LangGraph state.
    """
    column_names = {c["name"] for c in profile.get("columns", [])}
    if target_column not in column_names:
        raise TrainingError(
            f"Target '{target_column}' is not a column in this dataset. "
            f"Available: {', '.join(sorted(column_names))}"
        )

    df = pd.read_csv(path, nrows=MAX_TRAINING_ROWS)
    if target_column not in df.columns:
        raise TrainingError(f"Target '{target_column}' missing from the file.")

    df = df.dropna(subset=[target_column])
    if df.empty:
        raise TrainingError(f"Every value of '{target_column}' is null.")

    sampled_from = None
    if len(df) > INTERACTIVE_ROW_LIMIT:
        sampled_from = len(df)
        df = df.sample(INTERACTIVE_ROW_LIMIT, random_state=RANDOM_STATE)
        logger.info("Sampled for interactive run",
                    extra={"from": sampled_from, "to": len(df)})

    numeric, categorical, dropped = select_features(profile, target_column, exclude)
    features = numeric + categorical
    if not features:
        raise TrainingError(
            "No usable feature columns remain after filtering"
            + (" and exclusions." if exclude else ".")
        )

    X, y = df[features], df[target_column]

    # Leaked features are removed before training, not merely reported. A
    # model built on a restatement of the target is not a weak model, it is a
    # meaningless one, and reporting it as "strong" is the worst outcome.
    leakage = detect_leakage(X, y)
    if leakage:
        leaked = {f["column"] for f in leakage}
        if leaked >= set(features):
            raise TrainingError(
                "Every candidate feature restates the target "
                f"({', '.join(sorted(leaked))}). Nothing is left to learn from."
            )
        for item in leakage:
            dropped.append({"column": item["column"], "reason": item["reason"]})
        numeric = [c for c in numeric if c not in leaked]
        categorical = [c for c in categorical if c not in leaked]
        features = numeric + categorical
        X = df[features]
        logger.warning("Target leakage removed", extra={"columns": sorted(leaked)})

    if task_type == "classification":
        counts = y.value_counts()
        if len(counts) < 2:
            raise TrainingError(
                f"'{target_column}' has one distinct value; nothing to classify."
            )
        stratify = y if counts.min() >= 2 else None
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise TrainingError(
                f"Regression needs a numeric target, but '{target_column}' is "
                f"{y.dtype}. Classification may be the right task here."
            )
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Reported rather than auto-removed: unlike a single restated column, it is
    # ambiguous which component to drop, and that is the user's call. The
    # quality gate marks the run weak so reflection can propose an exclusion.
    additive = (
        detect_additive_leakage(X_train, y_train, numeric)
        if task_type == "regression" else None
    )
    if additive:
        logger.warning("Additive leakage detected", extra={"r2": additive["r2"]})

    return SplitData(X_train, X_test, y_train, y_test, numeric, categorical,
                     dropped, leakage, sampled_from, additive)


def train(
    path: Path,
    profile: dict[str, Any],
    target_column: str,
    task_type: Literal["classification", "regression"],
    exclude: list[str] | None = None,
) -> TrainingOutcome:
    """Train candidate models and return their scores. Blocking -- use a thread."""
    split = prepare_split(path, profile, target_column, task_type, exclude)
    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test
    numeric, categorical, dropped = split.numeric, split.categorical, split.dropped
    features = numeric + categorical

    outcome = TrainingOutcome(
        task_type=task_type,
        target_column=target_column,
        n_train=len(X_train),
        n_test=len(X_test),
        features_used=features,
        features_dropped=dropped,
        leaked_features=split.leakage,
        sampled_from=split.sampled_from,
        additive_leakage=split.additive_leakage,
    )

    fitted: dict[str, Pipeline] = {}
    for name, estimator in _candidate_models(task_type, len(X_train)).items():
        started = time.perf_counter()
        pipeline = Pipeline([
            ("prep", _build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if task_type == "classification":
                proba = (
                    pipeline.predict_proba(X_test)
                    if hasattr(pipeline["model"], "predict_proba") else None
                )
                metrics = _classification_metrics(y_test, y_pred, proba)
                primary = "f1"
            else:
                metrics = _regression_metrics(y_test, y_pred)
                primary = "r2"
        except Exception as exc:
            logger.warning("Model failed", extra={"model": name, "error": str(exc)})
            continue

        fitted[name] = pipeline
        n_features = pipeline["prep"].transform(X_train.head(1)).shape[1]
        outcome.experiments.append(ExperimentResult(
            model_name=name,
            hyperparameters={
                k: v for k, v in estimator.get_params().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            metrics=metrics,
            primary_metric=primary,
            primary_metric_value=metrics[primary],
            train_seconds=round(time.perf_counter() - started, 3),
            feature_count=int(n_features),
        ))

    if not outcome.experiments:
        raise TrainingError("Every candidate model failed to train.")

    best = max(outcome.experiments, key=lambda e: e.primary_metric_value)
    outcome.best_model = best.model_name
    outcome.fitted_pipelines = fitted
    outcome.X_test = X_test
    outcome.y_test = y_test
    outcome.numeric_columns = numeric
    outcome.categorical_columns = categorical
    logger.info(
        "Training complete",
        extra={"best": best.model_name, "score": best.primary_metric_value},
    )
    return outcome
