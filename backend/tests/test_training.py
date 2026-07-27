"""Training tests. Deterministic -- no LLM, no API quota."""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.profiling import profile_file
from app.services.training import TrainingError, select_features, train


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(3)
    n = 400
    tenure = rng.integers(1, 60, n)
    charges = np.round(rng.normal(70, 20, n).clip(20, 130), 2)
    signal = (0.5 - 0.01 * tenure + 0.006 * (charges - 70)).clip(0.02, 0.95)
    df = pd.DataFrame({
        "row_id": [f"R-{i:05d}" for i in range(n)],
        "tenure": tenure,
        "charges": charges,
        "plan": rng.choice(["A", "B", "C"], n),
        "currency": ["INR"] * n,
        "joined": pd.date_range("2022-01-01", periods=n).astype(str),
        "churn": (rng.random(n) < signal).astype(int),
    })
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return path, profile_file(path)


def test_feature_selection_drops_unusable_columns(dataset):
    _, profile = dataset
    numeric, categorical, dropped = select_features(profile, "churn")
    dropped_cols = {d["column"] for d in dropped}

    assert "row_id" in dropped_cols       # identifier
    assert "currency" in dropped_cols     # constant
    assert "joined" in dropped_cols       # datetime, not engineered yet
    assert "churn" not in numeric + categorical + list(dropped_cols)
    assert set(numeric) == {"tenure", "charges"}
    assert categorical == ["plan"]


def test_classification_produces_ranked_experiments(dataset):
    path, profile = dataset
    out = train(path, profile, "churn", "classification")

    assert out.task_type == "classification"
    assert len(out.experiments) == 2
    assert out.best_model in {"logistic_regression", "random_forest"}
    for e in out.experiments:
        assert e.primary_metric == "f1"
        assert 0.0 <= e.metrics["accuracy"] <= 1.0
        assert e.feature_count > 0
    # best_model must genuinely be the highest scorer
    assert out.best_model == max(
        out.experiments, key=lambda e: e.primary_metric_value
    ).model_name


def test_regression_produces_r2(dataset):
    path, profile = dataset
    out = train(path, profile, "charges", "regression")
    assert out.task_type == "regression"
    for e in out.experiments:
        assert e.primary_metric == "r2"
        assert {"r2", "mae", "rmse"} <= set(e.metrics)


def test_result_is_json_serialisable(dataset):
    path, profile = dataset
    out = train(path, profile, "churn", "classification")
    json.dumps(out.to_dict(), allow_nan=False)


def test_train_test_split_is_deterministic(dataset):
    """Same input must give the same numbers, or the report is meaningless."""
    path, profile = dataset
    a = train(path, profile, "churn", "classification")
    b = train(path, profile, "churn", "classification")
    assert [e.primary_metric_value for e in a.experiments] == \
           [e.primary_metric_value for e in b.experiments]


def test_unknown_target_is_rejected(dataset):
    path, profile = dataset
    with pytest.raises(TrainingError, match="not a column"):
        train(path, profile, "nope", "classification")


def test_constant_target_is_rejected(dataset):
    path, profile = dataset
    with pytest.raises(TrainingError, match="one distinct value"):
        train(path, profile, "currency", "classification")


def test_regression_on_text_target_is_rejected(dataset):
    path, profile = dataset
    with pytest.raises(TrainingError, match="numeric target"):
        train(path, profile, "plan", "regression")


def test_string_binary_column_routes_to_categorical(tmp_path):
    """A two-value STRING column is 'binary' but must not go to the numeric
    pipeline -- a median imputer on strings makes every model fail to fit."""
    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame({
        "flag_text": rng.choice(["yes", "no"], n),   # binary, but strings
        "flag_num": rng.choice([0, 1], n),           # binary, numeric
        "value": rng.normal(50, 10, n),
        "label": rng.choice([0, 1], n),
    })
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    numeric, categorical, _ = select_features(profile, "label")
    assert "flag_text" in categorical
    assert "flag_num" in numeric

    out = train(path, profile, "label", "classification")
    assert len(out.experiments) == 2   # both models fitted successfully


# --------------------------------------------------------------------------
# Target leakage
# --------------------------------------------------------------------------
def test_detects_a_restated_target(tmp_path):
    """The Titanic case: `alive` is `survived` as yes/no.

    A model using it scores a perfect 1.000 and has learned nothing. Pearson
    correlation cannot see this -- both columns are categorical.
    """
    from app.services.training import prepare_split

    rng = np.random.default_rng(12)
    n = 300
    survived = rng.choice([0, 1], n)
    df = pd.DataFrame({
        "survived": survived,
        "alive": np.where(survived == 1, "yes", "no"),   # the same variable
        "fare": rng.normal(30, 12, n),
        "pclass": rng.choice([1, 2, 3], n),
    })
    path = tmp_path / "leak.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    split = prepare_split(path, profile, "survived", "classification")
    assert [f["column"] for f in split.leakage] == ["alive"]
    assert "alive" not in split.numeric + split.categorical
    assert "fare" in split.numeric


def test_honest_features_are_not_flagged(tmp_path):
    """A genuinely predictive feature must survive; only restatements go."""
    from app.services.training import prepare_split

    rng = np.random.default_rng(13)
    n = 400
    driver = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "driver": driver,
        "noise": rng.normal(0, 1, n),
        "label": (driver + rng.normal(0, 0.4, n) > 0).astype(int),
    })
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)

    split = prepare_split(path, profile_file(path), "label", "classification")
    assert split.leakage == []
    assert "driver" in split.numeric


def test_all_features_leaking_is_an_error(tmp_path):
    from app.services.training import prepare_split

    n = 200
    y = np.arange(n) % 2
    df = pd.DataFrame({"y": y, "copy_a": y, "copy_b": np.where(y == 1, "T", "F")})
    path = tmp_path / "all_leak.csv"
    df.to_csv(path, index=False)

    with pytest.raises(TrainingError, match="restates the target"):
        prepare_split(path, profile_file(path), "y", "classification")


def test_boolean_columns_are_usable_features(tmp_path):
    """Regression: booleans were dropped as 'unhandled type', discarding
    Titanic's adult_male and alone."""
    rng = np.random.default_rng(14)
    n = 300
    flag = rng.choice([True, False], n)
    df = pd.DataFrame({
        "flag": flag,
        "value": rng.normal(0, 1, n),
        "label": np.where(flag, rng.choice([0, 1], n, p=[0.3, 0.7]),
                          rng.choice([0, 1], n, p=[0.8, 0.2])),
    })
    path = tmp_path / "bool.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    numeric, categorical, dropped = select_features(profile, "label")
    assert "flag" in numeric + categorical
    assert "flag" not in {d["column"] for d in dropped}


def test_large_tables_are_sampled_for_interactive_runs(tmp_path):
    from app.services.training import INTERACTIVE_ROW_LIMIT, prepare_split

    n = INTERACTIVE_ROW_LIMIT + 5_000
    rng = np.random.default_rng(15)
    df = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.choice([0, 1], n)})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)

    split = prepare_split(path, profile_file(path), "y", "classification")
    assert split.sampled_from == n
    assert len(split.X_train) + len(split.X_test) == INTERACTIVE_ROW_LIMIT


# --------------------------------------------------------------------------
# Additive leakage
# --------------------------------------------------------------------------
def test_target_that_sums_its_own_features_is_detected(tmp_path):
    """The taxis case: total = fare + tip + tolls.

    No single column restates the target, so per-feature mutual information
    passes every check while the model scores R2 0.997 by learning addition.
    """
    from app.services.training import prepare_split

    rng = np.random.default_rng(41)
    n = 400
    fare = np.round(rng.uniform(5, 60, n), 2)
    tip = np.round(rng.uniform(0, 12, n), 2)
    tolls = np.round(rng.choice([0, 0, 0, 6.5], n), 2)
    df = pd.DataFrame({
        "fare": fare, "tip": tip, "tolls": tolls,
        "distance": np.round(rng.uniform(0.5, 20, n), 2),
        "total": fare + tip + tolls,
    })
    path = tmp_path / "sum.csv"
    df.to_csv(path, index=False)

    split = prepare_split(path, profile_file(path), "total", "regression")
    assert split.additive_leakage is not None
    assert split.additive_leakage["r2"] >= 0.99
    contributors = {c["column"] for c in split.additive_leakage["contributors"]}
    assert {"fare", "tip"} <= contributors


def test_genuine_signal_is_not_flagged_as_derived(tmp_path):
    """Strong but honest prediction must survive."""
    from app.services.training import prepare_split

    rng = np.random.default_rng(42)
    n = 400
    size = rng.uniform(0.3, 3.0, n)
    df = pd.DataFrame({
        "size": size,
        "quality": rng.integers(1, 8, n),
        "price": 4000 * size**1.9 + rng.normal(0, 900, n),   # non-linear + noise
    })
    path = tmp_path / "honest.csv"
    df.to_csv(path, index=False)

    split = prepare_split(path, profile_file(path), "price", "regression")
    assert split.additive_leakage is None


def test_additive_check_is_regression_only(tmp_path):
    """A classification target cannot be a linear sum of its features."""
    from app.services.training import prepare_split

    rng = np.random.default_rng(43)
    n = 300
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "b": b, "label": (a + b > 0).astype(int)})
    path = tmp_path / "cls.csv"
    df.to_csv(path, index=False)

    split = prepare_split(path, profile_file(path), "label", "classification")
    assert split.additive_leakage is None


def test_derived_target_makes_the_verdict_weak(tmp_path):
    """A near-perfect score from arithmetic must not be reported as strong."""
    from app.services.quality import assess

    rng = np.random.default_rng(44)
    n = 400
    x, y_ = rng.uniform(1, 50, n), rng.uniform(1, 20, n)
    df = pd.DataFrame({"x": x, "y": y_, "other": rng.normal(0, 1, n), "total": x + y_})
    path = tmp_path / "derived.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    out = train(path, profile, "total", "regression")
    report = assess(out.to_dict())

    assert out.additive_leakage is not None
    assert report.verdict == "weak"
    check = next(c for c in report.checks if c["name"] == "target_is_derived")
    assert check["passed"] is False
