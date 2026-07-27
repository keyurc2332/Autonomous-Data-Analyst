"""Explainability tests. Deterministic -- no LLM, no API quota."""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.explain import (
    Explanation,
    _source_column,
    explain,
)
from app.services.profiling import profile_file
from app.services.training import train


@pytest.fixture
def trained(tmp_path):
    """A dataset where 'driver' genuinely determines the label and 'noise' does not."""
    rng = np.random.default_rng(4)
    n = 500
    driver = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "driver": driver,
        "noise": rng.normal(0, 1, n),
        "category": rng.choice(["X", "Y", "Z"], n),
        "label": (driver + rng.normal(0, 0.35, n) > 0).astype(int),
    })
    path = tmp_path / "e.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)
    return train(path, profile, "label", "classification")


def test_shap_used_for_tree_models(trained):
    result = explain(
        trained.fitted_pipelines["random_forest"], trained.X_test, trained.y_test,
        "random_forest", trained.numeric_columns, trained.categorical_columns,
    )
    assert result.method == "shap"
    assert result.rows_explained == len(trained.X_test)


def test_permutation_used_for_linear_models(trained):
    result = explain(
        trained.fitted_pipelines["logistic_regression"], trained.X_test, trained.y_test,
        "logistic_regression", trained.numeric_columns, trained.categorical_columns,
    )
    assert result.method == "permutation"


@pytest.mark.parametrize("model", ["random_forest", "logistic_regression"])
def test_real_driver_outranks_noise(trained, model):
    """The point of the whole module: importances must reflect reality."""
    result = explain(
        trained.fitted_pipelines[model], trained.X_test, trained.y_test,
        model, trained.numeric_columns, trained.categorical_columns,
    )
    ranks = {f.feature: f.rank for f in result.features}
    assert ranks["driver"] == 1
    assert ranks["driver"] < ranks["noise"]


def test_one_hot_columns_fold_into_source_column(trained):
    result = explain(
        trained.fitted_pipelines["random_forest"], trained.X_test, trained.y_test,
        "random_forest", trained.numeric_columns, trained.categorical_columns,
    )
    names = {f.feature for f in result.features}
    assert "category" in names            # aggregated
    assert not any("cat__" in n for n in names)   # raw encoded names not surfaced
    category = next(f for f in result.features if f.feature == "category")
    assert category.encoded_parts == 3    # X, Y, Z folded together


def test_ranks_are_dense_and_ordered(trained):
    result = explain(
        trained.fitted_pipelines["random_forest"], trained.X_test, trained.y_test,
        "random_forest", trained.numeric_columns, trained.categorical_columns,
    )
    assert [f.rank for f in result.features] == list(range(1, len(result.features) + 1))
    scores = [f.importance for f in result.features]
    assert scores == sorted(scores, reverse=True)


def test_explanation_is_json_serialisable(trained):
    result = explain(
        trained.fitted_pipelines["random_forest"], trained.X_test, trained.y_test,
        "random_forest", trained.numeric_columns, trained.categorical_columns,
    )
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ("num__age", "age"),
        ("cat__region_North", "region"),
        ("cat__region", "region"),
        # Longest match wins, so a prefix collision resolves correctly.
        ("cat__plan_type_A", "plan_type"),
        ("unprefixed", "unprefixed"),
    ],
)
def test_source_column_mapping(encoded, expected):
    numeric = ["age"]
    categorical = ["region", "plan", "plan_type"]
    assert _source_column(encoded, numeric, categorical) == expected


def test_broken_pipeline_degrades_instead_of_raising(trained):
    """Explainability is a nice-to-have; it must never take the run down."""
    class Broken:
        def __getitem__(self, _key):
            raise RuntimeError("boom")

    result = explain(
        Broken(), trained.X_test, trained.y_test, "x", [], []
    )
    assert isinstance(result, Explanation)
    assert result.method == "unavailable"


# --------------------------------------------------------------------------
# SHAP cost control
# --------------------------------------------------------------------------
def test_shap_row_budget_shrinks_as_the_model_grows():
    """Rows are budgeted against tree size, because that is what costs time.

    Regression: a flat 500-row cap meant a 3.7M-node forest took ~15 minutes.
    The work is CPU-bound, so it starved the event loop and the entire API
    stopped responding -- not just that one request.
    """
    from app.services.explain import (
        MAX_SHAP_ROWS,
        MIN_SHAP_ROWS,
        _shap_row_budget,
    )

    assert _shap_row_budget(0) == MAX_SHAP_ROWS          # not a forest
    assert _shap_row_budget(1_000) == MAX_SHAP_ROWS      # tiny model
    big = _shap_row_budget(400_000)
    assert MIN_SHAP_ROWS <= big < MAX_SHAP_ROWS
    assert _shap_row_budget(50_000_000) == MIN_SHAP_ROWS  # never below the floor


def test_node_counting_handles_non_forests(trained):
    from app.services.explain import _model_nodes

    forest = trained.fitted_pipelines["random_forest"]["model"]
    linear = trained.fitted_pipelines["logistic_regression"]["model"]
    assert _model_nodes(forest) > 0
    assert _model_nodes(linear) == 0


def test_depth_is_capped(trained):
    """Unbounded depth overfits and makes SHAP unusable."""
    from app.services.training import MAX_TREE_DEPTH

    forest = trained.fitted_pipelines["random_forest"]["model"]
    assert max(t.tree_.max_depth for t in forest.estimators_) <= MAX_TREE_DEPTH


def test_oversized_model_falls_back_to_permutation(trained, monkeypatch):
    import app.services.explain as ex

    monkeypatch.setattr(ex, "SHAP_MAX_MODEL_NODES", 1)   # force the guard
    result = ex.explain(
        trained.fitted_pipelines["random_forest"], trained.X_test, trained.y_test,
        "random_forest", trained.numeric_columns, trained.categorical_columns,
    )
    assert result.method == "permutation"
    assert result.features


# --------------------------------------------------------------------------
# Informative missingness
# --------------------------------------------------------------------------
def test_missingness_indicator_is_reported_separately(tmp_path):
    """'The value of X' and 'whether X was recorded' are different facts.

    In the planets dataset 'mass' is 99.7% missing for Transit detections and
    7.8% for Radial Velocity. Median imputation handed the model a near-perfect
    class signal that SHAP then attributed to the column's value.
    """
    import numpy as np
    import pandas as pd

    from app.services.profiling import profile_file
    from app.services.quality import assess
    from app.services.training import train

    rng = np.random.default_rng(51)
    n = 400
    group = rng.choice([0, 1], n)
    measured = np.where(group == 1, rng.normal(50, 8, n), np.nan)  # missing iff group 0
    df = pd.DataFrame({
        "measured": measured,
        "noise": rng.normal(0, 1, n),
        "label": group,
    })
    path = tmp_path / "miss.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    out = train(path, profile, "label", "classification")
    # Explain the forest specifically: SHAP works on encoded features and so
    # sees the indicator, whereas permutation importance shuffles the original
    # column (missingness included) and attributes the whole effect to it.
    result = explain(
        out.fitted_pipelines["random_forest"], out.X_test, out.y_test,
        "random_forest", out.numeric_columns, out.categorical_columns,
    )
    assert result.method == "shap"
    names = [f.feature for f in result.features]
    assert any(n.endswith("(was missing)") for n in names), names

    report = assess(out.to_dict(), result.to_dict())
    check = next(c for c in report.checks if c["name"] == "informative_missingness")
    assert check["passed"] is False
    assert "whether a value was recorded" in check["detail"]


def test_indicator_is_not_merged_into_its_source_column():
    from app.services.explain import _source_column

    assert _source_column("num__missingindicator_mass", ["mass"], []) == "mass (was missing)"
    assert _source_column("num__mass", ["mass"], []) == "mass"


def test_complete_data_produces_no_missingness_flag(tmp_path):
    import numpy as np
    import pandas as pd

    from app.services.profiling import profile_file
    from app.services.quality import assess
    from app.services.training import train

    rng = np.random.default_rng(52)
    n = 300
    driver = rng.normal(0, 1, n)
    df = pd.DataFrame({"driver": driver, "label": (driver > 0).astype(int)})
    path = tmp_path / "complete.csv"
    df.to_csv(path, index=False)
    profile = profile_file(path)

    out = train(path, profile, "label", "classification")
    result = explain(
        out.fitted_pipelines[out.best_model], out.X_test, out.y_test,
        out.best_model, out.numeric_columns, out.categorical_columns,
    )
    report = assess(out.to_dict(), result.to_dict())
    assert not any(c["name"] == "informative_missingness" for c in report.checks)
