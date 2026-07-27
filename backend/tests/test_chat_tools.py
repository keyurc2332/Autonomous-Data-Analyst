"""Chat tool tests. Deterministic -- no LLM.

These matter disproportionately: the tools are the only surface where a model
gets to act on user data, so their validation is a security boundary as much
as a correctness one.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.services.chat_tools import execute


@pytest.fixture
def df():
    rng = np.random.default_rng(21)
    n = 200
    return pd.DataFrame({
        "region": rng.choice(["North", "South", "East"], n),
        "revenue": np.round(rng.normal(1000, 200, n), 2),
        "units": rng.integers(1, 40, n),
        "notes": [f"order {i}" for i in range(n)],
        "optional": [np.nan if i % 4 == 0 else "set" for i in range(n)],
    })


def test_overview_lists_every_column(df):
    out = execute(df, "DatasetOverview", {})
    assert out["rows"] == 200
    assert {c["name"] for c in out["columns"]} == set(df.columns)
    types = {c["name"]: c["type"] for c in out["columns"]}
    assert types["revenue"] == "numeric" and types["region"] == "text"


def test_aggregate_mean_by_group(df):
    out = execute(df, "Aggregate", {
        "group_by": "region", "value_column": "revenue", "agg": "mean",
    })
    assert out["metric"] == "mean_revenue"
    assert len(out["groups"]) == 3
    values = [g["mean_revenue"] for g in out["groups"]]
    assert values == sorted(values, reverse=True)


def test_aggregate_rejects_non_numeric_value_column(df):
    out = execute(df, "Aggregate", {
        "group_by": "region", "value_column": "notes", "agg": "mean",
    })
    assert "not numeric" in out["error"]


def test_aggregate_count_needs_no_value_column(df):
    out = execute(df, "Aggregate", {"group_by": "region", "agg": "count"})
    assert sum(g["count"] for g in out["groups"]) == 200


def test_count_rows_with_comparison(df):
    out = execute(df, "CountRows", {
        "column": "units", "operator": ">", "value": "20",
    })
    assert out["matched"] == int((df["units"] > 20).sum())
    assert out["total"] == 200


def test_count_rows_isnull(df):
    out = execute(df, "CountRows", {"column": "optional", "operator": "isnull"})
    assert out["matched"] == 50


def test_count_rows_rejects_non_numeric_value_for_numeric_column(df):
    out = execute(df, "CountRows", {
        "column": "revenue", "operator": ">", "value": "lots",
    })
    assert "not a number" in out["error"]


def test_correlation_requires_numeric_columns(df):
    ok = execute(df, "Correlation", {"column_a": "revenue", "column_b": "units"})
    assert -1 <= ok["pearson_r"] <= 1

    bad = execute(df, "Correlation", {"column_a": "region", "column_b": "revenue"})
    assert "not numeric" in bad["error"]


def test_unknown_column_error_lists_valid_ones(df):
    """The error is written for the model to recover from, not just to fail."""
    out = execute(df, "DescribeColumn", {"column": "profit"})
    assert "no column called 'profit'" in out["error"]
    assert "revenue" in out["error"]


def test_column_names_are_matched_case_insensitively(df):
    """Models routinely lowercase column names."""
    out = execute(df, "DescribeColumn", {"column": "REVENUE"})
    assert out["column"] == "revenue"


def test_unknown_tool_is_refused(df):
    assert "no tool called" in execute(df, "RunPython", {"code": "1+1"})["error"]


def test_invalid_arguments_return_an_error_not_an_exception(df):
    out = execute(df, "TopValues", {"column": "region", "n": 9999})   # exceeds le=50
    assert "error" in out


def test_all_tool_output_is_json_serialisable(df):
    """numpy scalars survive round() and break json.dumps -- a real bug once."""
    for name, args in [
        ("DatasetOverview", {}),
        ("DescribeColumn", {"column": "revenue"}),
        ("TopValues", {"column": "region", "n": 3}),
        ("Aggregate", {"group_by": "region", "value_column": "revenue", "agg": "mean"}),
        ("CountRows", {"column": "units", "operator": ">=", "value": "10"}),
        ("Correlation", {"column_a": "revenue", "column_b": "units"}),
    ]:
        json.dumps(execute(df, name, args), allow_nan=False)


def test_group_results_are_truncated(df):
    """A high-cardinality group-by must not return thousands of rows."""
    wide = pd.DataFrame({"k": [f"k{i}" for i in range(500)], "v": range(500)})
    out = execute(wide, "Aggregate", {"group_by": "k", "agg": "count"})
    assert out["truncated"] is True
    assert len(out["groups"]) == 25
    assert out["total_groups"] == 500
