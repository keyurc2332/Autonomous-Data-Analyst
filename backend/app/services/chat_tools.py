"""Tools the chat agent can call.

Every tool is a plain function over a DataFrame with a validated schema. There
is deliberately no "run this Python" or "run this SQL" tool: a model that can
execute arbitrary code against a user's data is a much larger security problem
than this feature is worth, and a fixed vocabulary of verified operations
answers nearly every real question.

Column names are checked against the actual schema before anything executes,
so a hallucinated column produces a helpful error the model can recover from
rather than an exception.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

MAX_GROUPS_RETURNED = 25
MAX_ROWS_SCANNED = 200_000


class ToolError(ValueError):
    """A tool could not run. The message is written for the model to read."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_cached(path: str, mtime: float, size: int) -> pd.DataFrame:
    """Cached read. mtime and size are cache keys, not arguments."""
    del mtime, size
    return pd.read_csv(path, nrows=MAX_ROWS_SCANNED)


def load(path: Path) -> pd.DataFrame:
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime, stat.st_size)


def _jsonable(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        # np.float64 subclasses float, and round() on it returns np.float64
        # again -- which is not JSON-serialisable. Force a Python float.
        if math.isnan(value) or math.isinf(value):
            return None
        return float(round(value, 6))
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            pass
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def _require_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    # Case-insensitive rescue before giving up; models often lowercase names.
    lowered = {c.lower(): c for c in df.columns}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise ToolError(
        f"There is no column called '{name}'. Available columns: "
        f"{', '.join(df.columns)}"
    )


# --------------------------------------------------------------------------
# Schemas -- these are what the model sees
# --------------------------------------------------------------------------
class DatasetOverview(BaseModel):
    """Shape of the table and the type of every column. Call this first if you
    are unsure what the data contains."""


class DescribeColumn(BaseModel):
    """Summary statistics for one column: nulls, distinct values, and either
    numeric statistics or the most common values."""

    column: str = Field(description="Exact column name.")


class TopValues(BaseModel):
    """The most frequent values in a column, with counts and percentages."""

    column: str
    n: int = Field(default=10, ge=1, le=50)


class Aggregate(BaseModel):
    """Group rows by one column and compute a statistic per group. Use this for
    questions like 'average price by category' or 'how many per region'."""

    group_by: str = Field(description="Column to group by.")
    value_column: str | None = Field(
        default=None, description="Column to aggregate. Omit when agg is 'count'."
    )
    agg: Literal["count", "mean", "sum", "median", "min", "max"] = "count"


class CountRows(BaseModel):
    """Count rows matching a condition. Use for 'how many X are Y' questions."""

    column: str
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "isnull", "notnull"]
    value: str | None = Field(default=None, description="Omit for isnull/notnull.")


class Correlation(BaseModel):
    """Pearson correlation between two numeric columns."""

    column_a: str
    column_b: str


class LatestAnalysis(BaseModel):
    """Results of the most recent modelling run: target, scores, what drove the
    predictions, and the quality verdict. Use this for questions about the
    model rather than the raw data."""


TOOL_SCHEMAS = [
    DatasetOverview, DescribeColumn, TopValues, Aggregate,
    CountRows, Correlation, LatestAnalysis,
]


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame, _: DatasetOverview) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": [
            {
                "name": c,
                "type": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text",
                "nulls": int(df[c].isna().sum()),
                "distinct": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
    }


def describe_column(df: pd.DataFrame, args: DescribeColumn) -> dict[str, Any]:
    name = _require_column(df, args.column)
    series = df[name]
    non_null = series.dropna()
    out: dict[str, Any] = {
        "column": name,
        "nulls": int(series.isna().sum()),
        "null_pct": round(float(series.isna().mean() * 100), 2),
        "distinct": int(non_null.nunique()),
    }
    if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
        out["statistics"] = {
            k: _jsonable(v) for k, v in {
                "min": non_null.min(), "max": non_null.max(),
                "mean": non_null.mean(), "median": non_null.median(),
                "std": non_null.std() if len(non_null) > 1 else None,
            }.items()
        }
    elif not non_null.empty:
        counts = non_null.value_counts().head(10)
        out["most_common"] = [
            {"value": _jsonable(k), "count": int(v)} for k, v in counts.items()
        ]
    return out


def top_values(df: pd.DataFrame, args: TopValues) -> dict[str, Any]:
    name = _require_column(df, args.column)
    counts = df[name].value_counts().head(args.n)
    total = len(df)
    return {
        "column": name,
        "values": [
            {"value": _jsonable(k), "count": int(v),
             "pct": round(v / total * 100, 2)}
            for k, v in counts.items()
        ],
    }


def aggregate(df: pd.DataFrame, args: Aggregate) -> dict[str, Any]:
    group = _require_column(df, args.group_by)

    if args.agg == "count":
        result = df.groupby(group, dropna=False).size().sort_values(ascending=False)
        value_name = "count"
    else:
        if not args.value_column:
            raise ToolError(f"'{args.agg}' needs a value_column to aggregate.")
        value = _require_column(df, args.value_column)
        if not pd.api.types.is_numeric_dtype(df[value]):
            raise ToolError(
                f"'{value}' is not numeric, so '{args.agg}' cannot be computed. "
                "Use agg='count', or pick a numeric column."
            )
        result = (
            df.groupby(group, dropna=False)[value]
            .agg(args.agg)
            .sort_values(ascending=False)
        )
        value_name = f"{args.agg}_{value}"

    truncated = len(result) > MAX_GROUPS_RETURNED
    return {
        "group_by": group,
        "metric": value_name,
        "groups": [
            {"group": _jsonable(k), value_name: _jsonable(v)}
            for k, v in result.head(MAX_GROUPS_RETURNED).items()
        ],
        "truncated": truncated,
        "total_groups": int(len(result)),
    }


def count_rows(df: pd.DataFrame, args: CountRows) -> dict[str, Any]:
    name = _require_column(df, args.column)
    series = df[name]

    if args.operator == "isnull":
        mask = series.isna()
    elif args.operator == "notnull":
        mask = series.notna()
    else:
        if args.value is None:
            raise ToolError(f"Operator '{args.operator}' needs a value.")
        if args.operator == "contains":
            mask = series.astype("string").str.contains(args.value, case=False, na=False)
        else:
            target: Any = args.value
            if pd.api.types.is_numeric_dtype(series):
                try:
                    target = float(args.value)
                except ValueError as exc:
                    raise ToolError(
                        f"'{args.value}' is not a number, but '{name}' is numeric."
                    ) from exc
            ops = {
                "==": series.eq, "!=": series.ne, ">": series.gt,
                "<": series.lt, ">=": series.ge, "<=": series.le,
            }
            mask = ops[args.operator](target)

    matched = int(mask.sum())
    return {
        "column": name,
        "condition": (
            f"{name} {args.operator} "
            f"{args.value if args.value is not None else ''}"
        ).strip(),
        "matched": matched,
        "total": len(df),
        "pct": round(matched / len(df) * 100, 2) if len(df) else 0.0,
    }


def correlation(df: pd.DataFrame, args: Correlation) -> dict[str, Any]:
    a = _require_column(df, args.column_a)
    b = _require_column(df, args.column_b)
    for col in (a, b):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ToolError(
                f"'{col}' is not numeric. Correlation needs two numeric columns; "
                "for a text column try aggregate() instead."
            )
    r = df[[a, b]].corr().iloc[0, 1]
    return {
        "column_a": a, "column_b": b, "pearson_r": _jsonable(r),
        "note": "Correlation is not causation, and it only measures linear association.",
    }


EXECUTORS = {
    "DatasetOverview": (DatasetOverview, dataset_overview),
    "DescribeColumn": (DescribeColumn, describe_column),
    "TopValues": (TopValues, top_values),
    "Aggregate": (Aggregate, aggregate),
    "CountRows": (CountRows, count_rows),
    "Correlation": (Correlation, correlation),
}


def execute(df: pd.DataFrame, name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    """Validate and run one tool call. Errors are returned, not raised, so the
    model can read them and try something else."""
    entry = EXECUTORS.get(name)
    if entry is None:
        return {"error": f"There is no tool called '{name}'."}
    schema, fn = entry
    try:
        return fn(df, schema(**raw_args))
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
