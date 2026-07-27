"""Deterministic dataset profiling. No LLM involved.

This module is the factual foundation the agents reason over. It must be
boring, fast, and reproducible: the same CSV always yields the same profile.
Agents get *opinions*; this module produces *measurements*.

Everything returned must be JSON-serialisable for the `datasets.profile`
JSONB column. That is harder than it sounds -- see `_py()`.
"""
from __future__ import annotations

import csv
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILE_SCHEMA_VERSION = 1

# Profiling every row of a large file is wasteful; beyond this we sample.
MAX_PROFILE_ROWS = 200_000
# A column with this many distinct values or fewer is treated as categorical.
CATEGORICAL_MAX_UNIQUE = 50
# Only report the strongest correlated pairs, not the whole matrix.
TOP_CORRELATION_PAIRS = 15
CORRELATION_MIN_ABS = 0.30
# Near-unique is enough to call something an identifier; requiring exactly
# 100% distinct means one duplicated row reclassifies a primary key as a
# feature, and it then pollutes the target-candidate list.
IDENTIFIER_UNIQUE_RATIO = 0.98

# Column names that hint at a prediction target.
TARGET_KEYWORDS = frozenset({
    "target", "label", "class", "outcome", "churn", "price", "sales",
    "revenue", "score", "result", "y", "survived", "default", "fraud",
})
# Substring matching is only safe for keywords long enough not to collide.
# "y" as a substring matches any name containing the letter y -- it made
# "monthly_charges" look like a target. Short keywords must match a whole token.
_MIN_SUBSTRING_KEYWORD = 5


class DatasetParseError(ValueError):
    """The file could not be read as tabular data."""


# --------------------------------------------------------------------------
# JSON coercion
# --------------------------------------------------------------------------
def _py(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe Python natives.

    Two traps this defuses:
      1. np.int64 / np.float64 are not JSON-serialisable -- json.dumps raises
         TypeError, and asyncpg rejects them for a JSONB column.
      2. NaN and +/-Inf serialise to the literals `NaN` and `Infinity`, which
         are invalid JSON and which Postgres JSONB refuses outright. They
         must become null.
    """
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.str_, str)):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_py(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
@dataclass
class ParseInfo:
    encoding: str
    delimiter: str
    rows_read: int
    sampled: bool


def _sniff(path: Path) -> tuple[str, str]:
    """Detect encoding and delimiter from the first chunk of the file."""
    raw = path.read_bytes()[:64_000]
    if not raw.strip():
        raise DatasetParseError("File is empty.")

    encoding = "utf-8"
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    except UnicodeDecodeError:
        # latin-1 never fails; it is the pragmatic fallback for legacy exports.
        text = raw.decode("latin-1")
        encoding = "latin-1"

    # Drop a possibly-truncated final line before sniffing.
    sample = "\n".join(text.splitlines()[:-1]) or text
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def load_dataframe(path: Path, max_rows: int = MAX_PROFILE_ROWS) -> tuple[pd.DataFrame, ParseInfo]:
    """Read a delimited text file into a DataFrame, sampling if very large."""
    encoding, delimiter = _sniff(path)

    read_kwargs: dict[str, Any] = {
        "encoding": encoding,
        "sep": delimiter,
        "skipinitialspace": True,
        # Read one extra row so we can tell "exactly at limit" from "truncated".
        "nrows": max_rows + 1,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, **read_kwargs)
    except Exception as exc:  # pandas raises a wide variety here
        raise DatasetParseError(f"Could not parse as tabular data: {exc}") from exc

    if df.empty:
        raise DatasetParseError("File parsed but contains no data rows.")
    if df.columns.empty:
        raise DatasetParseError("File parsed but contains no columns.")

    sampled = len(df) > max_rows
    if sampled:
        df = df.iloc[:max_rows]

    # Normalise blank / duplicate header names so downstream code is safe.
    df.columns = [
        (str(c).strip() or f"unnamed_{i}") for i, c in enumerate(df.columns)
    ]

    return df, ParseInfo(
        encoding=encoding, delimiter=delimiter, rows_read=len(df), sampled=sampled
    )


# --------------------------------------------------------------------------
# Semantic typing
# --------------------------------------------------------------------------
def _infer_semantic_type(series: pd.Series, n_rows: int) -> str:
    """Classify a column beyond its storage dtype.

    pandas tells us `int64` or `object`; agents need to know whether a column
    is a target candidate, an identifier, or free text. That is a different
    question from how the bytes are stored.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    n_unique = non_null.nunique()
    if n_unique == 1:
        return "constant"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        # Integer-valued with exactly as many distinct values as rows is
        # almost always a surrogate key, not a feature.
        if (n_unique / n_rows >= IDENTIFIER_UNIQUE_RATIO
                and pd.api.types.is_integer_dtype(series)):
            return "identifier"
        if n_unique == 2:
            return "binary"
        if n_unique <= CATEGORICAL_MAX_UNIQUE and pd.api.types.is_integer_dtype(series):
            return "categorical_numeric"
        return "numeric"

    # Object / string columns: try datetime, then decide categorical vs text.
    if _looks_like_datetime(non_null):
        return "datetime_string"

    if n_unique / n_rows >= IDENTIFIER_UNIQUE_RATIO:
        return "identifier"
    if n_unique == 2:
        return "binary"
    if n_unique <= CATEGORICAL_MAX_UNIQUE:
        return "categorical"

    avg_len = non_null.astype(str).str.len().mean()
    if avg_len and avg_len > 40:
        return "text"
    return "high_cardinality_categorical"


def _looks_like_datetime(non_null: pd.Series, threshold: float = 0.9) -> bool:
    """True if most values in a sample parse as dates."""
    sample = non_null.head(500).astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
            except (ValueError, TypeError):
                return False
    return bool(parsed.notna().mean() >= threshold)


# --------------------------------------------------------------------------
# Per-column profile
# --------------------------------------------------------------------------
def _profile_column(name: str, series: pd.Series, n_rows: int) -> dict[str, Any]:
    non_null = series.dropna()
    n_null = int(series.isna().sum())
    n_unique = int(non_null.nunique()) if not non_null.empty else 0
    semantic = _infer_semantic_type(series, n_rows)

    profile: dict[str, Any] = {
        "name": name,
        "dtype": str(series.dtype),
        "semantic_type": semantic,
        "null_count": n_null,
        "null_pct": round(n_null / n_rows * 100, 3) if n_rows else 0.0,
        "unique_count": n_unique,
        "unique_pct": round(n_unique / n_rows * 100, 3) if n_rows else 0.0,
        "sample_values": [_py(v) for v in non_null.head(5).tolist()],
    }

    if semantic in {"numeric", "binary", "categorical_numeric", "identifier"} and \
            pd.api.types.is_numeric_dtype(series) and not non_null.empty:
        profile["stats"] = _numeric_stats(non_null)

    if semantic in {"categorical", "binary", "constant", "high_cardinality_categorical",
                    "categorical_numeric"} and not non_null.empty:
        counts = non_null.value_counts().head(10)
        profile["top_values"] = [
            {"value": _py(idx), "count": int(cnt), "pct": round(cnt / n_rows * 100, 3)}
            for idx, cnt in counts.items()
        ]

    if semantic == "datetime" and not non_null.empty:
        profile["stats"] = {"min": _py(non_null.min()), "max": _py(non_null.max())}

    if semantic == "datetime_string" and not non_null.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            except (ValueError, TypeError):
                parsed = pd.to_datetime(non_null, errors="coerce")
        valid = parsed.dropna()
        if not valid.empty:
            profile["stats"] = {
                "min": _py(valid.min()),
                "max": _py(valid.max()),
                "span_days": _py((valid.max() - valid.min()).days),
                "unparseable_count": int(parsed.isna().sum()),
            }

    if semantic == "text" and not non_null.empty:
        lengths = non_null.astype(str).str.len()
        profile["stats"] = {
            "min_length": _py(lengths.min()),
            "max_length": _py(lengths.max()),
            "mean_length": _py(round(float(lengths.mean()), 2)),
        }

    return profile


def _numeric_stats(non_null: pd.Series) -> dict[str, Any]:
    """Descriptive stats plus IQR-based outlier counts."""
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((non_null < lower) | (non_null > upper)).sum()) if iqr > 0 else 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skew = non_null.skew()

    return {
        "min": _py(non_null.min()),
        "max": _py(non_null.max()),
        "mean": _py(round(float(non_null.mean()), 6)),
        "median": _py(non_null.median()),
        "std": _py(round(float(non_null.std()), 6)) if len(non_null) > 1 else None,
        "q1": _py(q1),
        "q3": _py(q3),
        "iqr": _py(iqr),
        "skew": _py(round(float(skew), 4)) if pd.notna(skew) else None,
        "zero_count": int((non_null == 0).sum()),
        "negative_count": int((non_null < 0).sum()),
        "outlier_count": outliers,
        "outlier_pct": round(outliers / len(non_null) * 100, 3),
        "outlier_bounds": {"lower": _py(lower), "upper": _py(upper)},
    }


# --------------------------------------------------------------------------
# Correlations
# --------------------------------------------------------------------------
def _correlations(df: pd.DataFrame, columns: list[dict[str, Any]]) -> dict[str, Any]:
    """Strongest Pearson pairs among genuine numeric columns.

    Identifier columns are excluded: a surrogate key correlating with row
    order is noise, and it crowds out real signal in the top-N list.
    """
    usable = [
        c["name"] for c in columns
        if c["semantic_type"] in {"numeric", "binary", "categorical_numeric"}
    ]
    if len(usable) < 2:
        return {"method": "pearson", "pairs": [], "note": "Fewer than two numeric columns."}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = df[usable].corr(method="pearson", numeric_only=True)

    pairs = []
    cols = list(matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = matrix.loc[a, b]
            if pd.isna(r) or abs(float(r)) < CORRELATION_MIN_ABS:
                continue
            pairs.append({"a": a, "b": b, "r": round(float(r), 4)})

    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return {
        "method": "pearson",
        "threshold": CORRELATION_MIN_ABS,
        "pairs": pairs[:TOP_CORRELATION_PAIRS],
    }


# --------------------------------------------------------------------------
# Warnings and target suggestions
# --------------------------------------------------------------------------
def _build_warnings(
    df: pd.DataFrame, columns: list[dict[str, Any]], n_rows: int, corr: dict[str, Any]
) -> list[dict[str, Any]]:
    """Data-quality findings, in a shape the Cleaning agent can act on."""
    out: list[dict[str, Any]] = []

    def add(code: str, severity: str, message: str, column: str | None = None, **extra):
        out.append({"code": code, "severity": severity, "column": column,
                    "message": message, **extra})

    dupes = int(df.duplicated().sum())
    if dupes:
        add("duplicate_rows", "warning",
            f"{dupes} fully duplicated rows ({dupes / n_rows * 100:.1f}%).",
            count=dupes)

    for c in columns:
        name, sem = c["name"], c["semantic_type"]
        if sem == "empty":
            add("all_null", "error", "Column is entirely null and carries no information.", name)
        elif c["null_pct"] >= 50:
            add("high_null", "warning",
                f"{c['null_pct']:.1f}% missing -- imputation may be unreliable.", name)
        elif c["null_pct"] > 0:
            add("has_nulls", "info", f"{c['null_pct']:.1f}% missing.", name)

        if sem == "constant":
            add("constant", "warning", "Single distinct value; no predictive value.", name)
        if sem == "identifier":
            add("identifier", "info",
                "Every value distinct -- looks like an ID. Exclude from features.", name)
        if sem == "high_cardinality_categorical":
            add("high_cardinality", "warning",
                f"{c['unique_count']} distinct categories; one-hot encoding will explode.", name)

        stats = c.get("stats") or {}
        if stats.get("outlier_pct", 0) >= 5:
            add("outliers", "info",
                f"{stats['outlier_count']} IQR outliers ({stats['outlier_pct']:.1f}%).", name)
        if stats.get("skew") is not None and abs(stats["skew"]) > 2:
            add("skewed", "info",
                f"Skew {stats['skew']:.2f}; a log or power transform may help.", name)

    for p in corr.get("pairs", []):
        if abs(p["r"]) >= 0.95:
            add("collinear", "warning",
                f"Near-perfect correlation with '{p['b']}' (r={p['r']}). Likely redundant.",
                p["a"], partner=p["b"], r=p["r"])

    if n_rows < 100:
        add("few_rows", "warning",
            f"Only {n_rows} rows; model results will not be trustworthy.")

    return out


def _name_suggests_target(name: str) -> bool:
    """True if a column name hints that it is a prediction target.

    Tokenised rather than plain substring matching: "monthly_charges" split on
    non-alphanumerics gives {"monthly", "charges"}, neither of which is a
    keyword, whereas a substring test matched the "y" keyword inside "monthly".
    """
    lowered = name.lower()
    tokens = {t for t in re.split(r"[^a-z0-9]+", lowered) if t}
    if TARGET_KEYWORDS & tokens:
        return True
    return any(
        k in lowered for k in TARGET_KEYWORDS if len(k) >= _MIN_SUBSTRING_KEYWORD
    )


def _target_candidates(columns: list[dict[str, Any]], n_rows: int) -> list[dict[str, Any]]:
    """Rank plausible prediction targets and infer the task type.

    This is a heuristic shortlist, not a decision. The Planner agent picks
    the target; this just narrows the field so it isn't guessing blind.
    """
    out = []
    for c in columns:
        name, sem = c["name"], c["semantic_type"]
        if sem in {"identifier", "constant", "empty", "text"}:
            continue
        if c["null_pct"] > 20:
            continue

        named_like_target = _name_suggests_target(name)

        if sem == "binary":
            task, score = "classification", 0.9
        elif sem in {"categorical", "categorical_numeric"}:
            task, score = "classification", 0.7
        elif sem == "numeric":
            task, score = "regression", 0.6
        else:
            continue

        if named_like_target:
            score += 0.25
        # A last column is conventionally the target in many public datasets.
        if c is columns[-1]:
            score += 0.1

        out.append({
            "column": name,
            "task_type": task,
            "confidence": round(min(score, 1.0), 2),
            "reason": f"{sem} column with {c['unique_count']} distinct values"
                      + (", name suggests a target" if named_like_target else ""),
        })

    out.sort(key=lambda t: t["confidence"], reverse=True)
    return out[:5]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame, parse: ParseInfo) -> dict[str, Any]:
    """Build the full profile. Pure function: no I/O, no LLM, no randomness."""
    n_rows = len(df)
    columns = [_profile_column(name, df[name], n_rows) for name in df.columns]
    corr = _correlations(df, columns)

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "shape": {"rows": n_rows, "columns": len(df.columns)},
        "parse": {
            "encoding": parse.encoding,
            "delimiter": parse.delimiter,
            "sampled": parse.sampled,
            "rows_profiled": parse.rows_read,
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "columns": columns,
        "correlations": corr,
        "warnings": _build_warnings(df, columns, n_rows, corr),
        "target_candidates": _target_candidates(columns, n_rows),
    }


def profile_file(path: Path, max_rows: int = MAX_PROFILE_ROWS) -> dict[str, Any]:
    """Read a file from disk and profile it. Blocking -- call in a threadpool."""
    df, parse = load_dataframe(path, max_rows=max_rows)
    return profile_dataframe(df, parse)
