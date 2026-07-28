"""Deterministic data cleaning.

The profiler already detected duplicate rows, disguised nulls and whitespace;
nothing acted on any of it, so models trained on data the system had already
flagged as dirty. This module closes that gap.

No LLM. Every action here has a correct answer and a stated rule, and each one
is recorded so the report can say exactly what was changed and why. Cleaning
that cannot be explained is indistinguishable from corruption.

Imputation deliberately stays in the sklearn pipeline: it must be fitted on
training data only, or statistics leak from the test set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

# Values that mean "missing" but survive CSV parsing as text. pandas already
# maps a bare NA/NaN/null; these are the ones it leaves as literal strings.
MISSING_MARKERS = {
    "", "-", "--", "?", "n/a", "na", "n.a.", "none", "null", "nil",
    "missing", "unknown", "not available", "#n/a", "\\n", ".",
}


@dataclass
class CleaningAction:
    action: str
    detail: str
    rows_affected: int = 0
    columns: list[str] = field(default_factory=list)


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    actions: list[CleaningAction] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [asdict(a) for a in self.actions]
        payload["changed"] = self.changed
        return payload


def _is_textlike(series: pd.Series) -> bool:
    """Whether a column holds text.

    Checked by exclusion, not by dtype name. pandas 3.0 made `str` the default
    dtype for text where 2.x used `object`, so a name-based test silently
    skipped every text column and cleaning did nothing at all.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


def _strip_whitespace(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Trim padded strings. ' Male' and 'Male' are one category, not two."""
    touched: list[str] = []
    for col in df.columns:
        if not _is_textlike(df[col]):
            continue
        stripped = df[col].astype("string").str.strip()
        original = df[col].astype("string")
        if not stripped.equals(original):
            touched.append(col)
            df[col] = stripped
    if touched:
        report.actions.append(CleaningAction(
            action="trim_whitespace",
            detail="Removed leading and trailing spaces, which would otherwise "
                   "split one category into several.",
            columns=touched,
        ))
    return df


def _normalise_missing(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Turn placeholder text into real nulls.

    A column containing 'N/A' as text is read as a category, so a model learns
    'N/A' as a meaningful value and the null rate is reported as zero.
    """
    touched: dict[str, int] = {}
    for col in df.columns:
        if not _is_textlike(df[col]):
            continue
        lowered = df[col].astype("string").str.strip().str.lower()
        mask = lowered.isin(MISSING_MARKERS) & df[col].notna()
        count = int(mask.sum())
        if count:
            touched[col] = count
            df.loc[mask, col] = pd.NA
    if touched:
        report.actions.append(CleaningAction(
            action="normalise_missing",
            detail="Converted placeholder text (N/A, unknown, -, ?) to real nulls so "
                   "it is not learned as a category.",
            rows_affected=sum(touched.values()),
            columns=sorted(touched),
        ))
    return df


def _drop_duplicate_rows(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove exact duplicates.

    They inflate apparent sample size and, worse, can place identical rows in
    both train and test, which quietly overstates every score.
    """
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        report.actions.append(CleaningAction(
            action="drop_duplicate_rows",
            detail="Identical rows can land in both train and test, which "
                   "overstates every score.",
            rows_affected=duplicates,
        ))
    return df


def _drop_empty_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Remove columns with no values at all."""
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
        report.actions.append(CleaningAction(
            action="drop_empty_columns",
            detail="Columns with no values carry no information.",
            columns=empty,
        ))
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply every rule in order. Pure and deterministic."""
    report = CleaningReport(
        rows_before=len(df), rows_after=len(df),
        columns_before=len(df.columns), columns_after=len(df.columns),
    )
    df = df.copy()

    # Order matters: whitespace is trimmed before placeholders are matched,
    # and placeholders become nulls before empty columns are detected.
    df = _strip_whitespace(df, report)
    df = _normalise_missing(df, report)
    df = _drop_empty_columns(df, report)
    df = _drop_duplicate_rows(df, report)

    report.rows_after = len(df)
    report.columns_after = len(df.columns)
    return df, report


def clean_file(source: Path, destination: Path) -> CleaningReport:
    """Clean a delimited file and write the result. Blocking -- run in a thread.

    The delimiter and encoding are sniffed, not assumed. Reading with pandas'
    default comma separator turned a semicolon-delimited file (UCI's bank
    marketing set) into a single column of quoted text, and since cleaning runs
    before training, every downstream step then failed on a file that had
    parsed perfectly at upload. Output is always written as comma-delimited
    UTF-8, so everything after this point can rely on one format.
    """
    from app.services.profiling import sniff_dialect

    encoding, delimiter = sniff_dialect(source)
    df = pd.read_csv(source, encoding=encoding, sep=delimiter, skipinitialspace=True)
    cleaned, report = clean_dataframe(df)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)
    logger.info(
        "Cleaned dataset",
        extra={"actions": len(report.actions),
               "rows_removed": report.rows_before - report.rows_after},
    )
    return report
