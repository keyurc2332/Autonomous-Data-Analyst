"""Cleaning tests. Deterministic -- no LLM."""
import numpy as np
import pandas as pd
import pytest

from app.services.cleaning import clean_dataframe, clean_file


def test_exact_duplicates_removed():
    """Duplicates can land in both train and test, overstating every score."""
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})
    cleaned, report = clean_dataframe(df)

    assert len(cleaned) == 3
    assert report.rows_before == 5 and report.rows_after == 3
    action = next(a for a in report.actions if a.action == "drop_duplicate_rows")
    assert action.rows_affected == 2


def test_placeholder_text_becomes_null():
    """'N/A' as text is learned as a category and hides the true null rate."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],           # keeps rows distinct
        "city": ["Mumbai", "N/A", "Delhi", "unknown", "-", "?"],
    })
    cleaned, report = clean_dataframe(df)

    assert cleaned["city"].isna().sum() == 4
    assert set(cleaned["city"].dropna()) == {"Mumbai", "Delhi"}
    action = next(a for a in report.actions if a.action == "normalise_missing")
    assert action.rows_affected == 4


def test_rows_emptied_by_normalisation_can_become_duplicates():
    """Placeholders are nulled first, so rows that become identical are then
    deduplicated. Documented because the interaction is easy to misread."""
    df = pd.DataFrame({"city": ["Mumbai", "N/A", "unknown", "-"]})
    cleaned, report = clean_dataframe(df)

    assert len(cleaned) == 2            # Mumbai + one null row
    assert [a.action for a in report.actions] == [
        "normalise_missing", "drop_duplicate_rows",
    ]


def test_whitespace_is_trimmed_so_categories_merge():
    df = pd.DataFrame({"sex": [" male", "male ", "female", "  female"]})
    cleaned, report = clean_dataframe(df)

    assert set(cleaned["sex"]) == {"male", "female"}
    assert any(a.action == "trim_whitespace" for a in report.actions)


def test_all_null_columns_dropped():
    df = pd.DataFrame({"keep": [1, 2, 3], "empty": [np.nan] * 3})
    cleaned, report = clean_dataframe(df)

    assert list(cleaned.columns) == ["keep"]
    action = next(a for a in report.actions if a.action == "drop_empty_columns")
    assert action.columns == ["empty"]


def test_ordering_whitespace_then_placeholders_then_empty():
    """' N/A ' must be trimmed before it can be recognised as a placeholder,
    and only then does the column become entirely empty."""
    df = pd.DataFrame({"a": [1, 2], "junk": [" N/A ", "unknown"]})
    cleaned, report = clean_dataframe(df)

    assert "junk" not in cleaned.columns
    order = [a.action for a in report.actions]
    assert order.index("normalise_missing") < order.index("drop_empty_columns")


def test_clean_data_is_left_alone():
    """No action should be reported when there is nothing wrong."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cleaned, report = clean_dataframe(df)

    assert report.actions == []
    assert report.changed is False
    assert cleaned.equals(df)


def test_original_dataframe_is_not_mutated():
    df = pd.DataFrame({"a": [1, 1], "b": [" x", " x"]})
    before = df.copy()
    clean_dataframe(df)
    assert df.equals(before)


def test_report_is_json_serialisable():
    import json

    df = pd.DataFrame({"a": [1, 1, np.nan], "b": [" x", " x", "N/A"]})
    _, report = clean_dataframe(df)
    json.dumps(report.to_dict(), allow_nan=False)


def test_clean_file_writes_a_separate_artifact(tmp_path):
    """The original upload must never be modified."""
    source = tmp_path / "raw.csv"
    source.write_text("a,b\n1, x\n1, x\n2,N/A\n")
    dest = tmp_path / "cleaned" / "out.csv"

    report = clean_file(source, dest)

    assert dest.exists()
    assert source.read_text() == "a,b\n1, x\n1, x\n2,N/A\n"   # untouched
    assert report.rows_before == 3 and report.rows_after == 2


@pytest.mark.parametrize("marker", ["", "NA", "n/a", "NULL", "None", "-", "?", "unknown"])
def test_common_missing_markers_recognised(marker):
    df = pd.DataFrame({"id": [1, 2], "c": ["real", marker]})
    cleaned, _ = clean_dataframe(df)
    assert cleaned["c"].isna().sum() == 1


def test_text_columns_detected_regardless_of_pandas_dtype():
    """Regression: pandas 3.0 defaults text columns to `str`, not `object`.

    A dtype-name check skipped every text column, so cleaning silently did
    nothing. Detection is now by exclusion of numeric/datetime/bool.
    """
    from app.services.cleaning import _is_textlike

    assert _is_textlike(pd.Series(["a", "b"])) is True
    assert _is_textlike(pd.Series(["a", "b"], dtype="string")) is True
    assert _is_textlike(pd.Series(["a", "b"], dtype=object)) is True
    assert _is_textlike(pd.Series([1, 2])) is False
    assert _is_textlike(pd.Series([1.5, 2.5])) is False
    assert _is_textlike(pd.Series([True, False])) is False
    assert _is_textlike(pd.to_datetime(pd.Series(["2024-01-01"]))) is False


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_delimiter_is_sniffed_not_assumed(tmp_path, delimiter):
    """Regression: cleaning read with pandas' default comma separator.

    UCI's bank marketing set is semicolon-delimited. Cleaning collapsed it to a
    single column of quoted text and wrote that out, so every later step failed
    on a file that had parsed perfectly at upload. Cleaning runs first, so it
    has to be at least as careful as profiling.
    """
    source = tmp_path / "delim.csv"
    rows = "\n".join(delimiter.join(str(v) for v in row)
                     for row in [(1, "a", 10), (2, "b", 20), (3, "c", 30)])
    source.write_text(delimiter.join(["id", "label", "value"]) + "\n" + rows + "\n")
    dest = tmp_path / "out.csv"

    report = clean_file(source, dest)

    assert report.columns_before == 3
    assert report.rows_before == 3
    # Output is normalised to comma, so everything downstream sees one format.
    cleaned = pd.read_csv(dest)
    assert list(cleaned.columns) == ["id", "label", "value"]
    assert len(cleaned) == 3


def test_quoted_semicolon_file_survives_cleaning(tmp_path):
    """The exact shape of the bank marketing file: quoted fields, semicolons."""
    source = tmp_path / "bank.csv"
    source.write_text(
        '"age";"job";"y"\n'
        '30;"admin.";"no"\n'
        '45;"technician";"yes"\n'
        '52;"admin.";"no"\n'
    )
    dest = tmp_path / "out.csv"

    clean_file(source, dest)
    cleaned = pd.read_csv(dest)

    assert list(cleaned.columns) == ["age", "job", "y"]
    assert cleaned["age"].tolist() == [30, 45, 52]
    assert set(cleaned["y"]) == {"no", "yes"}
