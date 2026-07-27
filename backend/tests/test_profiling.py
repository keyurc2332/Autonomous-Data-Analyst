import json

import numpy as np
import pandas as pd
import pytest

from app.services.profiling import (
    DatasetParseError,
    _py,
    profile_file,
)


@pytest.fixture
def messy_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    income = np.round(rng.lognormal(10.0, 0.5, n), 2)
    df = pd.DataFrame({
        "row_id": range(1, n + 1),
        "age": rng.integers(18, 80, n).astype(float),
        "income": income,
        "income_k": income / 1000,
        "region": rng.choice(["N", "S", "E", "W"], n),
        "joined": pd.date_range("2023-01-01", periods=n).astype(str),
        "country": ["IN"] * n,
        "blank": [np.nan] * n,
        "churn": rng.choice([0, 1], n),
        "mostly_missing": [1.0] * 100 + [np.nan] * (n - 100),
    })
    df.loc[0:29, "age"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return path


def test_numpy_and_nan_are_json_safe():
    """JSONB rejects NaN and asyncpg rejects numpy scalars -- both must coerce."""
    payload = _py({"i": np.int64(3), "f": np.float64("nan"), "inf": np.float64("inf")})
    assert payload == {"i": 3, "f": None, "inf": None}
    json.dumps(payload, allow_nan=False)


def test_whole_profile_is_json_serialisable(messy_csv):
    profile = profile_file(messy_csv)
    json.dumps(profile, allow_nan=False)


def test_semantic_types(messy_csv):
    profile = profile_file(messy_csv)
    types = {c["name"]: c["semantic_type"] for c in profile["columns"]}
    assert types["row_id"] == "identifier"
    assert types["country"] == "constant"
    assert types["blank"] == "empty"
    assert types["churn"] == "binary"
    assert types["region"] == "categorical"
    assert types["joined"] == "datetime_string"


def test_identifier_survives_duplicate_rows(messy_csv):
    """Five duplicated rows must not demote a primary key to a feature."""
    profile = profile_file(messy_csv)
    row_id = next(c for c in profile["columns"] if c["name"] == "row_id")
    assert row_id["semantic_type"] == "identifier"
    assert "row_id" not in {t["column"] for t in profile["target_candidates"]}


def test_collinearity_detected(messy_csv):
    profile = profile_file(messy_csv)
    codes = {(w["code"], w["column"]) for w in profile["warnings"]}
    assert ("collinear", "income") in codes


def test_quality_warnings(messy_csv):
    profile = profile_file(messy_csv)
    codes = {w["code"] for w in profile["warnings"]}
    assert {"duplicate_rows", "constant", "all_null", "high_null"} <= codes


def test_target_candidates_prefer_named_binary(messy_csv):
    profile = profile_file(messy_csv)
    assert profile["target_candidates"][0]["column"] == "churn"
    assert profile["target_candidates"][0]["task_type"] == "classification"


def test_semicolon_delimiter_is_sniffed(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n7;8;9\n")
    profile = profile_file(path)
    assert profile["parse"]["delimiter"] == ";"
    assert profile["shape"]["columns"] == 3


def test_latin1_file_does_not_crash(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name,city\nJos\xe9,M\xe1laga\nAna,Madrid\n".encode("latin-1"))
    profile = profile_file(path)
    assert profile["parse"]["encoding"] == "latin-1"


def test_sampling_flag_set_for_large_files(tmp_path):
    path = tmp_path / "big.csv"
    pd.DataFrame({"x": range(500)}).to_csv(path, index=False)
    profile = profile_file(path, max_rows=100)
    assert profile["parse"]["sampled"] is True
    assert profile["shape"]["rows"] == 100


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetParseError):
        profile_file(path)


def test_header_only_file_rejected(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("a,b,c\n")
    with pytest.raises(DatasetParseError):
        profile_file(path)


def test_blank_column_names_are_replaced(tmp_path):
    """A CSV with an empty header cell must still produce a usable profile."""
    path = tmp_path / "blank_header.csv"
    path.write_text(",b,c\n1,2,3\n4,5,6\n")
    profile = profile_file(path)
    names = [c["name"] for c in profile["columns"]]
    # pandas substitutes "Unnamed: N" on read; our normalisation is a backstop
    # for whitespace-only headers it does not catch.
    assert "" not in names
    assert all(n.strip() for n in names)
    assert len(names) == 3


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("churn", True),
        ("is_churned", True),
        ("customer_churn_flag", True),
        ("target", True),
        ("y", True),                 # a column literally named y
        ("unit_price", True),
        ("satisfaction_score", True),
        ("monthly_charges", False),  # regression: matched "y" inside "monthly"
        ("delivery_days", False),    # ditto
        ("company", False),
        ("city", False),
        ("age", False),
    ],
)
def test_target_keyword_matching_is_token_based(name, expected):
    from app.services.profiling import _name_suggests_target

    assert _name_suggests_target(name) is expected


def test_unrelated_numeric_column_gets_no_name_bonus(tmp_path):
    """A y-containing name must not be scored above a real categorical."""
    path = tmp_path / "bonus.csv"
    rows = "\n".join(f"{i},{i * 3.5},{'A' if i % 2 else 'B'}" for i in range(1, 61))
    path.write_text("monthly_charges,delivery_days,segment\n" + rows + "\n")
    profile = profile_file(path)
    scores = {t["column"]: t["confidence"] for t in profile["target_candidates"]}
    assert scores.get("monthly_charges", 0) <= 0.7
