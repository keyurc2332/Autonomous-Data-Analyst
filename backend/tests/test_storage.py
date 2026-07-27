import pytest

from app.services import storage


def test_rejects_unsupported_extension():
    with pytest.raises(storage.UnsupportedFileType):
        storage._safe_suffix("data.xlsx")


def test_rejects_missing_extension():
    with pytest.raises(storage.UnsupportedFileType):
        storage._safe_suffix("data")


def test_accepts_csv_and_tsv():
    assert storage._safe_suffix("a.CSV") == ".csv"
    assert storage._safe_suffix("a.tsv") == ".tsv"


def test_path_traversal_is_blocked():
    """A crafted storage_path must not be able to escape the storage root."""
    with pytest.raises(ValueError, match="escapes the storage directory"):
        storage.resolve("../../etc/passwd")


def test_normal_relative_path_resolves():
    resolved = storage.resolve("datasets/abc/file.csv")
    assert resolved.name == "file.csv"
