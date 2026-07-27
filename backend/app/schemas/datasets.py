import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetSummary(BaseModel):
    """Lightweight view. Excludes the profile, which can be large."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    original_filename: str
    size_bytes: int
    n_rows: int | None
    n_columns: int | None
    created_at: datetime


class DatasetRead(DatasetSummary):
    profile: dict[str, Any] | None


class DatasetUploadResponse(BaseModel):
    dataset: DatasetSummary
    deduplicated: bool
    warning_count: int
    error_count: int
    target_candidates: list[dict[str, Any]]


class ColumnPreview(BaseModel):
    name: str
    semantic_type: str
    null_pct: float
    unique_count: int
