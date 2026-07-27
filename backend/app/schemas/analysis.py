import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import RunStatus


class AnalysisRequest(BaseModel):
    dataset_id: uuid.UUID
    user_goal: str | None = Field(
        default=None, max_length=1000,
        description="Optional steer, e.g. 'predict which customers will leave'.",
    )


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    metrics: dict[str, Any] | None
    primary_metric: str | None
    primary_metric_value: float | None
    train_seconds: float | None
    is_selected: bool
    artifact_path: str | None


class AttemptRead(BaseModel):
    round: int
    target_column: str
    task_type: str
    excluded_features: list[str]
    best_model: str | None
    primary_metric: str
    primary_metric_value: float


class AnalysisRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    status: RunStatus
    agent_name: str
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    output_payload: dict[str, Any] | None
    experiments: list[ExperimentRead] = []


class LLMCheckResponse(BaseModel):
    provider: str
    model: str
    ok: bool
    reply: str | None = None
    error_type: str | None = None
    error: str | None = None
    latency_ms: float | None = None
