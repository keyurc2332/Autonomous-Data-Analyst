from typing import Literal

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    environment: str
    version: str
    llm_provider: str


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    dependencies: dict[str, DependencyStatus] = Field(default_factory=dict)
