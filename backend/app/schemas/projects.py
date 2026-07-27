import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import TaskType


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    target_column: str | None
    task_type: TaskType
    created_at: datetime
