"""ORM models for the seven entities in the blueprint.

Design note: dataset *contents* never live in the database, and never live
in LangGraph state either. They live on disk (or object storage) and are
referenced by `Dataset.storage_path`. Graph state carries only IDs and
paths, which keeps checkpoints small and serialisable.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, Timestamps, UUIDPrimaryKey


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Reuse a single Enum type object per PG type. If you construct Enum(...) twice
# for the same `name`, Alembic emits two CREATE TYPE statements and migration fails.
RunStatusType = Enum(RunStatus, name="run_status")
TaskTypeType = Enum(TaskType, name="task_type")
MessageRoleType = Enum(MessageRole, name="message_role")


class User(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    projects: Mapped[list[Project]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Project(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "projects"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    target_column: Mapped[str | None] = mapped_column(String(255))
    task_type: Mapped[TaskType] = mapped_column(TaskTypeType, default=TaskType.UNKNOWN)

    owner: Mapped[User] = relationship(back_populates="projects")
    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Dataset(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "datasets"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(500))
    # Path on disk / object-storage key. Contents are NEVER stored in Postgres.
    storage_path: Mapped[str] = mapped_column(String(1000))
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    n_rows: Mapped[int | None] = mapped_column(Integer)
    n_columns: Mapped[int | None] = mapped_column(Integer)
    # Deterministic profile output: dtypes, null counts, cardinality, outliers.
    profile: Mapped[dict | None] = mapped_column(JSONB)

    project: Mapped[Project] = relationship(back_populates="datasets")


class AgentRun(UUIDPrimaryKey, Timestamps, Base):
    """One execution of the LangGraph pipeline (or a single node within it)."""

    __tablename__ = "agent_runs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    agent_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[RunStatus] = mapped_column(
        RunStatusType, default=RunStatus.PENDING, index=True
    )
    # Correlate with the LangSmith trace for this run.
    langsmith_run_id: Mapped[str | None] = mapped_column(String(100))
    input_payload: Mapped[dict | None] = mapped_column(JSONB)
    output_payload: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped[Project] = relationship(back_populates="agent_runs")
    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_agent_runs_project_status", "project_id", "status"),)


class Experiment(UUIDPrimaryKey, Timestamps, Base):
    """One trained candidate model, so the Evaluation agent can compare them."""

    __tablename__ = "experiments"

    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(120))
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    primary_metric: Mapped[str | None] = mapped_column(String(60))
    primary_metric_value: Mapped[float | None] = mapped_column(Float)
    train_seconds: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(String(1000))
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)

    agent_run: Mapped[AgentRun] = relationship(back_populates="experiments")


class Report(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "reports"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300))
    summary_markdown: Mapped[str | None] = mapped_column(Text)
    pdf_path: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[RunStatus] = mapped_column(RunStatusType, default=RunStatus.PENDING)

    project: Mapped[Project] = relationship(back_populates="reports")


class ConversationMessage(UUIDPrimaryKey, Timestamps, Base):
    """Chat-with-dataset history. Ordered by (project_id, created_at)."""

    __tablename__ = "conversation_messages"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(MessageRoleType)
    content: Mapped[str] = mapped_column(Text)
    # Tool calls / retrieved context attached to this turn.
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    project: Mapped[Project] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_conversation_project_created", "project_id", "created_at"),
    )
