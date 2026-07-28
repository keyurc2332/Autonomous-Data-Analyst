"""Project CRUD, always scoped to the owning user."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentRun, Dataset, Project, RunStatus, TaskType, User


async def create_project(
    db: AsyncSession, owner: User, name: str, description: str | None = None
) -> Project:
    project = Project(owner_id=owner.id, name=name, description=description,
                      task_type=TaskType.UNKNOWN)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


async def get_project(
    db: AsyncSession, project_id: uuid.UUID, owner_id: uuid.UUID
) -> Project | None:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == owner_id)
    )
    return result.scalar_one_or_none()


async def list_projects(db: AsyncSession, owner_id: uuid.UUID) -> list[Project]:
    result = await db.execute(
        select(Project)
        .where(Project.owner_id == owner_id)
        .order_by(Project.created_at.desc())
    )
    return list(result.scalars().all())


async def list_projects_with_summary(
    db: AsyncSession, owner_id: uuid.UUID
) -> list[dict[str, Any]]:
    """Projects plus enough of their last run to render a card.

    Three aggregate queries rather than one per project: a home screen that
    issues N+1 queries is fine at portfolio scale and wrong as a habit.
    """
    projects = await list_projects(db, owner_id)
    if not projects:
        return []
    ids = [p.id for p in projects]

    dataset_counts = dict(
        (await db.execute(
            select(Dataset.project_id, func.count(Dataset.id))
            .where(Dataset.project_id.in_(ids))
            .group_by(Dataset.project_id)
        )).all()
    )
    row_counts = dict(
        (await db.execute(
            select(Dataset.project_id, func.max(Dataset.n_rows))
            .where(Dataset.project_id.in_(ids))
            .group_by(Dataset.project_id)
        )).all()
    )
    run_counts = dict(
        (await db.execute(
            select(AgentRun.project_id, func.count(AgentRun.id))
            .where(
                AgentRun.project_id.in_(ids),
                AgentRun.agent_name == "analysis_graph",
            )
            .group_by(AgentRun.project_id)
        )).all()
    )

    latest = (await db.execute(
        select(AgentRun)
        .where(
            AgentRun.project_id.in_(ids),
            AgentRun.agent_name == "analysis_graph",
            AgentRun.status == RunStatus.SUCCEEDED,
        )
        .order_by(AgentRun.project_id, AgentRun.created_at.desc())
    )).scalars().all()

    last_run: dict[uuid.UUID, AgentRun] = {}
    for run in latest:                      # ordered desc, so first wins
        last_run.setdefault(run.project_id, run)

    summaries = []
    for project in projects:
        run = last_run.get(project.id)
        payload = (run.output_payload or {}) if run else {}
        quality = payload.get("quality") or {}
        training = payload.get("training") or {}
        summaries.append({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "target_column": project.target_column,
            "task_type": project.task_type,
            "created_at": project.created_at,
            "dataset_count": dataset_counts.get(project.id, 0),
            "run_count": run_counts.get(project.id, 0),
            "row_count": row_counts.get(project.id),
            "last_run_id": run.id if run else None,
            "last_verdict": quality.get("verdict"),
            "last_metric": quality.get("gate_metric"),
            "last_value": quality.get("gate_value"),
            "last_run_at": run.finished_at or run.created_at if run else None,
            "leaked_count": len(training.get("leaked_features") or []),
            "derived_target": bool(training.get("additive_leakage")),
        })
    return summaries


async def delete_project(db: AsyncSession, project: Project) -> None:
    # Datasets, runs and reports cascade at the DB level.
    await db.delete(project)
    await db.commit()
