"""Project CRUD, always scoped to the owning user."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Project, TaskType, User


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


async def delete_project(db: AsyncSession, project: Project) -> None:
    # Datasets, runs and reports cascade at the DB level.
    await db.delete(project)
    await db.commit()
