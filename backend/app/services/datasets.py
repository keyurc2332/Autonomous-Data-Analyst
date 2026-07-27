"""Dataset orchestration: store -> parse -> profile -> persist."""
from __future__ import annotations

import uuid

import anyio
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Dataset, Project
from app.services import storage
from app.services.profiling import DatasetParseError, profile_file

logger = get_logger(__name__)


async def create_dataset(
    db: AsyncSession, project: Project, upload: UploadFile
) -> tuple[Dataset, bool]:
    """Ingest an upload. Returns (dataset, was_deduplicated)."""
    stored = await storage.save_upload(upload, project.id)

    # Content-addressed dedupe: re-uploading the same bytes to the same
    # project returns the existing row instead of reprofiling.
    existing = await db.execute(
        select(Dataset).where(
            Dataset.project_id == project.id,
            Dataset.content_hash == stored.sha256,
        )
    )
    duplicate = existing.scalar_one_or_none()
    if duplicate is not None:
        storage.delete(stored.relative_path)
        logger.info("Duplicate upload ignored", extra={"dataset_id": str(duplicate.id)})
        return duplicate, True

    # pandas is synchronous and CPU-bound. Running it directly would block
    # the event loop and stall every other request for the duration.
    try:
        profile = await anyio.to_thread.run_sync(profile_file, stored.path)
    except DatasetParseError:
        storage.delete(stored.relative_path)
        raise

    dataset = Dataset(
        project_id=project.id,
        original_filename=stored.original_filename,
        storage_path=stored.relative_path,
        content_hash=stored.sha256,
        size_bytes=stored.size_bytes,
        n_rows=profile["shape"]["rows"],
        n_columns=profile["shape"]["columns"],
        profile=profile,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    logger.info(
        "Dataset profiled",
        extra={
            "dataset_id": str(dataset.id),
            "rows": dataset.n_rows,
            "columns": dataset.n_columns,
            "warnings": len(profile["warnings"]),
        },
    )
    return dataset, False


async def get_dataset(
    db: AsyncSession, dataset_id: uuid.UUID, owner_id: uuid.UUID
) -> Dataset | None:
    """Fetch a dataset, scoped to its owner so IDs cannot be enumerated."""
    result = await db.execute(
        select(Dataset)
        .join(Project, Dataset.project_id == Project.id)
        .where(Dataset.id == dataset_id, Project.owner_id == owner_id)
    )
    return result.scalar_one_or_none()


async def list_datasets(
    db: AsyncSession, project_id: uuid.UUID, owner_id: uuid.UUID
) -> list[Dataset]:
    result = await db.execute(
        select(Dataset)
        .join(Project, Dataset.project_id == Project.id)
        .where(Dataset.project_id == project_id, Project.owner_id == owner_id)
        .order_by(Dataset.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_dataset(db: AsyncSession, dataset: Dataset) -> None:
    storage.delete(dataset.storage_path)
    await db.delete(dataset)
    await db.commit()
