import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas.datasets import DatasetRead, DatasetSummary, DatasetUploadResponse
from app.services import datasets as svc
from app.services import projects as project_svc
from app.services.profiling import DatasetParseError
from app.services.storage import UnsupportedFileType, UploadTooLarge

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])


@router.post("", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a CSV, profile it, and store the result.

    Profiling runs inline. That is fine for files up to a few tens of MB;
    Phase 4 moves it to a background worker alongside model training.
    """
    project = await project_svc.get_project(db, project_id, user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    try:
        dataset, deduplicated = await svc.create_dataset(db, project, file)
    except UnsupportedFileType as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except UploadTooLarge as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc)) from exc
    except DatasetParseError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    profile = dataset.profile or {}
    warnings = profile.get("warnings", [])
    return DatasetUploadResponse(
        dataset=DatasetSummary.model_validate(dataset),
        deduplicated=deduplicated,
        warning_count=sum(1 for w in warnings if w["severity"] == "warning"),
        error_count=sum(1 for w in warnings if w["severity"] == "error"),
        target_candidates=profile.get("target_candidates", []),
    )


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await svc.list_datasets(db, project_id, user.id)


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    project_id: uuid.UUID,
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Full record including the complete profile."""
    dataset = await svc.get_dataset(db, dataset_id, user.id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return dataset


@router.get("/{dataset_id}/profile")
async def get_profile(
    project_id: uuid.UUID,
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Just the profile. This is what the Profiling agent will call."""
    dataset = await svc.get_dataset(db, dataset_id, user.id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return dataset.profile or {}


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    project_id: uuid.UUID,
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    dataset = await svc.get_dataset(db, dataset_id, user.id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    await svc.delete_dataset(db, dataset)
