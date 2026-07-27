import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.models import MessageRole, User
from app.db.session import get_db
from app.services import chat as svc
from app.services import datasets as dataset_svc
from app.services import projects as project_svc

router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    created_at: datetime
    metadata_: dict[str, Any] | None = Field(default=None, alias="metadata")


@router.get("", response_model=list[ChatMessage])
async def history(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if await project_svc.get_project(db, project_id, user.id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return await svc.get_history(db, project_id)


@router.post("", response_model=ChatMessage, status_code=status.HTTP_201_CREATED)
async def ask(
    project_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Ask a question about the dataset.

    The model chooses which tools to call. It cannot execute arbitrary code:
    the tool vocabulary is fixed and every argument is schema-validated.
    """
    project = await project_svc.get_project(db, project_id, user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    datasets = await dataset_svc.list_datasets(db, project_id, user.id)
    if not datasets:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Upload a table before asking questions about it."
        )

    return await svc.ask(db, project, datasets[0], payload.message)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if await project_svc.get_project(db, project_id, user.id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    await svc.clear_history(db, project_id)
