"""Chat orchestration: history, tool execution, persistence."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat import answer
from app.core.logging import get_logger
from app.db.models import AgentRun, ConversationMessage, Dataset, MessageRole, Project, RunStatus
from app.services import storage

logger = get_logger(__name__)

HISTORY_LIMIT = 50


async def get_history(
    db: AsyncSession, project_id: uuid.UUID
) -> list[ConversationMessage]:
    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.project_id == project_id)
        .order_by(ConversationMessage.created_at.asc())
        .limit(HISTORY_LIMIT)
    )
    return list(result.scalars().all())


async def _latest_successful_run(
    db: AsyncSession, project_id: uuid.UUID
) -> dict[str, Any] | None:
    result = await db.execute(
        select(AgentRun)
        .where(
            AgentRun.project_id == project_id,
            AgentRun.status == RunStatus.SUCCEEDED,
            AgentRun.agent_name == "analysis_graph",
        )
        .order_by(AgentRun.created_at.desc())
        .limit(1)
    )
    run = result.scalar_one_or_none()
    return run.output_payload if run else None


async def ask(
    db: AsyncSession, project: Project, dataset: Dataset, question: str
) -> ConversationMessage:
    """Answer a question and persist both sides of the exchange."""
    history = [(m.role.value, m.content) for m in await get_history(db, project.id)]
    latest_run = await _latest_successful_run(db, project.id)

    db.add(ConversationMessage(
        project_id=project.id, role=MessageRole.USER, content=question,
    ))
    await db.commit()

    # Prefer the cleaned copy when one exists, so answers match what was modelled.
    path = storage.resolve(dataset.storage_path)
    if latest_run and latest_run.get("model_path"):
        cleaned = storage.resolve(f"cleaned/{latest_run['model_path'].split('/')[1]}.csv")
        if cleaned.exists():
            path = cleaned

    try:
        reply, tools_used = await answer(
            question=question,
            dataset_path=path,
            profile=dataset.profile,
            latest_run=latest_run,
            history=history,
        )
    except Exception as exc:
        logger.warning("Chat failed", extra={"error": str(exc)})
        reply = (
            "That question could not be answered right now "
            f"({type(exc).__name__}). Try rephrasing it, or check the model "
            "is reachable at /api/v1/llm/check."
        )
        tools_used = []

    message = ConversationMessage(
        project_id=project.id,
        role=MessageRole.ASSISTANT,
        content=reply,
        metadata_={"tools": tools_used},
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def clear_history(db: AsyncSession, project_id: uuid.UUID) -> None:
    for message in await get_history(db, project_id):
        await db.delete(message)
    await db.commit()
