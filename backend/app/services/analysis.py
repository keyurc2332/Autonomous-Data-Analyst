"""Runs the analysis graph and persists the outcome.

The graph's own checkpoint is an implementation detail. What the rest of the
system reads is `agent_runs` and `experiments`, written here.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.graph import get_compiled_graph
from app.core.logging import get_logger
from app.db.models import (
    AgentRun,
    Dataset,
    Experiment,
    Project,
    Report,
    RunStatus,
    TaskType,
)

logger = get_logger(__name__)


async def _finalise(db: AsyncSession, run: AgentRun) -> AgentRun:
    """Commit and reload, including the experiments relationship.

    `db.refresh(run)` reloads column attributes but NOT relationships. The
    response model serialises `run.experiments`, which would then trigger a
    lazy load outside the async greenlet context and raise MissingGreenlet.
    Naming the attribute forces it to be loaded eagerly, here, where we are
    still inside a proper async context.
    """
    await db.commit()
    await db.refresh(run, attribute_names=["experiments"])
    return run


async def run_analysis(
    db: AsyncSession,
    project: Project,
    dataset: Dataset,
    user_goal: str | None = None,
) -> AgentRun:
    """Execute the graph synchronously and record everything it produced."""
    run = AgentRun(
        project_id=project.id,
        agent_name="analysis_graph",
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
        input_payload={"dataset_id": str(dataset.id), "user_goal": user_goal},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    state = {
        "run_id": str(run.id),
        "project_id": str(project.id),
        "dataset_id": str(dataset.id),
        "storage_path": dataset.storage_path,
        "profile": dataset.profile or {},
        "user_goal": user_goal,
        "steps": [],
    }

    try:
        final = await get_compiled_graph().ainvoke(
            state, config={"configurable": {"thread_id": str(run.id)}}
        )
    except Exception as exc:
        logger.exception("Graph invocation failed")
        run.status = RunStatus.FAILED
        run.error = f"{type(exc).__name__}: {exc}"
        run.finished_at = datetime.now(UTC)
        return await _finalise(db, run)

    run.finished_at = datetime.now(UTC)
    run.output_payload = {
        "plan": final.get("plan"),
        "summary": final.get("summary"),
        "steps": final.get("steps", []),
        "training": final.get("training"),
        "explanation": final.get("explanation"),
        "model_path": final.get("model_path"),
        "cleaning": final.get("cleaning"),
        "quality": final.get("quality"),
        "reflection": final.get("reflection"),
        "attempts": final.get("attempts", []),
    }

    if final.get("error"):
        run.status = RunStatus.FAILED
        run.error = final["error"]
        return await _finalise(db, run)

    run.status = RunStatus.SUCCEEDED

    training = final.get("training") or {}
    best_name = training.get("best_model")
    for exp in training.get("experiments", []):
        db.add(Experiment(
            agent_run_id=run.id,
            model_name=exp["model_name"],
            hyperparameters=exp["hyperparameters"],
            metrics=exp["metrics"],
            primary_metric=exp["primary_metric"],
            primary_metric_value=exp["primary_metric_value"],
            train_seconds=exp["train_seconds"],
            is_selected=exp["model_name"] == best_name,
            artifact_path=(
                final.get("model_path") if exp["model_name"] == best_name else None
            ),
        ))

    # One child row per training round, linked via parent_run_id. That column
    # has existed since Phase 1 for exactly this: a run tree that mirrors the
    # graph's actual execution, including retries.
    for attempt in final.get("attempts", []):
        db.add(AgentRun(
            project_id=project.id,
            parent_run_id=run.id,
            agent_name=f"training_round_{attempt['round']}",
            status=RunStatus.SUCCEEDED,
            input_payload={
                "target_column": attempt["target_column"],
                "task_type": attempt["task_type"],
                "excluded_features": attempt["excluded_features"],
            },
            output_payload={
                "best_model": attempt["best_model"],
                "primary_metric": attempt["primary_metric"],
                "primary_metric_value": attempt["primary_metric_value"],
            },
        ))

    # Record the chosen target on the project so later runs inherit context.
    plan = final.get("plan") or {}
    if plan.get("target_column"):
        project.target_column = plan["target_column"]
        project.task_type = TaskType(plan["task_type"])

    return await _finalise(db, run)


async def build_and_store_report(
    db: AsyncSession, run: AgentRun, project: Project
) -> tuple[bytes, Report]:
    """Render the run as a PDF, caching the artifact and a Report row."""
    import anyio

    from app.core.config import settings
    from app.services.report import build_report

    experiments = [
        {
            "model_name": e.model_name,
            "primary_metric": e.primary_metric,
            "primary_metric_value": e.primary_metric_value or 0.0,
            "train_seconds": e.train_seconds,
            "is_selected": e.is_selected,
        }
        for e in run.experiments
    ]

    pdf = await anyio.to_thread.run_sync(
        build_report, project.name, run.output_payload or {}, experiments
    )

    relative = f"reports/{run.id}.pdf"
    destination = settings.STORAGE_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pdf)

    payload = run.output_payload or {}
    report = Report(
        project_id=project.id,
        title=f"{project.name} — {(payload.get('plan') or {}).get('target_column', 'analysis')}",
        summary_markdown=payload.get("summary"),
        pdf_path=relative,
        status=RunStatus.SUCCEEDED,
    )
    db.add(report)
    await db.commit()
    logger.info("Report generated", extra={"run_id": str(run.id), "bytes": len(pdf)})
    return pdf, report


async def get_run(
    db: AsyncSession, run_id: uuid.UUID, owner_id: uuid.UUID
) -> AgentRun | None:
    result = await db.execute(
        select(AgentRun)
        .join(Project, AgentRun.project_id == Project.id)
        .where(AgentRun.id == run_id, Project.owner_id == owner_id)
        .options(selectinload(AgentRun.experiments))
    )
    return result.scalar_one_or_none()


async def list_runs(
    db: AsyncSession, project_id: uuid.UUID, owner_id: uuid.UUID
) -> list[AgentRun]:
    result = await db.execute(
        select(AgentRun)
        .join(Project, AgentRun.project_id == Project.id)
        .where(AgentRun.project_id == project_id, Project.owner_id == owner_id)
        .options(selectinload(AgentRun.experiments))
        .order_by(AgentRun.created_at.desc())
    )
    return list(result.scalars().all())
