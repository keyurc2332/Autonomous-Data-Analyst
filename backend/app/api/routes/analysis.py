import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.llm import get_llm, message_text
from app.db.models import RunStatus, User
from app.db.session import get_db
from app.schemas.analysis import AnalysisRequest, AnalysisRunRead, LLMCheckResponse
from app.services import analysis as svc
from app.services import datasets as dataset_svc
from app.services import projects as project_svc

router = APIRouter(tags=["analysis"])


@router.get("/llm/check", response_model=LLMCheckResponse, tags=["health"])
async def llm_check():
    """Make one trivial LLM call and report exactly what happened.

    Worth having as an endpoint rather than a script: it isolates provider
    and credential problems from graph problems, so when a run fails you
    know immediately which half to look at.
    """
    model_name = {
        "google": settings.GOOGLE_MODEL,
        "groq": settings.GROQ_MODEL,
        "ollama": settings.OLLAMA_MODEL,
    }.get(settings.LLM_PROVIDER, "unknown")

    started = time.perf_counter()
    try:
        response = await get_llm().ainvoke(
            [("human", "Reply with exactly: OK")]
        )
        return LLMCheckResponse(
            provider=settings.LLM_PROVIDER, model=model_name, ok=True,
            reply=message_text(response)[:200],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    except Exception as exc:
        return LLMCheckResponse(
            provider=settings.LLM_PROVIDER, model=model_name, ok=False,
            error_type=type(exc).__name__, error=str(exc)[:500],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )


@router.post(
    "/projects/{project_id}/analysis",
    response_model=AnalysisRunRead,
    status_code=status.HTTP_201_CREATED,
)
async def start_analysis(
    project_id: uuid.UUID,
    payload: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Run the analysis graph and return the completed run.

    Synchronous for now. Training a couple of models on a small CSV takes a
    few seconds, which an HTTP request tolerates. Phase 4 moves this to the
    arq worker with a job id, because it will not stay small.
    """
    project = await project_svc.get_project(db, project_id, user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    dataset = await dataset_svc.get_dataset(db, payload.dataset_id, user.id)
    if dataset is None or dataset.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    if not dataset.profile:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Dataset has no profile; re-upload it."
        )

    return await svc.run_analysis(db, project, dataset, payload.user_goal)


@router.get("/projects/{project_id}/analysis", response_model=list[AnalysisRunRead])
async def list_runs(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await svc.list_runs(db, project_id, user.id)


@router.get("/analysis/{run_id}/report.pdf", response_class=Response)
async def download_report(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Render the run as an executive PDF report."""
    run = await svc.get_run(db, run_id, user.id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    if run.status != RunStatus.SUCCEEDED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This run did not complete, so there is nothing to report.",
        )

    project = await project_svc.get_project(db, run.project_id, user.id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    pdf, _ = await svc.build_and_store_report(db, run, project)
    filename = f"{project.name.replace(' ', '-').lower()}-analysis.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analysis/{run_id}", response_model=AnalysisRunRead)
async def get_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    run = await svc.get_run(db, run_id, user.id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run
