"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.router import api_router
from app.core.config import settings
from app.core.llm import configure_tracing
from app.core.logging import configure_logging, get_logger
from app.db.session import AsyncSessionLocal, engine

logger = get_logger(__name__)


async def _fail_orphaned_runs() -> None:
    """Close out runs left RUNNING by a crash or restart.

    A run row is written before the graph executes. If the process dies mid-run
    the row keeps status=running forever, so the UI shows a job that will never
    finish. Nothing can be in-flight at startup, so any survivor is orphaned.
    """
    from sqlalchemy import update

    from app.db.models import AgentRun, RunStatus

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                update(AgentRun)
                .where(AgentRun.status.in_([RunStatus.RUNNING, RunStatus.PENDING]))
                .values(
                    status=RunStatus.FAILED,
                    error="Interrupted: the server restarted while this run was in progress.",
                    finished_at=datetime.now(UTC),
                )
            )
            await db.commit()
            if result.rowcount:
                logger.warning("Closed orphaned runs", extra={"count": result.rowcount})
    except Exception as exc:  # a cold database must not block startup
        logger.warning("Could not reconcile runs", extra={"error": str(exc)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    configure_tracing()
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    await _fail_orphaned_runs()
    logger.info(
        "Starting up",
        extra={
            "environment": settings.ENVIRONMENT,
            "llm_provider": settings.LLM_PROVIDER,
            "version": __version__,
        },
    )
    yield
    await engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak stack traces to clients; log them instead."""
    logger.exception(
        "Unhandled exception", extra={"path": request.url.path, "method": request.method}
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"name": settings.APP_NAME, "version": __version__, "docs": "/docs"}
