"""Liveness and readiness endpoints.

/health   -> is the process up?  (never touches dependencies)
/ready    -> can it actually serve traffic?  (checks Postgres + Redis)

Keeping these separate matters for container orchestrators: a liveness
probe that queries the database will restart your app during a brief DB
blip, which is exactly the wrong response.
"""
import time

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.schemas.health import DependencyStatus, HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        version=__version__,
        llm_provider=settings.LLM_PROVIDER,
    )


async def _check_postgres(db: AsyncSession) -> DependencyStatus:
    started = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        return DependencyStatus(
            status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        logger.warning("Postgres readiness check failed", extra={"error": str(exc)})
        return DependencyStatus(status="error", detail=type(exc).__name__)


async def _check_redis() -> DependencyStatus:
    started = time.perf_counter()
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(settings.redis_url)
        try:
            await client.ping()
        finally:
            await client.aclose()
        return DependencyStatus(
            status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2)
        )
    except Exception as exc:
        logger.warning("Redis readiness check failed", extra={"error": str(exc)})
        return DependencyStatus(status="error", detail=type(exc).__name__)


@router.get("/ready", response_model=ReadinessResponse)
async def ready(
    response: Response, db: AsyncSession = Depends(get_db)
) -> ReadinessResponse:
    deps = {
        "postgres": await _check_postgres(db),
        "redis": await _check_redis(),
    }
    healthy = all(d.status == "ok" for d in deps.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ok" if healthy else "degraded", dependencies=deps)
