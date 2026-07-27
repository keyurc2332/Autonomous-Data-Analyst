import os

# Tests must never write to the real LangSmith project. LangGraph traces graph
# execution even when the LLM is stubbed, so one pytest run would bury real
# traces in fixture noise and consume free-tier quota.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.db.session import engine
from app.main import app


@pytest.fixture(scope="session", autouse=True)
async def _dispose_engine():
    """Release pooled connections at the end of the session.

    Without this, the event loop closes while asyncpg connections are still
    checked out, and pytest reports unawaited-coroutine warnings on exit.
    """
    yield
    await engine.dispose()


@pytest.fixture
async def client() -> AsyncClient:
    """HTTP client bound directly to the ASGI app -- no network, no server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
async def _db_probe() -> bool:
    """Probe the dependencies exactly once per session, not per test.

    Probing per test made the skip decision depend on transient connection
    state, which is how four tests skipped while four identical ones ran.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        try:
            resp = await ac.get(f"{settings.API_V1_PREFIX}/ready")
        except Exception:  # noqa: BLE001 -- any transport failure means unavailable
            return False
        return resp.status_code == 200


@pytest.fixture
async def db_ready(_db_probe):
    """Skip cleanly and consistently when Postgres/Redis are not running."""
    if not _db_probe:
        pytest.skip("Postgres or Redis unavailable. Run: docker compose up -d")
