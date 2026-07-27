from app.core.config import settings


async def test_root_returns_metadata(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == settings.APP_NAME


async def test_health_does_not_touch_dependencies(client):
    """Liveness must succeed even when Postgres and Redis are unreachable."""
    resp = await client.get(f"{settings.API_V1_PREFIX}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] in {"google", "groq", "ollama"}


async def test_ready_reports_each_dependency(client):
    resp = await client.get(f"{settings.API_V1_PREFIX}/ready")
    # 200 when the compose stack is up, 503 when it is not. Both are valid;
    # what we assert is that the shape of the answer is correct.
    assert resp.status_code in (200, 503)
    deps = resp.json()["dependencies"]
    assert set(deps) == {"postgres", "redis"}
