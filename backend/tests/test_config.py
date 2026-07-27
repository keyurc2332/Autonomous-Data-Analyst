from app.core.config import Settings


def test_database_url_uses_async_driver():
    s = Settings(POSTGRES_HOST="db", POSTGRES_PORT=5432)
    assert s.database_url.startswith("postgresql+asyncpg://")
    assert "@db:5432/" in s.database_url


def test_sync_url_uses_psycopg_for_alembic():
    assert Settings().sync_database_url.startswith("postgresql+psycopg://")


def test_upload_limit_converts_to_bytes():
    assert Settings(MAX_UPLOAD_MB=10).max_upload_bytes == 10 * 1024 * 1024
