"""End-to-end API tests. These require the docker compose stack to be up."""
import io

import pytest

from app.core.config import settings

PREFIX = settings.API_V1_PREFIX

CSV = (
    "customer_id,age,income,region,churn\n"
    "1,25,50000,North,0\n"
    "2,34,62000,South,1\n"
    "3,45,71000,East,0\n"
    "4,29,48000,West,1\n"
    "5,52,95000,North,0\n"
    "6,38,58000,South,1\n"
    "7,41,67000,East,0\n"
    "8,33,53000,West,0\n"
)


def _upload(name: str = "customers.csv", content: str = CSV):
    return {"file": (name, io.BytesIO(content.encode()), "text/csv")}


@pytest.fixture
def stub_llm_for_summary(monkeypatch):
    """Keep the summary test off the network."""
    from app.agents import nodes
    from app.agents.nodes import PlanOutput
    from tests.test_graph import FakeLLM

    monkeypatch.setattr(
        nodes, "get_llm",
        lambda *a, **k: FakeLLM(
            plan=PlanOutput(target_column="churn", task_type="classification",
                            rationale="Binary target."),
            text="A short summary.",
        ),
    )


@pytest.fixture
async def project(client, db_ready):
    """Create a project and tear it down afterwards."""
    resp = await client.post(f"{PREFIX}/projects", json={"name": "Test Project"})
    if resp.status_code != 201:
        pytest.skip(f"Database unavailable (got {resp.status_code}); is the stack up?")
    data = resp.json()
    yield data
    await client.delete(f"{PREFIX}/projects/{data['id']}")


async def test_create_and_fetch_project(client, project):
    assert project["name"] == "Test Project"
    assert project["task_type"] == "unknown"

    resp = await client.get(f"{PREFIX}/projects/{project['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == project["id"]


async def test_upload_profiles_the_dataset(client, project):
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/datasets", files=_upload()
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["deduplicated"] is False
    assert body["dataset"]["n_rows"] == 8
    assert body["dataset"]["n_columns"] == 5
    # 'churn' is binary and its name matches a target keyword.
    assert body["target_candidates"][0]["column"] == "churn"


async def test_profile_endpoint_returns_full_analysis(client, project):
    up = await client.post(f"{PREFIX}/projects/{project['id']}/datasets", files=_upload())
    dataset_id = up.json()["dataset"]["id"]

    resp = await client.get(
        f"{PREFIX}/projects/{project['id']}/datasets/{dataset_id}/profile"
    )
    assert resp.status_code == 200
    profile = resp.json()

    assert profile["schema_version"] == 1
    assert profile["shape"] == {"rows": 8, "columns": 5}
    types = {c["name"]: c["semantic_type"] for c in profile["columns"]}
    assert types["customer_id"] == "identifier"
    assert types["region"] == "categorical"
    assert types["churn"] == "binary"


async def test_reupload_same_bytes_is_deduplicated(client, project):
    first = await client.post(f"{PREFIX}/projects/{project['id']}/datasets", files=_upload())
    second = await client.post(f"{PREFIX}/projects/{project['id']}/datasets", files=_upload())

    assert second.status_code == 201
    assert second.json()["deduplicated"] is True
    assert second.json()["dataset"]["id"] == first.json()["dataset"]["id"]

    listing = await client.get(f"{PREFIX}/projects/{project['id']}/datasets")
    assert len(listing.json()) == 1


async def test_rejects_wrong_extension(client, project):
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/datasets",
        files={"file": ("data.xlsx", io.BytesIO(b"junk"), "application/vnd.ms-excel")},
    )
    assert resp.status_code == 415


async def test_rejects_unparseable_content(client, project):
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/datasets",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert resp.status_code in (400, 422)


async def test_unknown_project_returns_404(client, db_ready):
    resp = await client.post(
        f"{PREFIX}/projects/00000000-0000-0000-0000-000000000000/datasets",
        files=_upload(),
    )
    assert resp.status_code == 404


async def test_dataset_delete_removes_it(client, project):
    up = await client.post(f"{PREFIX}/projects/{project['id']}/datasets", files=_upload())
    dataset_id = up.json()["dataset"]["id"]

    assert (await client.delete(
        f"{PREFIX}/projects/{project['id']}/datasets/{dataset_id}")).status_code == 204
    assert (await client.get(
        f"{PREFIX}/projects/{project['id']}/datasets/{dataset_id}")).status_code == 404


async def test_project_list_carries_last_run_summary(client, db_ready, stub_llm_for_summary):
    """The home screen renders verdicts, so the list endpoint must carry them
    without a request per project."""
    created = await client.post(f"{PREFIX}/projects", json={"name": "Summary Test"})
    project = created.json()
    try:
        await client.post(
            f"{PREFIX}/projects/{project['id']}/datasets", files=_upload()
        )
        await client.post(
            f"{PREFIX}/projects/{project['id']}/analysis",
            json={"dataset_id": (await client.get(
                f"{PREFIX}/projects/{project['id']}/datasets")).json()[0]["id"]},
        )

        listing = await client.get(f"{PREFIX}/projects")
        entry = next(p for p in listing.json() if p["id"] == project["id"])

        assert entry["dataset_count"] == 1
        assert entry["run_count"] == 1
        assert entry["row_count"] == 8
        assert entry["last_verdict"] in {"strong", "acceptable", "weak"}
        assert entry["last_metric"] is not None
        assert entry["last_run_id"] is not None
    finally:
        await client.delete(f"{PREFIX}/projects/{project['id']}")


async def test_project_without_runs_has_empty_summary(client, db_ready):
    created = await client.post(f"{PREFIX}/projects", json={"name": "Untouched"})
    project = created.json()
    try:
        listing = await client.get(f"{PREFIX}/projects")
        entry = next(p for p in listing.json() if p["id"] == project["id"])
        assert entry["dataset_count"] == 0
        assert entry["run_count"] == 0
        assert entry["last_verdict"] is None
        assert entry["leaked_count"] == 0
        assert entry["derived_target"] is False
    finally:
        await client.delete(f"{PREFIX}/projects/{project['id']}")
