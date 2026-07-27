"""Analysis endpoint tests, driven through the real HTTP layer with a stub LLM.

These exist because the graph tests never touch FastAPI, and the response
serialisation step is where the async-ORM lazy-load bug lived.
"""
import io

import pytest

from app.agents import nodes
from app.agents.nodes import PlanOutput
from app.core.config import settings
from tests.test_graph import FakeLLM

PREFIX = settings.API_V1_PREFIX

CSV = "tenure,charges,plan,churn\n" + "\n".join(
    f"{i},{50 + (i % 40)},{'A' if i % 2 else 'B'},{i % 2}" for i in range(1, 121)
) + "\n"


@pytest.fixture
def stub_llm(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda *a, **k: FakeLLM(
            plan=PlanOutput(
                target_column="churn", task_type="classification",
                rationale="Binary target with balanced classes.",
            ),
            text="Churn was predicted with modest accuracy.",
        ),
    )


@pytest.fixture
async def project_with_dataset(client, db_ready):
    resp = await client.post(f"{PREFIX}/projects", json={"name": "Analysis Test"})
    project = resp.json()
    up = await client.post(
        f"{PREFIX}/projects/{project['id']}/datasets",
        files={"file": ("t.csv", io.BytesIO(CSV.encode()), "text/csv")},
    )
    yield project, up.json()["dataset"]
    await client.delete(f"{PREFIX}/projects/{project['id']}")


async def test_analysis_run_serialises_with_experiments(client, project_with_dataset, stub_llm):
    """Regression: response serialisation must not lazy-load experiments.

    Accessing run.experiments during serialisation triggered a lazy load
    outside the greenlet context -> MissingGreenlet -> 500.
    """
    project, dataset = project_with_dataset
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/analysis",
        json={"dataset_id": dataset["id"], "user_goal": "who will leave"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    assert body["status"] == "succeeded"
    steps = body["output_payload"]["steps"]
    assert steps[0].startswith("clean:")
    assert steps[1] == "planner:ok"
    assert steps[2].startswith("training:ok")
    assert steps[-1] == "summary:ok"
    assert body["output_payload"]["explanation"] is not None
    assert body["output_payload"]["quality"] is not None
    assert len(body["output_payload"]["attempts"]) >= 1
    assert len(body["experiments"]) == 2
    assert sum(1 for e in body["experiments"] if e["is_selected"]) == 1


async def test_failed_run_also_serialises(client, project_with_dataset, monkeypatch):
    """The failure path returns before any experiment exists -- also must not 500."""
    project, dataset = project_with_dataset
    monkeypatch.setattr(
        nodes, "get_llm",
        lambda *a, **k: FakeLLM(plan=PlanOutput(
            target_column="plan", task_type="regression",  # text target: invalid
            rationale="Deliberately wrong.",
        )),
    )
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/analysis",
        json={"dataset_id": dataset["id"]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "failed"
    assert "numeric target" in body["error"]
    assert body["experiments"] == []


async def test_run_is_retrievable_and_listed(client, project_with_dataset, stub_llm):
    project, dataset = project_with_dataset
    created = await client.post(
        f"{PREFIX}/projects/{project['id']}/analysis",
        json={"dataset_id": dataset["id"]},
    )
    run_id = created.json()["id"]

    fetched = await client.get(f"{PREFIX}/analysis/{run_id}")
    assert fetched.status_code == 200
    assert len(fetched.json()["experiments"]) == 2

    listed = await client.get(f"{PREFIX}/projects/{project['id']}/analysis")
    assert listed.status_code == 200
    assert any(r["id"] == run_id for r in listed.json())


async def test_project_records_chosen_target(client, project_with_dataset, stub_llm):
    """A successful run writes the target back so later runs inherit context."""
    project, dataset = project_with_dataset
    await client.post(
        f"{PREFIX}/projects/{project['id']}/analysis",
        json={"dataset_id": dataset["id"]},
    )
    refreshed = await client.get(f"{PREFIX}/projects/{project['id']}")
    assert refreshed.json()["target_column"] == "churn"
    assert refreshed.json()["task_type"] == "classification"


async def test_unknown_dataset_returns_404(client, project_with_dataset, stub_llm):
    project, _ = project_with_dataset
    resp = await client.post(
        f"{PREFIX}/projects/{project['id']}/analysis",
        json={"dataset_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert resp.status_code == 404
