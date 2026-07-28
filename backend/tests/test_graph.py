"""Graph tests with a stubbed LLM.

Agent behaviour is testable without API calls, and it should be: these cover
the paths that matter most and are hardest to trigger on demand -- a
hallucinated column name, a dead provider, a failed training step.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.agents import nodes
from app.agents.graph import build_graph
from app.agents.nodes import PlanOutput, ReflectionOutput
from app.services.profiling import profile_file


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({
        "id": [f"R-{i}" for i in range(n)],
        "tenure": rng.integers(1, 60, n),
        "charges": np.round(rng.normal(70, 15, n), 2),
        "plan": rng.choice(["A", "B"], n),
        "churn": rng.choice([0, 1], n),
    })
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)

    # Point the storage root at tmp so storage.resolve() finds the file.
    from app.core.config import settings
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    return path, profile_file(path)


class _Structured:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, _messages):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeLLM:
    """Stands in for a chat model. Never touches the network.

    Schema-aware: the graph asks for a PlanOutput from the planner and a
    ReflectionOutput from the reflection node. Returning the plan for both
    made reflection receive an object with no `exclude_features` field.
    """

    def __init__(self, plan=None, text="A short summary.", plan_error=None,
                 text_error=None, reflection=None):
        self._plan, self._plan_error = plan, plan_error
        self._text, self._text_error = text, text_error
        # Default: accept, so tests unconcerned with reflection are unaffected.
        self._reflection = reflection or ReflectionOutput(
            action="accept", reasoning="Stub accepts by default."
        )

    def with_structured_output(self, schema):
        if schema is ReflectionOutput:
            return _Structured(self._reflection)
        return _Structured(self._plan_error or self._plan)

    async def ainvoke(self, _messages):
        if self._text_error:
            raise self._text_error
        return SimpleNamespace(content=self._text)


def _state(path, profile, **overrides):
    return {
        "run_id": "r1", "project_id": "p1", "dataset_id": "d1",
        "storage_path": path.name, "profile": profile,
        "user_goal": None, "steps": [], **overrides,
    }


async def _run(monkeypatch, llm, path, profile):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    graph = build_graph().compile()
    return await graph.ainvoke(_state(path, profile))


async def test_happy_path(monkeypatch, dataset):
    path, profile = dataset
    llm = FakeLLM(plan=PlanOutput(
        target_column="churn", task_type="classification",
        rationale="Binary column, balanced classes.",
    ))
    final = await _run(monkeypatch, llm, path, profile)

    assert final.get("error") is None
    steps = final["steps"]
    assert steps[0].startswith("clean:")
    assert steps[1] == "planner:ok"
    assert steps[2].startswith("training:ok")
    assert steps[3].startswith("explain:")
    assert steps[4].startswith("reflect:")
    assert steps[-1] == "summary:ok"
    assert final["model_path"] is not None
    assert final["cleaned_path"] is not None
    assert final["plan"]["target_column"] == "churn"
    assert final["training"]["best_model"] in {"logistic_regression", "random_forest"}
    assert len(final["training"]["experiments"]) == 2
    assert final["summary"] == "A short summary."


async def test_hallucinated_column_falls_back(monkeypatch, dataset):
    """A model-invented column must never reach sklearn."""
    path, profile = dataset
    llm = FakeLLM(plan=PlanOutput(
        target_column="customer_lifetime_value_v2",  # does not exist
        task_type="regression", rationale="Made up.",
    ))
    final = await _run(monkeypatch, llm, path, profile)

    assert "planner:corrected" in final["steps"]
    assert final["plan"]["target_column"] in {c["name"] for c in profile["columns"]}
    assert final["training"] is not None  # run still completed


async def test_provider_failure_falls_back_to_heuristic(monkeypatch, dataset):
    """A dead LLM must degrade to the deterministic target, not kill the run."""
    path, profile = dataset
    llm = FakeLLM(plan_error=RuntimeError("401 UNAUTHENTICATED"))
    final = await _run(monkeypatch, llm, path, profile)

    assert "planner:fallback" in final["steps"]
    assert final["training"] is not None
    assert "not produced by the planner" in final["plan"]["data_quality_concerns"][0]


async def test_training_failure_short_circuits(monkeypatch, dataset):
    """On a training error the graph must end, not attempt a summary."""
    path, profile = dataset
    llm = FakeLLM(plan=PlanOutput(
        target_column="plan", task_type="regression",  # text target, invalid
        rationale="Deliberately wrong.",
    ))
    final = await _run(monkeypatch, llm, path, profile)

    assert final["failed_node"] == "training"
    assert "numeric target" in final["error"]
    assert final.get("summary") is None
    assert "summary:ok" not in final["steps"]
    assert not any(st.startswith("explain:") for st in final["steps"])


async def test_summary_failure_does_not_fail_the_run(monkeypatch, dataset):
    """Metrics are the deliverable; a missing narrative is cosmetic."""
    path, profile = dataset
    llm = FakeLLM(
        plan=PlanOutput(target_column="churn", task_type="classification", rationale="x"),
        text_error=RuntimeError("429 rate limited"),
    )
    final = await _run(monkeypatch, llm, path, profile)

    assert "summary:fallback" in final["steps"]
    assert "f1=" in final["summary"]
    assert final.get("error") is None


async def test_profile_summary_is_compact(dataset):
    """The prompt digest must stay small -- free tiers are token-limited."""
    from app.agents.prompting import summarise_profile

    _, profile = dataset
    text = summarise_profile(profile)
    assert "churn" in text and "SUGGESTED TARGETS" in text
    # Roughly 4 chars per token; keep the digest well under 2k tokens.
    assert len(text) < 8000


async def test_summary_facts_quote_indicator_names_intact(monkeypatch, dataset):
    """Regression: the summary reported 'mass' as strongest when the actual
    top feature was 'mass (was missing)'.

    Formatting facts as `mass (was missing) (0.0334)` produced two bracketed
    groups and the model read the first as the column name -- exactly the
    misattribution the indicator was added to prevent.
    """
    from app.agents.nodes import summary_node

    captured: dict = {}

    class Capturing(FakeLLM):
        async def ainvoke(self, messages):
            captured["human"] = messages[-1][1]
            return await super().ainvoke(messages)

    path, profile = dataset
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: Capturing(text="ok"))

    state = {
        "steps": [],
        "plan": {"target_column": "churn", "task_type": "classification",
                 "rationale": "x", "data_quality_concerns": []},
        "training": {
            "target_column": "churn", "task_type": "classification",
            "n_train": 100, "n_test": 25, "features_used": ["a"],
            "features_dropped": [], "best_model": "random_forest",
            "experiments": [{"model_name": "random_forest", "primary_metric": "f1",
                             "primary_metric_value": 0.9,
                             "metrics": {"f1": 0.9}}],
        },
        "explanation": {"method": "shap", "features": [
            {"feature": "mass (was missing)", "importance": 0.0334},
            {"feature": "mass", "importance": 0.0168},
        ]},
    }
    await summary_node(state)

    prompt = captured["human"]
    assert '"mass (was missing)" = 0.0334' in prompt
    assert "Never shorten it to the bare column name" in prompt


async def test_summary_is_told_when_the_target_is_derived(monkeypatch, dataset):
    """Regression: on the bike dataset (cnt = casual + registered) the summary
    reported R2 1.0000 as an achievement while the banner beside it said the
    number was meaningless. The fix reached training, quality, the UI and the
    PDF -- but not the layer that actually speaks."""
    from app.agents.nodes import summary_node

    captured: dict = {}

    class Capturing(FakeLLM):
        async def ainvoke(self, messages):
            captured["system"] = messages[0][1]
            captured["human"] = messages[-1][1]
            return await super().ainvoke(messages)

    path, profile = dataset
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: Capturing(text="ok"))

    await summary_node({
        "steps": [],
        "plan": {"target_column": "cnt", "task_type": "regression",
                 "rationale": "x", "data_quality_concerns": []},
        "training": {
            "target_column": "cnt", "task_type": "regression",
            "n_train": 582, "n_test": 146, "features_used": ["casual"],
            "features_dropped": [], "best_model": "ridge",
            "additive_leakage": {
                "r2": 1.0,
                "reason": "A linear model reconstructs 'cnt' from its own features.",
                "contributors": [{"column": "casual", "coefficient": 1.0}],
            },
            "experiments": [{"model_name": "ridge", "primary_metric": "r2",
                             "primary_metric_value": 1.0, "metrics": {"r2": 1.0}}],
        },
    })

    assert "CRITICAL" in captured["human"]
    assert "arithmetic, not prediction" in captured["human"]
    assert "derived from its own columns" in captured["system"]
