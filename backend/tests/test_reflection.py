"""Reflection loop tests.

The failure mode that matters is a loop that never terminates. Every stop
condition gets its own test.
"""

import numpy as np
import pandas as pd
import pytest

from app.agents import nodes
from app.agents.graph import build_graph
from app.agents.nodes import MAX_REFLECTION_ROUNDS, PlanOutput, ReflectionOutput
from app.services.profiling import profile_file
from tests.test_graph import FakeLLM


@pytest.fixture
def weak_dataset(tmp_path, monkeypatch):
    """Pure noise: no model can do better than chance, so reflection fires."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.choice(["X", "Y"], n),
        "label": rng.choice([0, 1], n),   # independent of everything
    })
    path = tmp_path / "w.csv"
    df.to_csv(path, index=False)
    from app.core.config import settings
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    return path, profile_file(path)


class ScriptedLLM(FakeLLM):
    """Returns a queued response per structured call, so a multi-round loop
    can be driven deterministically."""

    def __init__(self, plan, reflections, text="Summary."):
        super().__init__(plan=plan, text=text)
        self._reflections = list(reflections)
        self.reflection_calls = 0

    def with_structured_output(self, schema):
        if schema is ReflectionOutput:
            self.reflection_calls += 1
            idx = min(self.reflection_calls - 1, len(self._reflections) - 1)
            return _Return(self._reflections[idx])
        return super().with_structured_output(schema)


class _Return:
    def __init__(self, value):
        self._value = value

    async def ainvoke(self, _messages):
        return self._value


def _state(path, profile):
    return {
        "run_id": "r1", "project_id": "p1", "dataset_id": "d1",
        "storage_path": path.name, "profile": profile, "user_goal": None,
        "steps": [], "round": 0, "excluded_features": [], "attempts": [],
    }


async def _run(monkeypatch, llm, path, profile):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    return await build_graph().compile().ainvoke(_state(path, profile))


PLAN = PlanOutput(target_column="label", task_type="classification",
                  rationale="Binary target.")


async def test_strong_result_never_calls_the_reflection_llm(monkeypatch, tmp_path):
    """A good run must cost zero extra tokens."""
    rng = np.random.default_rng(2)
    n = 500
    driver = rng.normal(0, 1, n)
    df = pd.DataFrame({"driver": driver, "label": (driver > 0).astype(int)})
    path = tmp_path / "s.csv"
    df.to_csv(path, index=False)
    from app.core.config import settings
    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path)
    profile = profile_file(path)

    llm = ScriptedLLM(PLAN, [ReflectionOutput(action="retry", reasoning="should not run")])
    final = await _run(monkeypatch, llm, path, profile)

    assert llm.reflection_calls == 0
    assert any(s.startswith("reflect:accept") for s in final["steps"])
    assert len(final["attempts"]) == 1


async def test_weak_result_triggers_one_retry(monkeypatch, weak_dataset):
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="Drop inert features.",
                         exclude_features=["b"]),
        ReflectionOutput(action="abandon", reasoning="Data cannot answer this."),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert len(final["attempts"]) == 2
    assert final["attempts"][1]["excluded_features"] == ["b"]
    assert final["attempts"][1]["round"] == 1
    assert final["summary"] is not None


async def test_round_cap_terminates_the_loop(monkeypatch, weak_dataset):
    """An LLM that always says retry must still stop."""
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="again", exclude_features=["b"]),
        ReflectionOutput(action="retry", reasoning="again", exclude_features=["c"]),
        ReflectionOutput(action="retry", reasoning="again", exclude_features=["a"]),
        ReflectionOutput(action="retry", reasoning="again", exclude_features=["a"]),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert len(final["attempts"]) <= MAX_REFLECTION_ROUNDS + 1
    assert final["summary"] is not None
    assert final["steps"][-1] == "summary:ok"


async def test_attempt_records_the_gate_metric(monkeypatch, weak_dataset):
    """The attempts table must show the number decisions were made on.

    Regression: attempts carried only the primary metric (F1) while every
    retry decision was made on the gate metric (ROC-AUC), so a rising gate
    value looked like a falling score.
    """
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [ReflectionOutput(action="abandon", reasoning="done")])
    final = await _run(monkeypatch, llm, path, profile)

    attempt = final["attempts"][0]
    assert attempt["gate_metric"] == "roc_auc"
    assert attempt["primary_metric"] == "f1"
    assert attempt["verdict"] in {"strong", "acceptable", "weak"}


async def test_abandon_goes_straight_to_summary(monkeypatch, weak_dataset):
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="abandon", reasoning="Features carry no signal."),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert len(final["attempts"]) == 1
    assert final["reflection"]["action"] == "abandon"
    assert "reflect:abandon" in final["steps"]


async def test_no_op_retry_is_converted_to_accept(monkeypatch, weak_dataset):
    """A retry that changes nothing would loop at the identical score."""
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="do it again", exclude_features=[]),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert len(final["attempts"]) == 1
    assert final["reflection"]["action"] == "accept"
    assert "no actual change" in final["reflection"]["reasoning"]


async def test_hallucinated_exclusions_are_dropped(monkeypatch, weak_dataset):
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="drop these",
                         exclude_features=["b", "does_not_exist"]),
        ReflectionOutput(action="abandon", reasoning="enough"),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    # The invented column was stripped; the real one survived.
    assert final["excluded_features"] == ["b"]
    # And the second round ran with only the valid exclusion applied.
    assert final["attempts"][1]["excluded_features"] == ["b"]
    assert final["summary"] is not None


async def test_refuses_to_exclude_every_feature(monkeypatch, weak_dataset):
    """Excluding all features would make the next round raise, not retry."""
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="drop everything",
                         exclude_features=["a", "b", "c"]),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert final["reflection"]["action"] == "abandon"
    assert "every remaining feature" in final["reflection"]["reasoning"]
    assert final["summary"] is not None


async def test_reflection_llm_failure_ends_cleanly(monkeypatch, weak_dataset):
    path, profile = weak_dataset

    class Failing(FakeLLM):
        def with_structured_output(self, schema):
            if schema is ReflectionOutput:
                raise RuntimeError("429 rate limited")
            return super().with_structured_output(schema)

    llm = Failing(plan=PLAN, text="Summary.")
    final = await _run(monkeypatch, llm, path, profile)

    assert "reflect:unavailable" in final["steps"]
    assert final["summary"] is not None
    assert final.get("error") is None


def test_reflection_schema_has_no_nullable_fields():
    """Regression: Groq returned BadRequestError on every reflection call.

    `str | None` generates an `anyOf` containing `null`, which some providers
    reject in a tool schema. The failure was silent -- reflection degraded to
    "accept" and the run looked normal. PlanOutput has no optional fields,
    which is why planning kept working and masked it.
    """
    from app.agents.nodes import ReflectionOutput

    schema = ReflectionOutput.model_json_schema()
    for name, spec in schema["properties"].items():
        assert "anyOf" not in spec, f"{name} is nullable and may be rejected"
        assert spec.get("type") != "null", name


async def test_empty_string_sentinels_normalise_to_no_change(monkeypatch, weak_dataset):
    """Empty strings replace None in the schema; they must not be treated as a
    request to change the target to ''."""
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="drop noise",
                         exclude_features=["b"], new_target="", new_task_type=""),
        ReflectionOutput(action="abandon", reasoning="done"),
    ])
    final = await _run(monkeypatch, llm, path, profile)

    assert final["plan"]["target_column"] == "label"     # unchanged
    assert final["excluded_features"] == ["b"]


async def test_garbage_task_type_is_ignored(monkeypatch, weak_dataset):
    path, profile = weak_dataset
    llm = ScriptedLLM(PLAN, [
        ReflectionOutput(action="retry", reasoning="x", exclude_features=["b"],
                         new_target="label", new_task_type="clustering"),
        ReflectionOutput(action="abandon", reasoning="done"),
    ])
    final = await _run(monkeypatch, llm, path, profile)
    assert final["plan"]["task_type"] == "classification"
