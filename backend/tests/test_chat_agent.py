"""Chat agent loop, with a stubbed LLM.

The loop is where a model gets to act, so the properties worth pinning are:
it terminates, hallucinated tools cannot execute, and tool errors come back to
the model rather than ending the turn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from app.agents import chat as chat_agent
from app.agents.chat import MAX_TOOL_ROUNDS, answer


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(31)
    n = 150
    df = pd.DataFrame({
        "region": rng.choice(["North", "South"], n),
        "revenue": np.round(rng.normal(500, 90, n), 2),
    })
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return path


class ScriptedChatLLM:
    """Emits a queued list of AIMessages, recording what it was asked."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self._last = messages
        if self._script:
            return self._script.pop(0)
        return AIMessage(content="No more script.")


def _tool_call(name, args, tid="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": tid}])


async def test_answers_without_tools_when_none_are_needed(monkeypatch, dataset):
    llm = ScriptedChatLLM([AIMessage(content="Nothing to compute.")])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    reply, used = await answer("hello", dataset)
    assert reply == "Nothing to compute."
    assert used == []


async def test_executes_a_tool_and_feeds_the_result_back(monkeypatch, dataset):
    llm = ScriptedChatLLM([
        _tool_call("Aggregate", {"group_by": "region", "value_column": "revenue",
                                 "agg": "mean"}),
        AIMessage(content="North averages more."),
    ])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    reply, used = await answer("average revenue by region?", dataset)
    assert reply == "North averages more."
    assert [u["tool"] for u in used] == ["Aggregate"]
    # The tool result must have been appended before the second call.
    assert any("mean_revenue" in str(m.content) for m in llm._last)


async def test_tool_error_is_returned_to_the_model_not_raised(monkeypatch, dataset):
    """A bad column should let the model recover, not end the turn."""
    llm = ScriptedChatLLM([
        _tool_call("DescribeColumn", {"column": "profit"}),
        _tool_call("DescribeColumn", {"column": "revenue"}, tid="2"),
        AIMessage(content="Recovered."),
    ])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    reply, used = await answer("describe profit", dataset)
    assert reply == "Recovered."
    assert len(used) == 2


async def test_hallucinated_tool_cannot_execute(monkeypatch, dataset):
    llm = ScriptedChatLLM([
        _tool_call("ExecutePython", {"code": "import os; os.system('rm -rf /')"}),
        AIMessage(content="That tool does not exist."),
    ])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    reply, used = await answer("run some code", dataset)
    assert used[0]["tool"] == "ExecutePython"
    assert "does not exist" in reply
    # The refusal must have been surfaced to the model as a tool result.
    assert any("no tool called" in str(m.content) for m in llm._last)


async def test_loop_is_bounded(monkeypatch, dataset):
    """A model that calls tools forever must still produce an answer."""
    llm = ScriptedChatLLM(
        [_tool_call("DatasetOverview", {}, tid=str(i)) for i in range(20)]
    )
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    reply, used = await answer("loop please", dataset)
    assert len(used) == MAX_TOOL_ROUNDS
    assert isinstance(reply, str)


async def test_latest_analysis_is_served_from_the_run_not_the_csv(monkeypatch, dataset):
    llm = ScriptedChatLLM([
        _tool_call("LatestAnalysis", {}),
        AIMessage(content="The model scored 0.87."),
    ])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)

    run = {
        "training": {"target_column": "churn", "task_type": "classification",
                     "best_model": "random_forest"},
        "quality": {"verdict": "acceptable", "gate_metric": "roc_auc",
                    "gate_value": 0.87, "reasons": []},
        "explanation": {"method": "shap",
                        "features": [{"feature": "tenure", "importance": 0.4}]},
        "summary": "A summary.",
    }
    reply, used = await answer("how did the model do?", dataset, latest_run=run)
    assert used[0]["tool"] == "LatestAnalysis"
    assert any("random_forest" in str(m.content) for m in llm._last)
    assert reply == "The model scored 0.87."


async def test_all_tool_schemas_are_offered_to_the_model(monkeypatch, dataset):
    from app.services.chat_tools import TOOL_SCHEMAS

    llm = ScriptedChatLLM([AIMessage(content="ok")])
    monkeypatch.setattr(chat_agent, "get_llm", lambda *a, **k: llm)
    await answer("hi", dataset)
    assert llm.bound_tools == TOOL_SCHEMAS
