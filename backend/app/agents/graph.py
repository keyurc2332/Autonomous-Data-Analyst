"""The analysis graph.

                    +--------------- retry ---------------+
                    v                                     |
    clean -> planner -> training -> explain -> reflect ----+
        |          |                     |
        |          |                     +--> summary -> END
        +----------+--> END (on error)

The explain step sits before reflection deliberately. Reflection without
measured importances would be re-planning blind; with them it can see that
four of seven features contribute nothing and act on that.

The retry loop is bounded three ways: a hard round cap, a required
improvement margin, and a guard that rejects revisions which change nothing.
Any of them ends the loop.

Checkpointing uses MemorySaver for now. A durable Postgres checkpointer only
earns its keep once runs are long-lived and resumable, which arrives with the
background worker in Phase 4. Results are already durable: they are written to
`agent_runs` and `experiments` by the analysis service, independently of the
graph's own checkpoint.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    cleaning_node,
    explain_node,
    planner_node,
    reflection_node,
    summary_node,
    training_node,
)
from app.agents.state import AnalysisState


def _route_after_reflection(state: AnalysisState) -> str:
    """Loop back to training only on an explicit, validated retry."""
    reflection = state.get("reflection") or {}
    return "training" if reflection.get("action") == "retry" else "summary"


def _route_after(node: str):
    def router(state: AnalysisState) -> str:
        return "error" if state.get("error") else node
    return router


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("clean", cleaning_node)
    graph.add_node("planner", planner_node)
    graph.add_node("training", training_node)
    graph.add_node("explain", explain_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("summary", summary_node)

    graph.set_entry_point("clean")
    graph.add_edge("clean", "planner")
    graph.add_conditional_edges(
        "planner", _route_after("training"), {"training": "training", "error": END}
    )
    graph.add_conditional_edges(
        "training", _route_after("explain"), {"explain": "explain", "error": END}
    )
    # Explanation failure is never fatal: metrics still stand on their own.
    graph.add_edge("explain", "reflect")
    graph.add_conditional_edges(
        "reflect", _route_after_reflection,
        {"training": "training", "summary": "summary"},
    )
    graph.add_edge("summary", END)
    return graph


@lru_cache
def get_compiled_graph():
    """Compiled once; compilation is not free and the graph is stateless."""
    return build_graph().compile(checkpointer=MemorySaver())
