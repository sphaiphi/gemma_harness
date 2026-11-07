"""LangGraph wiring: sanitize -> think -> work -> verify -> gate -> (work | think | END)."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ..schemas import HarnessState
from ..verification import Verifier
from .gate import Gate
from .nodes import Nodes


def route(state: HarnessState) -> str:
    return state["next_node"]


def build_graph(nodes: Nodes, verifier: Verifier, gate: Gate, checkpointer: Any):
    g = StateGraph(HarnessState)
    g.add_node("sanitize", nodes.sanitize)
    g.add_node("think", nodes.think)
    g.add_node("work", nodes.work)
    g.add_node("verify", verifier.verify)
    g.add_node("gate", gate.decide)
    g.set_entry_point("sanitize")
    g.add_edge("sanitize", "think")
    g.add_conditional_edges("think", route, {"work": "work", END: END})
    g.add_edge("work", "verify")
    g.add_edge("verify", "gate")
    g.add_conditional_edges("gate", route, {"work": "work", "think": "think", END: END})
    return g.compile(checkpointer=checkpointer)