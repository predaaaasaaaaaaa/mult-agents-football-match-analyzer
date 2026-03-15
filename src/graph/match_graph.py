"""
match_graph.py — LangGraph Pipeline Definition
================================================
This wires all nodes into a StateGraph.

THE FLOW:
    START → vision → interpolation → teams → events → analytics → reporting → END

HOW IT WORKS:
    1. Call: graph.invoke(initial_state)
    2. LangGraph runs each node in order
    3. Each node reads from state, does its work, returns updates
    4. LangGraph merges the updates into state
    5. Next node gets the updated state
    6. After the last node, get the final state back

LANGGRAPH CONCEPTS USED HERE:
    StateGraph  — the graph definition, parameterized by state type
    add_node    — register a function as a named node
    add_edge    — connect two nodes (A runs, then B runs)
    START / END — special markers for entry and exit
    compile()   — finalize the graph into an executable
"""

from langgraph.graph import StateGraph, START, END
from src.state.match_state import MatchAnalysisState
from src.graph.nodes import (
    vision_node,
    interpolation_node,
    teams_node,
    events_node,
    analytics_node,
    reporting_node,
)


def build_match_graph():
    """
    Build and compile the full match analysis graph.

    Returns a compiled graph that can run with:
        result = graph.invoke(initial_state)
    """

    # Create the graph with our state schema
    graph = StateGraph(MatchAnalysisState)

    # Register nodes (name → function)
    graph.add_node("vision", vision_node)
    graph.add_node("interpolation", interpolation_node)
    graph.add_node("teams", teams_node)
    graph.add_node("events", events_node)
    graph.add_node("analytics", analytics_node)
    graph.add_node("reporting", reporting_node)

    # Wire the edges (execution order)
    graph.add_edge(START, "vision")
    graph.add_edge("vision", "interpolation")
    graph.add_edge("interpolation", "teams")
    graph.add_edge("teams", "events")
    graph.add_edge("events", "analytics")
    graph.add_edge("analytics", "reporting")
    graph.add_edge("reporting", END)

    # Compile — validates and locks the graph
    compiled = graph.compile()

    print("[MatchGraph] Compiled: START → vision → interpolation → teams → events → analytics → reporting → END")

    return compiled