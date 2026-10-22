"""Shared helpers. Imports of the graph extra stay inside fixtures, so the detector tests
run on an install without it."""

import pytest


@pytest.fixture
def run_graph():
    """Run the graph to the end with an in-memory checkpointer.

    At each human review pause, answer with ``decide(request)``; the default confirms every
    flag. Returns ``(final_state, graph, config, requests)``.
    """
    from langgraph.types import Command

    from review_synthesizer.graph import build_graph, memory_checkpointer

    def run(client, inp, *, decide=None, **build_kwargs):
        graph = build_graph(client, checkpointer=memory_checkpointer(), **build_kwargs)
        config = {"configurable": {"thread_id": "test"}}
        requests = []
        out = graph.invoke(inp, config)
        while "__interrupt__" in out:
            request = out["__interrupt__"][0].value
            requests.append(request)
            answer = decide(request) if decide else {f["key"]: True for f in request["flags"]}
            out = graph.invoke(Command(resume=answer), config)
        return out, graph, config, requests

    return run
