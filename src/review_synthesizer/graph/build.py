"""Wire the nodes into a LangGraph ``StateGraph``.

Shape (``docs/build-plan.md``, step 5)::

    load_batch -> authenticity   -\\
               -> extract_claims -+-> claim_duplicates -> verified_set
    -> themes_and_scores -> actions -> human_review -> compose_brief

``authenticity`` and ``extract_claims`` run in the same step; ``claim_duplicates`` waits
for both. Everything after it is sequential. The return path from ``human_review`` to
``verified_set`` (dismissed flags release reviews) arrives with the interrupt.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from review_synthesizer.authenticity import Embedder
from review_synthesizer.authenticity.detector import Classifier

from . import nodes
from .model import ModelClient
from .state import BriefState

NODE_NAMES = (
    "load_batch",
    "authenticity",
    "extract_claims",
    "claim_duplicates",
    "verified_set",
    "themes_and_scores",
    "actions",
    "human_review",
    "compose_brief",
)
"""Nodes in plan order; the README diagram and the tests use the same names."""

_USES_MODEL = {"extract_claims", "claim_duplicates", "themes_and_scores", "actions"}


def build_graph(
    client: ModelClient,
    *,
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    checkpointer: Any = None,
):
    """Compile the graph with ``client`` bound into the nodes that call a model.

    ``embedder`` and ``classifier`` switch on the detector's optional signals, as in
    ``authenticity.detect``. A checkpointer is needed to pause at ``human_review`` and
    resume; pass one (for example LangGraph's ``InMemorySaver``) whenever the run will be
    resumed.
    """
    graph = StateGraph(BriefState)
    for name in NODE_NAMES:
        fn = getattr(nodes, name)
        if name in _USES_MODEL:
            fn = partial(fn, client=client)
        elif name == "authenticity":
            fn = partial(fn, embedder=embedder, classifier=classifier)
        graph.add_node(name, fn)

    graph.add_edge(START, "load_batch")
    graph.add_edge("load_batch", "authenticity")
    graph.add_edge("load_batch", "extract_claims")
    graph.add_edge(["authenticity", "extract_claims"], "claim_duplicates")
    graph.add_edge("claim_duplicates", "verified_set")
    graph.add_edge("verified_set", "themes_and_scores")
    graph.add_edge("themes_and_scores", "actions")
    graph.add_edge("actions", "human_review")
    graph.add_edge("human_review", "compose_brief")
    graph.add_edge("compose_brief", END)
    return graph.compile(checkpointer=checkpointer)
