"""Wire the nodes into a LangGraph ``StateGraph``.

Shape (``docs/build-plan.md``, step 5)::

    load_batch -> authenticity   -\\
               -> extract_claims -+-> claim_duplicates -> verified_set
    -> themes_and_scores -> actions -> human_review -> compose_brief
                                            \\-> verified_set (a dismissal released reviews)

``authenticity`` and ``extract_claims`` run in the same step; ``claim_duplicates`` waits
for both. Everything after it is sequential. ``human_review`` interrupts the run, so a
checkpointer is required whenever the batch has flags.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from review_synthesizer.authenticity import (
    DetectionResult,
    Embedder,
    Flag,
    RatingSummary,
    Review,
)
from review_synthesizer.authenticity.detector import Classifier

from . import nodes
from .model import ModelClient
from .state import (
    Action,
    AspectScore,
    Brief,
    BriefState,
    Claim,
    PairJudgment,
    Property,
    SeriousIssue,
    Theme,
)

STATE_TYPES = (
    Property,
    Review,
    Flag,
    DetectionResult,
    RatingSummary,
    Claim,
    PairJudgment,
    Theme,
    AspectScore,
    SeriousIssue,
    Action,
    Brief,
)
"""Every record type the state holds; the only types a checkpoint may rebuild."""


def state_serializer() -> JsonPlusSerializer:
    """Checkpoint serializer that rebuilds the state's records and nothing else."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in STATE_TYPES]
    )


def memory_checkpointer() -> InMemorySaver:
    """In-process checkpointer for one run (``run_brief``, tests). The API uses a database."""
    return InMemorySaver(serde=state_serializer())


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
    resume; use :func:`memory_checkpointer`, or any saver built with
    :func:`state_serializer`.
    """
    graph = StateGraph(BriefState)
    for name in NODE_NAMES:
        fn = getattr(nodes, name)
        if name in _USES_MODEL:
            fn = partial(fn, client=client)
        elif name == "authenticity":
            fn = partial(fn, embedder=embedder, classifier=classifier)
        elif name == "compose_brief":
            fn = partial(fn, models=client.config, embedder_name=getattr(embedder, "name", None))
        graph.add_node(name, fn)

    graph.add_edge(START, "load_batch")
    graph.add_edge("load_batch", "authenticity")
    graph.add_edge("load_batch", "extract_claims")
    graph.add_edge(["authenticity", "extract_claims"], "claim_duplicates")
    graph.add_edge("claim_duplicates", "verified_set")
    graph.add_edge("verified_set", "themes_and_scores")
    graph.add_edge("themes_and_scores", "actions")
    graph.add_edge("actions", "human_review")
    graph.add_conditional_edges(
        "human_review", nodes.route_after_review, ["verified_set", "compose_brief"]
    )
    graph.add_edge("compose_brief", END)
    return graph.compile(checkpointer=checkpointer)
