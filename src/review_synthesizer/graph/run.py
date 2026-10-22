"""One call from a batch of reviews to a finished brief: :func:`run_brief`.

The caller supplies ``approve``, the person in the loop: it receives each review request
(every undecided flag with its reason, and the draft rating) and returns a decision per
flag. The graph does not finish without it. :func:`approve_all` exists for the live
evaluation script, which compares against the golden result with every flag confirmed; it
is not a way to skip review in the product.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langgraph.types import Command

from review_synthesizer.authenticity import Embedder, Review
from review_synthesizer.authenticity.detector import Classifier

from .build import build_graph, memory_checkpointer
from .model import AnthropicModelClient, ModelClient
from .state import Brief, Property

Approve = Callable[[dict[str, Any]], Mapping[str, bool]]

MAX_REVIEW_ROUNDS = 3
"""A run asks at most once per flag, so more rounds than this means a bug, not a decision."""


AUTO_APPROVED = "automatic: every flag confirmed without review (evaluation run)"
"""``approved_by`` for runs that use :func:`approve_all`."""


def approve_all(request: dict[str, Any]) -> dict[str, bool]:
    """Confirm every flag. For evaluation runs against the golden result only."""
    return {f["key"]: True for f in request["flags"]}


def _row(review: Review | Mapping[str, Any]) -> Mapping[str, Any]:
    return dataclasses.asdict(review) if isinstance(review, Review) else review


def run_brief(
    reviews: Sequence[Review | Mapping[str, Any]],
    property: Property | str,
    approve: Approve,
    *,
    client: ModelClient | None = None,
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    run_id: str | None = None,
    approved_by: str = "the operator, flag by flag",
) -> Brief:
    """Run the graph on one batch and return the brief once the operator has decided.

    ``client`` defaults to Claude via :class:`AnthropicModelClient` (needs a key); tests pass
    a :class:`FakeModelClient`. ``approved_by`` says who answered ``approve`` and goes into
    the brief's methods note; an evaluation run with :func:`approve_all` must say so there.
    Raises :class:`BatchError` on invalid input.
    """
    prop = property if isinstance(property, Property) else Property(name=property)
    run_id = run_id or uuid.uuid4().hex[:12]
    graph = build_graph(
        client or AnthropicModelClient(),
        embedder=embedder,
        classifier=classifier,
        checkpointer=memory_checkpointer(),
    )
    config = {"configurable": {"thread_id": run_id}}
    inp = {
        "property": prop,
        "run_id": run_id,
        "approved_by": approved_by,
        "raw_reviews": [_row(r) for r in reviews],
    }

    out = graph.invoke(inp, config)
    rounds = 0
    while "__interrupt__" in out:
        rounds += 1
        if rounds > MAX_REVIEW_ROUNDS:
            raise RuntimeError(
                f"Run {run_id} asked for review more than {MAX_REVIEW_ROUNDS} times."
            )
        request = out["__interrupt__"][0].value
        out = graph.invoke(Command(resume=dict(approve(request))), config)
    return out["brief"]
