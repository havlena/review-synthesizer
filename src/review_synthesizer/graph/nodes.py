"""Node functions. Each takes the state and returns the keys it changes.

Skeleton: every node returns empty values of the right type, so the wiring, the fan out
and join, and checkpointing can be tested before any node does real work. Nodes that
will call a model take the client as a keyword argument, bound by :func:`.build.build_graph`;
none of them calls it yet.
"""

from __future__ import annotations

from typing import Any

from review_synthesizer.authenticity import DetectionResult, RatingSummary, Review

from .model import ModelClient
from .state import Brief, BriefState


def load_batch(state: BriefState) -> dict[str, Any]:
    """Normalise the input into ``Review`` objects. Validation arrives in the next commit."""
    return {"reviews": [Review.from_dict(r) for r in state.get("raw_reviews", [])]}


def authenticity(state: BriefState) -> dict[str, Any]:
    """Run the detector. Deterministic, no model call."""
    return {"detection": DetectionResult(flags=[], flagged_ids=set(), peak_similarity=0.0)}


def extract_claims(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """One model call per review: the claims it makes, by aspect and polarity."""
    return {"claims": {r.id: [] for r in state["reviews"]}}


def claim_duplicates(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Join point of the fan out: needs both the detector result and the claims."""
    if "detection" not in state or "claims" not in state:
        raise RuntimeError("claim_duplicates ran before both parallel branches finished")
    return {"claim_judgments": [], "claim_flags": []}


def verified_set(state: BriefState) -> dict[str, Any]:
    """Reviews not covered by any flag, and the displayed and verified ratings."""
    ids = [r.id for r in state["reviews"]]
    return {
        "verified_ids": ids,
        "rating": RatingSummary(total=len(ids), removed=0, displayed=0.0, verified=None),
    }


def themes_and_scores(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Ranked themes and the aspect scorecard, from verified reviews only."""
    return {"themes": [], "aspect_scores": []}


def actions(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Ranked action list with impact and effort."""
    return {"actions": []}


def human_review(state: BriefState) -> dict[str, Any]:
    """The operator confirms or dismisses each flag. The interrupt arrives in commit 5."""
    return {"decisions": dict(state.get("decisions", {}))}


def compose_brief(state: BriefState) -> dict[str, Any]:
    """Assemble the brief as data and Markdown."""
    return {"brief": Brief(data={}, markdown="")}
