"""Node functions. Each takes the state and returns the keys it changes.

Complete except ``themes_and_scores``, ``actions`` and ``compose_brief``, which are still
stubs returning empty values of the right type (``docs/build-plan.md``, step 5). Dependencies
(the model client, the optional embedder and classifier) are keyword arguments bound by
:func:`.build.build_graph`, so they never enter the state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from langgraph.types import interrupt

from review_synthesizer.authenticity import Embedder, Review, ReviewId, detect, rating_summary
from review_synthesizer.authenticity.detector import Classifier

from .approval import all_flags, flag_key, held_ids, parse_decisions, review_request
from .batch import validate_batch
from .config import MAX_PARALLEL_CALLS
from .duplicates import candidate_pairs, claim_flags
from .model import ModelClient
from .prompts import EXTRACT_SYSTEM, JUDGE_SYSTEM, extract_prompt, judge_prompt
from .schemas import ExtractedClaims, PairVerdict
from .state import Brief, BriefState, Claim, PairJudgment, Property

T = TypeVar("T")
R = TypeVar("R")


def load_batch(state: BriefState) -> dict[str, Any]:
    """Validate the input into ``Review`` objects; an invalid batch stops the run."""
    return {"reviews": validate_batch(state.get("raw_reviews", []))}


def authenticity(
    state: BriefState,
    *,
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
) -> dict[str, Any]:
    """Run the detector with whichever optional signals are configured. No model call."""
    return {"detection": detect(state["reviews"], embedder=embedder, classifier=classifier)}


def _property(state: BriefState) -> Property:
    return state.get("property") or Property(name="unnamed property")


def _attempt_all(fn: Callable[[T], R], items: Sequence[T]) -> list[R | Exception]:
    """Run ``fn`` over ``items`` concurrently, in order; a failure is returned, not raised."""

    def attempt(item: T) -> R | Exception:
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded by the caller
            return exc

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        return list(pool.map(attempt, items))


def _extract_one(client: ModelClient, prop: Property, review: Review) -> list[Claim]:
    out = client.structured(
        ExtractedClaims, system=EXTRACT_SYSTEM, prompt=extract_prompt(prop, review)
    )
    return [
        Claim(c.aspect, c.polarity, c.specific, c.statement.strip())
        for c in out.claims
        if c.statement.strip()
    ]


def extract_claims(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """One model call per review: the claims it makes, by aspect and polarity.

    Fails soft: a review whose extraction fails keeps an empty claim list and gets a note,
    and the run continues. Such a review cannot join a claim duplicate group.
    """
    reviews = state["reviews"]
    prop = _property(state)
    results = _attempt_all(lambda r: _extract_one(client, prop, r), reviews)

    claims: dict[ReviewId, list[Claim]] = {}
    notes: list[str] = []
    for review, result in zip(reviews, results, strict=True):
        if isinstance(result, Exception):
            claims[review.id] = []
            notes.append(
                f"Claim extraction failed for review {review.id} ({type(result).__name__}); "
                "it stays in the batch with no claims, so it cannot join a claim duplicate group."
            )
        else:
            claims[review.id] = result
    return {"claims": claims, "notes": notes}


def claim_duplicates(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Join point of the fan out: judge candidate pairs, group those making the same claims.

    A failed judgment is noted and the pair is left unjoined; the run continues.
    """
    if "detection" not in state or "claims" not in state:
        raise RuntimeError("claim_duplicates ran before both parallel branches finished")
    reviews, claims, prop = state["reviews"], state["claims"], _property(state)
    pairs = candidate_pairs(reviews, claims, state["detection"])

    def judge(pair: tuple[Review, Review]) -> PairJudgment:
        a, b = pair
        verdict = client.structured(
            PairVerdict,
            system=JUDGE_SYSTEM,
            prompt=judge_prompt(prop, a, b, claims.get(a.id, []), claims.get(b.id, [])),
            small=True,
        )
        return PairJudgment(a.id, b.id, verdict.same_claims, verdict.reason.strip())

    judgments: list[PairJudgment] = []
    notes: list[str] = []
    for (a, b), result in zip(pairs, _attempt_all(judge, pairs), strict=True):
        if isinstance(result, Exception):
            notes.append(
                f"The claim comparison of reviews {a.id} and {b.id} failed "
                f"({type(result).__name__}); the pair was not joined."
            )
        else:
            judgments.append(result)
    return {
        "claim_judgments": judgments,
        "claim_flags": claim_flags(reviews, judgments, client.config.small),
        "notes": notes,
    }


def _verified_ids(state: BriefState) -> list[ReviewId]:
    held = held_ids(all_flags(state), state.get("decisions") or {})
    return [r.id for r in state["reviews"] if r.id not in held]


def verified_set(state: BriefState) -> dict[str, Any]:
    """Reviews not held by any flag, and the displayed and verified ratings.

    A flag holds its reviews until the operator dismisses it, so before review this is
    the cautious set: every flagged review is out.
    """
    held = held_ids(all_flags(state), state.get("decisions") or {})
    return {
        "verified_ids": _verified_ids(state),
        "rating": rating_summary(state["reviews"], held),
    }


def themes_and_scores(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Ranked themes and the aspect scorecard, from verified reviews only."""
    return {"themes": [], "aspect_scores": []}


def actions(state: BriefState, *, client: ModelClient) -> dict[str, Any]:
    """Ranked action list with impact and effort."""
    return {"actions": []}


def human_review(state: BriefState) -> dict[str, Any]:
    """Pause for the operator to confirm or dismiss every flag not yet decided.

    A LangGraph interrupt: the run stops here with the flags and their reasons, and resumes
    with ``Command(resume={flag_key: True | False, ...})``. A batch with no flags passes
    straight through; there is nothing to ask.
    """
    decisions = dict(state.get("decisions") or {})
    pending = [f for f in all_flags(state) if flag_key(f) not in decisions]
    if pending:
        answer = interrupt(review_request(pending, state))
        decisions.update(parse_decisions(answer, [flag_key(f) for f in pending]))
    return {"decisions": decisions}


def route_after_review(state: BriefState) -> str:
    """Rerun the verified set and synthesis if a dismissal released reviews; else finish."""
    if _verified_ids(state) != state.get("verified_ids"):
        return "verified_set"
    return "compose_brief"


def compose_brief(state: BriefState) -> dict[str, Any]:
    """Assemble the brief as data and Markdown."""
    return {"brief": Brief(data={}, markdown="")}
