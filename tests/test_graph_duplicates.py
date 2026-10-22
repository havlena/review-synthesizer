"""claim_duplicates: candidate pairs, the skip rule, grouping, flags and fail soft.

Model judgments come from FakeModelClient; no network, no key.
"""

import json
import re
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.authenticity import DetectionResult, Flag, Review
from review_synthesizer.graph import Claim, FakeModelClient, PairJudgment, Property
from review_synthesizer.graph.duplicates import (
    JUDGE_STYLE,
    MIN_SHARED_SPECIFIC,
    candidate_pairs,
    claim_flags,
    shared_specific,
)
from review_synthesizer.graph.nodes import claim_duplicates
from review_synthesizer.graph.prompts import JUDGE_SYSTEMS
from review_synthesizer.graph.schemas import ClaimOut, ExtractedClaims, PairVerdict

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")

HOT_TUB = Claim("amenities", "positive", True, "rooftop hot tub open late")
ROLLS = Claim("breakfast", "positive", True, "cinnamon rolls baked each morning")
STAFF = Claim("service", "positive", False, "staff were friendly")
FEE = Claim("parking", "negative", True, "parking costs $40 a night")


def mk(id_, text="text"):
    return Review(id=id_, text=f"{text} {id_}", rating=5, posted=date(2026, 3, 1))


def no_flags():
    return DetectionResult(flags=[], flagged_ids=set(), peak_similarity=0.0)


def flag(signal, ids):
    return Flag(signal, list(ids), 0.9, "high", "test flag")


def test_shared_specific_counts_by_aspect_and_polarity_and_ignores_general_claims():
    other_tub = Claim("amenities", "positive", True, "hot tub on the roof")
    assert shared_specific([HOT_TUB, ROLLS, STAFF], [other_tub, ROLLS, STAFF]) == 2
    assert shared_specific([HOT_TUB, HOT_TUB], [other_tub]) == 1
    assert shared_specific([FEE], [Claim("parking", "positive", True, "free parking")]) == 0
    assert shared_specific([STAFF, STAFF], [STAFF, STAFF]) == 0


def test_candidate_rule_counts_shared_specific_claims():
    reviews = [mk(1), mk(2), mk(3), mk(4)]
    claims = {1: [HOT_TUB, ROLLS], 2: [HOT_TUB, ROLLS, FEE], 3: [HOT_TUB, STAFF], 4: [STAFF]}
    at_two = candidate_pairs(reviews, claims, no_flags(), min_shared=2)
    assert [(a.id, b.id) for a, b in at_two] == [(1, 2)]
    # The product default, chosen by the ablation: one shared specific claim.
    assert MIN_SHARED_SPECIFIC == 1
    default = candidate_pairs(reviews, claims, no_flags())
    assert [(a.id, b.id) for a, b in default] == [(1, 2), (1, 3), (2, 3)]


def test_pairs_joined_by_detector_duplicate_signals_are_skipped_but_not_bursts():
    reviews = [mk(1), mk(2), mk(3), mk(4)]
    claims = {i: [HOT_TUB, ROLLS] for i in range(1, 5)}
    detection = DetectionResult(
        flags=[flag("duplicate_phrasing", [2, 1]), flag("rating_burst", [3, 4])],
        flagged_ids={1, 2, 3, 4},
        peak_similarity=0.9,
    )
    pairs = [(a.id, b.id) for a, b in candidate_pairs(reviews, claims, detection)]
    assert (1, 2) not in pairs
    assert pairs == [(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


def test_chained_pairs_form_one_flag_scored_by_the_share_judged_same():
    reviews = [mk(1), mk(2), mk(3), mk(4)]
    judgments = [
        PairJudgment(1, 2, True, "Both describe the late rooftop hot tub and the rolls."),
        PairJudgment(2, 3, True, "Same hot tub and rolls."),
        PairJudgment(1, 3, False, "Review 3 does not mention the hot tub hours."),
        PairJudgment(1, 4, False, "Different claims."),
    ]
    flags = claim_flags(reviews, judgments, "small-model")
    assert len(flags) == 1
    f = flags[0]
    assert f.signal_type == "claim_duplicate"
    assert f.review_ids == [1, 2, 3]
    assert f.score == pytest.approx(2 / 3) and f.confidence == "medium"
    assert f.rationale.startswith(
        "3 reviews make the same specific claims in different words: 2 of 3 pairs"
    )
    assert "Reviews 1 and 2: Both describe the late rooftop hot tub" in f.rationale
    assert f.details["model"] == "small-model"
    assert [(j["a"], j["b"]) for j in f.details["judgments"]] == [(1, 2), (2, 3), (1, 3)]


def test_no_pair_judged_same_means_no_flag():
    reviews = [mk(1), mk(2)]
    assert claim_flags(reviews, [PairJudgment(1, 2, False, "Different.")], "m") == []


def verdict_for(same_pairs):
    def answer(prompt):
        ids = tuple(int(x) for x in re.findall(r'<review id="(\d+)"', prompt))
        return PairVerdict(same_claims=ids in same_pairs, reason=f"Pair {ids}.")

    return answer


def test_node_judges_candidates_with_the_small_model_and_flags_groups():
    reviews = [mk(1), mk(2), mk(3)]
    state = {
        "property": HARBOR,
        "reviews": reviews,
        "claims": {1: [HOT_TUB, ROLLS], 2: [HOT_TUB, ROLLS], 3: [HOT_TUB, ROLLS]},
        "detection": no_flags(),
    }
    client = FakeModelClient({PairVerdict: verdict_for({(1, 3)})})
    out = claim_duplicates(state, client=client)

    assert len(client.calls) == 3
    assert all(c.small and c.system == JUDGE_SYSTEMS[JUDGE_STYLE] for c in client.calls)
    assert "- amenities, positive: rooftop hot tub open late" in client.calls[0].prompt
    assert [f.review_ids for f in out["claim_flags"]] == [[1, 3]]
    assert out["claim_flags"][0].details["model"] == "fake-small"
    assert out["notes"] == []


def test_a_failed_judgment_is_noted_and_the_pair_left_unjoined():
    reviews = [mk(1), mk(2)]
    state = {
        "reviews": reviews,
        "claims": {1: [HOT_TUB, ROLLS], 2: [HOT_TUB, ROLLS]},
        "detection": no_flags(),
    }
    out = claim_duplicates(state, client=FakeModelClient({PairVerdict: RuntimeError("down")}))
    assert out["claim_flags"] == [] and out["claim_judgments"] == []
    assert out["notes"] == [
        "The claim comparison of reviews 1 and 2 failed (RuntimeError); the pair was not joined."
    ]


def test_in_the_graph_the_planted_cluster_pairs_are_not_judged_again(run_graph):
    rows = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]
    same_two = ExtractedClaims(
        claims=[
            ClaimOut(
                aspect="location", polarity="positive", specific=True, statement="harbor view"
            ),
            ClaimOut(aspect="breakfast", polarity="negative", specific=True, statement="cold eggs"),
        ]
    )
    client = FakeModelClient({ExtractedClaims: same_two, PairVerdict: verdict_for(set())})
    out, *_ = run_graph(client, {"property": HARBOR, "raw_reviews": rows})

    judged = [c for c in client.calls if c.schema == "PairVerdict"]
    planted_pairs = 5 * 4 // 2  # reviews 21 to 25, joined by duplicate phrasing
    assert len(judged) == 25 * 24 // 2 - planted_pairs
    assert out["claim_flags"] == []
