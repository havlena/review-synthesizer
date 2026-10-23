"""The aspect scorecard computed from claims: bands, mixed stances, the two review minimum,
'other' excluded, verified reviews only."""

from datetime import date

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.authenticity import Review
from review_synthesizer.graph import Claim
from review_synthesizer.graph.scorecard import compute_scorecard, score_for, stance


def mk(i):
    return Review(id=i, text=f"review {i}", rating=4, posted=date(2026, 3, 1))


def c(aspect, polarity, specific=True):
    return Claim(aspect, polarity, specific, f"{aspect} {polarity}")


@pytest.mark.parametrize(
    ("negative", "mixed", "total", "score"),
    [
        (0, 0, 4, 5),  # share 0
        (1, 0, 4, 4),  # 0.25, the top of the 4 band
        (0, 1, 4, 4),  # 0.125: mixed counts half
        (2, 0, 4, 3),  # 0.5
        (1, 2, 4, 3),  # 0.5 with mixed
        (3, 0, 4, 2),  # 0.75
        (4, 0, 4, 1),  # 1.0
        (2, 1, 3, 1),  # 0.833
        (0, 2, 2, 3),  # all mixed: 0.5
    ],
)
def test_score_bands_follow_the_negative_share(negative, mixed, total, score):
    assert score_for(negative, mixed, total) == score


def test_a_review_with_praise_and_complaint_on_one_aspect_is_mixed():
    assert stance(["positive", "positive"]) == "positive"
    assert stance(["negative"]) == "negative"
    assert stance(["positive", "negative"]) == "mixed"
    assert stance(["mixed"]) == "mixed"


def test_scorecard_counts_stances_needs_two_reviews_and_skips_other():
    verified = [mk(1), mk(2), mk(3), mk(4)]
    claims = {
        1: [c("cleanliness", "positive"), c("noise", "negative"), c("other", "negative")],
        2: [c("cleanliness", "negative"), c("cleanliness", "positive", specific=False)],
        3: [c("cleanliness", "negative"), c("other", "negative")],
        4: [c("breakfast", "positive")],
        9: [c("cleanliness", "negative"), c("noise", "negative")],  # not verified: ignored
    }
    card = compute_scorecard(verified, claims, {"cleanliness": "Mostly complaints."})
    by = {s.aspect: s for s in card}

    assert [s.aspect for s in card] == ["cleanliness", "breakfast", "noise"]
    clean = by["cleanliness"]
    assert (clean.positive, clean.mixed, clean.negative) == (1, 1, 1)
    assert clean.review_count == 3 and clean.review_ids == [1, 2, 3]
    assert clean.score == 3  # (1 + 0.5) / 3 = 0.5
    assert clean.evidence == "Mostly complaints."
    assert by["breakfast"].score is None and by["breakfast"].review_count == 1
    assert by["noise"].score is None  # review 9 is not verified
    assert "other" not in by
