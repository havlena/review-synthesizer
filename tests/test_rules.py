"""Golden and unit tests for the rule based authenticity layer."""

import json
from datetime import date
from pathlib import Path

from review_synthesizer.authenticity import (
    Review,
    find_duplicate_clusters,
    find_rating_bursts,
    jaccard,
    rating_summary,
    tokenize,
)
from review_synthesizer.authenticity.rules import detect, round_half_up

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_light_inn.json"


def load_fixture() -> tuple[list[Review], set[int]]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reviews = [Review.from_dict(r) for r in data["reviews"]]
    planted = {r["id"] for r in data["reviews"] if r["intent"] == "planted"}
    return reviews, planted


def mk(id_, rating, posted, text="some text here"):
    return Review(id=id_, text=text, rating=rating, posted=date.fromisoformat(posted))


# ---------- golden: the capstone demo must reproduce exactly ----------


def test_golden_flags_exactly_the_planted_cluster():
    reviews, planted = load_fixture()
    result = detect(reviews)
    assert result.flagged_ids == planted == {21, 22, 23, 24, 25}


def test_golden_rating_drops_from_3_8_to_3_5():
    reviews, _ = load_fixture()
    result = detect(reviews)
    summary = rating_summary(reviews, result.flagged_ids)
    assert (summary.total, summary.removed) == (25, 5)
    assert summary.displayed == 3.8
    assert summary.verified == 3.5


def test_golden_both_signals_cover_the_same_five_reviews():
    reviews, planted = load_fixture()
    result = detect(reviews)
    by_type = {f.signal_type: f for f in result.flags}
    assert set(by_type) == {"duplicate_phrasing", "rating_burst"}
    assert set(by_type["duplicate_phrasing"].review_ids) == planted
    assert set(by_type["rating_burst"].review_ids) == planted
    assert by_type["rating_burst"].details["rating"] == 5
    assert by_type["rating_burst"].details["span_days"] == 2
    assert 0.55 <= result.peak_similarity <= 1.0
    for f in result.flags:
        assert f.rationale and f.confidence in {"low", "medium", "high"}


def test_golden_no_genuine_review_is_flagged():
    reviews, planted = load_fixture()
    result = detect(reviews)
    genuine = {r.id for r in reviews} - planted
    assert not (result.flagged_ids & genuine)


# ---------- tokenisation and similarity ----------


def test_tokenize_strips_punctuation_case_and_stop_words():
    assert tokenize("The Staff, at the DESK: were helpful!") == frozenset({"staff", "desk", "helpful"})
    assert tokenize("a an the") == frozenset()


def test_jaccard_bounds():
    a, b = tokenize("clean quiet room"), tokenize("clean quiet room")
    assert jaccard(a, b) == 1.0
    assert jaccard(tokenize("harbor view"), tokenize("breakfast coffee")) == 0.0
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_duplicate_threshold_is_inclusive():
    a = mk(1, 5, "2026-01-01", "best hotel amazing staff perfect rooms")
    b = mk(2, 5, "2026-02-01", "best hotel amazing staff perfect rooms five stars")
    sim = jaccard(tokenize(a.text), tokenize(b.text))
    clusters = find_duplicate_clusters([a, b], threshold=sim)
    assert clusters and set(clusters[0].review_ids) == {1, 2}
    assert find_duplicate_clusters([a, b], threshold=sim + 1e-9) == []


def test_duplicate_pairs_join_transitively():
    a = mk(1, 5, "2026-01-01", "alpha beta gamma delta")
    b = mk(2, 5, "2026-01-02", "alpha beta gamma epsilon")
    c = mk(3, 5, "2026-01-03", "beta gamma epsilon zeta")
    clusters = find_duplicate_clusters([a, b, c], threshold=0.6)
    assert len(clusters) == 1 and set(clusters[0].review_ids) == {1, 2, 3}
    assert clusters[0].pair_count == 2


# ---------- rating bursts ----------


def test_burst_window_is_forward_and_inclusive_of_the_fourth_day():
    reviews = [mk(1, 4, "2026-01-01"), mk(2, 4, "2026-01-03"), mk(3, 4, "2026-01-05")]
    bursts = find_rating_bursts(reviews)
    assert len(bursts) == 1 and set(bursts[0].review_ids) == {1, 2, 3}
    reviews[2] = mk(3, 4, "2026-01-06")  # fifth day: outside a 4 day window
    assert find_rating_bursts(reviews) == []


def test_burst_requires_same_rating():
    reviews = [mk(1, 5, "2026-01-01"), mk(2, 4, "2026-01-01"), mk(3, 5, "2026-01-02")]
    assert find_rating_bursts(reviews) == []


def test_burst_reports_one_entry_per_rating():
    reviews = [mk(i, 5, "2026-01-01") for i in range(1, 4)] + [mk(i, 1, "2026-02-01") for i in range(4, 7)]
    bursts = find_rating_bursts(reviews)
    assert [b.rating for b in bursts] == [1, 5]


def test_separate_bursts_of_the_same_rating_are_reported_separately():
    reviews = [mk(i, 5, "2026-01-0%d" % i) for i in range(1, 4)] + [mk(i, 5, "2026-02-0%d" % (i - 3)) for i in range(4, 7)]
    bursts = find_rating_bursts(reviews)
    assert len(bursts) == 2
    assert set(bursts[0].review_ids) == {1, 2, 3} and set(bursts[1].review_ids) == {4, 5, 6}
    assert bursts[0].span_days == 2 and bursts[1].span_days == 2


def test_chained_windows_form_one_burst_and_a_bystander_is_swept_in():
    # Three reviews on days 1, 3, 5 and one on day 8: the window from day 5 holds days 5 and 8
    # only (two reviews), so day 8 stays out; the window from day 3 holds 3, 5 and 7 if a
    # review on day 7 exists, chaining it in even if that review is genuine.
    reviews = [mk(1, 5, "2026-01-01"), mk(2, 5, "2026-01-03"), mk(3, 5, "2026-01-05"), mk(4, 5, "2026-01-07")]
    bursts = find_rating_bursts(reviews)
    assert len(bursts) == 1 and set(bursts[0].review_ids) == {1, 2, 3, 4}
    reviews[3] = mk(4, 5, "2026-01-10")
    bursts = find_rating_bursts(reviews)
    assert len(bursts) == 1 and set(bursts[0].review_ids) == {1, 2, 3}


# ---------- rating summary ----------


def test_round_half_up_matches_javascript():
    assert round_half_up(3.75) == 3.8
    assert round_half_up(3.45) == 3.5
    assert round_half_up(3.44) == 3.4


def test_rating_summary_when_everything_is_flagged():
    reviews = [mk(1, 5, "2026-01-01"), mk(2, 5, "2026-01-01")]
    summary = rating_summary(reviews, {1, 2})
    assert summary.verified is None and summary.removed == 2
