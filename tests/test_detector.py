"""Composite detector, embedding signal, and classifier signal, with injected fakes."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from review_synthesizer.authenticity import (
    Review,
    detect,
    embedding_flags,
    find_embedding_clusters,
    rating_summary,
    similar_pairs,
)
from review_synthesizer.authenticity import rules
from review_synthesizer.authenticity.clustering import union_clusters
from review_synthesizer.authenticity.embeddings import normalize_rows, similarity_matrix
from review_synthesizer.evals import similarity

FIXTURES = Path(__file__).parent / "fixtures"


def mk(id_, text, rating=4, posted="2026-01-01"):
    return Review(id=id_, text=text, rating=rating, posted=date.fromisoformat(posted))


class VectorEmbedder:
    """Returns a fixed vector per text, so similarity is under the test's control."""

    name = "fake-vectors"

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def embed(self, texts):
        return normalize_rows(np.array([self.table[t] for t in texts], dtype=float))


@dataclass
class FakeTerm:
    term: str
    weight: float


@dataclass
class FakeExplanation:
    probability: float
    toward_deceptive: list
    toward_truthful: list


class FakeClassifier:
    def __init__(self, probs: dict[str, float]):
        self.probs = probs

    def predict_proba(self, texts):
        return np.array([self.probs[t] for t in texts])

    def explain(self, text, k=3):
        return FakeExplanation(
            self.probs[text], [FakeTerm("amazing", 1.2), FakeTerm("luxury", 0.9)], [FakeTerm("floor", -0.7)]
        )


# ---------- clustering helper ----------


def test_union_clusters_groups_transitively_and_orders_by_first_member():
    clusters = union_clusters(6, [(4, 5, 0.9), (0, 1, 0.7), (1, 2, 0.8)])
    assert [c.members for c in clusters] == [(0, 1, 2), (4, 5)]
    assert clusters[0].peak == 0.8 and clusters[0].pair_count == 2
    assert abs(clusters[0].mean - 0.75) < 1e-9
    assert union_clusters(3, []) == []


# ---------- embedding signal ----------


def test_similarity_matrix_is_cosine_on_unit_vectors():
    m = similarity_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]))
    assert abs(m[0, 2] - 1.0) < 1e-9 and abs(m[0, 1]) < 1e-9


def test_embedding_clusters_join_paraphrases_that_share_no_words():
    a, b, c = "best inn around", "finest lodging anywhere", "parking was tight"
    emb = VectorEmbedder({a: [1.0, 0.0], b: [0.95, 0.31], c: [0.0, 1.0]})
    reviews = [mk(1, a), mk(2, b), mk(3, c)]
    pairs = similar_pairs(reviews, emb, 0.8)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    clusters = find_embedding_clusters(reviews, emb, 0.8)
    assert len(clusters) == 1 and clusters[0].review_ids == (1, 2)
    assert clusters[0].model == "fake-vectors"
    assert find_embedding_clusters(reviews, emb, 0.99) == []
    assert similar_pairs(reviews[:1], emb, 0.5) == []


def test_embedding_flag_reports_pairs_word_overlap_missed():
    a, b, c = "best inn around", "finest lodging anywhere", "best inn around here"
    emb = VectorEmbedder({a: [1.0, 0.0], b: [0.95, 0.31], c: [0.99, 0.14]})
    reviews = [mk(1, a), mk(2, b), mk(3, c)]
    flags = embedding_flags(reviews, emb, threshold=0.8, dup_threshold=0.55)
    assert len(flags) == 1
    f = flags[0]
    assert f.signal_type == "embedding_similarity" and sorted(f.review_ids) == [1, 2, 3]
    # (1, 3) share words and are caught by Jaccard; (1, 2) and (2, 3) are not.
    assert f.details["pairs_missed_by_word_overlap"] == 2
    assert "2 of 3 pairs in the group were not caught by word overlap" in f.rationale
    assert f.confidence == "high" and 0.8 <= f.score <= 1.0


# ---------- classifier signal ----------


def test_classifier_flags_only_above_threshold_with_cues():
    t1, t2 = "amazing luxury vacation", "room on the fourth floor"
    reviews = [mk(1, t1), mk(2, t2)]
    clf = FakeClassifier({t1: 0.93, t2: 0.12})
    result = detect(reviews, classifier=clf, classifier_threshold=0.8)
    lang = [f for f in result.flags if f.signal_type == "deceptive_language"]
    assert len(lang) == 1 and lang[0].review_ids == [1]
    assert lang[0].score == 0.93 and lang[0].confidence == "high"
    assert "p = 0.93" in lang[0].rationale and "'amazing'" in lang[0].rationale
    assert lang[0].details["toward_deceptive"] == ["amazing", "luxury"]
    assert result.signals == ["duplicate_phrasing", "rating_burst", "deceptive_language"]
    assert result.flags_for(1) == lang and result.flags_for(2) == []


def test_classifier_threshold_is_inclusive():
    t = "amazing luxury vacation"
    clf = FakeClassifier({t: 0.8})
    assert len(detect([mk(1, t)], classifier=clf, classifier_threshold=0.8).flags) == 1
    assert len(detect([mk(1, t)], classifier=clf, classifier_threshold=0.81).flags) == 0


# ---------- composite equals the rules when nothing else is given ----------


def load_harbor():
    data = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))
    reviews = [Review.from_dict(r) for r in data["reviews"]]
    planted = {r["id"] for r in data["reviews"] if r["intent"] == "planted"}
    return reviews, planted


def test_composite_without_extras_matches_the_rules_golden_result():
    reviews, planted = load_harbor()
    composite = detect(reviews)
    rules_only = rules.detect(reviews)
    assert composite.flagged_ids == rules_only.flagged_ids == planted
    assert [f.signal_type for f in composite.flags] == [f.signal_type for f in rules_only.flags]
    assert composite.peak_similarity == rules_only.peak_similarity
    assert composite.signals == ["duplicate_phrasing", "rating_burst"]
    summary = rating_summary(reviews, composite.flagged_ids)
    assert (summary.displayed, summary.verified) == (3.8, 3.5)


def test_union_of_signals_drives_the_verified_rating():
    reviews, planted = load_harbor()
    genuine_text = next(r.text for r in reviews if r.id == 7)
    clf = FakeClassifier({r.text: (0.95 if r.id == 7 else 0.1) for r in reviews})
    result = detect(reviews, classifier=clf)
    assert result.flagged_ids == planted | {7}
    assert rating_summary(reviews, result.flagged_ids).removed == 6
    assert genuine_text in clf.probs


# ---------- paraphrase evaluation harness ----------


def test_paraphrase_fixture_shape():
    rows, clusters = similarity.load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    assert len(rows) == 24 and set(clusters) == {"A", "B", "C"}
    truth = similarity.truth_pairs(rows)
    assert len(truth) == 3 * 6  # three clusters of four reviews


def test_similarity_evaluate_scores_pairs_and_finds_overlap():
    rows, clusters = similarity.load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    # A fake that knows the clusters: same cluster -> near identical vectors.
    axis = {"A": 0, "B": 1, "C": 2}
    table = {}
    distinct_seen = 0
    for k, r in enumerate(rows):
        base = np.zeros(16)
        if r.cluster:
            base[axis[r.cluster]] = 1.0
            base[3] = 0.15 * (k % 4)  # a little spread inside each cluster
        else:
            base[4 + distinct_seen] = 1.0  # every distinct review has its own direction
            base[k % 3] = 0.3  # and leans toward a cluster's topic without matching it
            distinct_seen += 1
        table[r.review.text] = list(base)
    emb = VectorEmbedder(table)
    report = similarity.evaluate(rows, clusters, emb, chosen_threshold=0.80)
    assert report.truth_pairs == 18 and report.negative_pairs == 276 - 18
    assert report.chosen.recall == 1.0 and report.chosen.false_positive == 0
    assert report.jaccard.recall < report.chosen.recall
    o = report.overlap
    assert o["both"] + o["embedding_only"] + o["jaccard_only"] + o["neither"] == 18
    assert o["embedding_only"] >= 12
    md = report.to_markdown()
    assert "Word overlap (Jaccard)" in md and "(default)" in md
    assert '"chosen_threshold": 0.8' in report.to_json()
