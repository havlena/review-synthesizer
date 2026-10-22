"""Confidence bands by margin over a signal's threshold, and the similarity negative control."""

import pytest

from review_synthesizer.authenticity import confidence_band, margin_band
from review_synthesizer.evals.similarity import negative_control


@pytest.mark.parametrize(
    ("score", "threshold", "band"),
    [
        (0.801, 0.80, "low"),  # the Harbor Light borderline pair: just over is low
        (0.8499, 0.80, "low"),  # margin 0.2495
        (0.85, 0.80, "medium"),
        (0.92, 0.80, "high"),
        (1.00, 0.85, "high"),
        (0.82, 0.80, "low"),  # the classifier's p = 0.82 flags on Harbor Light
        (0.93, 0.80, "high"),
        (0.92, 0.55, "high"),  # duplicate phrasing on Harbor Light
        (0.60, 0.55, "low"),
        (0.70, 0.55, "medium"),
    ],
)
def test_margin_band_follows_the_margin_not_the_raw_score(score, threshold, band):
    assert margin_band(score, threshold) == band


def test_margin_band_edges():
    assert margin_band(0.80 + 0.25 * 0.20, 0.80) == "medium"
    assert margin_band(0.80 + 0.60 * 0.20, 0.80) == "high"


def test_raw_bands_unchanged_for_scores_without_a_threshold():
    assert [confidence_band(s) for s in (0.4, 0.5, 2 / 3, 0.75, 1.0)] == [
        "low",
        "medium",
        "medium",
        "high",
        "high",
    ]


def test_negative_control_pass_and_fail():
    ok = negative_control(0.801, 0.60, 0.85)
    assert ok["passed"] and ok["line"].startswith("PASS: most similar genuine Harbor Light pair 0.801")
    bad = negative_control(0.801, 0.60, 0.80)
    assert not bad["passed"] and bad["line"].startswith("FAIL")
    assert not negative_control(0.70, 0.85, 0.85)["passed"]
