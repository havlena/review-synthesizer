"""The ablation harness, offline: one extraction per run, every cell scored on it, and the
fixed decision rule."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.evals.claims import CallFailure
from review_synthesizer.evals.claims_ablation import choose_default, evaluate_grid
from review_synthesizer.evals.similarity import load_paraphrase_set
from review_synthesizer.graph import ModelConfig
from review_synthesizer.graph.prompts import JUDGE_SYSTEM_LENIENT, JUDGE_SYSTEM_STRICT
from review_synthesizer.graph.schemas import ClaimOut, ExtractedClaims, PairVerdict

FIXTURES = Path(__file__).parent / "fixtures"


def claim(aspect):
    return ClaimOut(aspect=aspect, polarity="negative", specific=True, statement=aspect)


def extraction(rid):
    """C shares two specific claims, B and the pair G1, G2 share one."""
    if rid[0] == "C":
        return [claim("amenities"), claim("breakfast")]
    if rid[0] == "B":
        return [claim("service")]
    if rid in ("G1", "G2"):
        return [claim("parking")]
    return []


class PromptAwareFake:
    """Strict judge: only C pairs are the same. Lenient: B and C pairs, and G1-G2 (wrong)."""

    config = ModelConfig("fake-main", "fake-small")

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def structured(self, schema, *, system, prompt, small=False):
        ids = re.findall(r'<review id="([^"]+)"', prompt)
        self.calls.append((schema.__name__, system, tuple(ids)))
        if self.fail_on and self.fail_on in ids:
            raise ConnectionError("reset by peer")
        if schema is ExtractedClaims:
            return ExtractedClaims(claims=extraction(ids[0]))
        a, b = ids
        groups = {a[0], b[0]}
        if system == JUDGE_SYSTEM_STRICT:
            same = groups == {"C"}
        else:
            assert system == JUDGE_SYSTEM_LENIENT
            same = groups in ({"C"}, {"B"}) and a[0] != "G" or {a, b} == {"G1", "G2"}
        return PairVerdict(same_claims=same, reason=f"{a} and {b}.")


def test_one_extraction_per_run_and_every_cell_scored_on_it(tmp_path):
    rows, _ = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    clients = []

    def make():
        clients.append(PromptAwareFake())
        return clients[-1]

    report = evaluate_grid(rows, make, runs=2, save_dir=tmp_path, current=(2, "strict"))

    for c in clients:
        extractions = [x for x in c.calls if x[0] == "ExtractedClaims"]
        judged = [x for x in c.calls if x[0] == "PairVerdict"]
        assert len(extractions) == 24
        assert len(judged) == 2 * 13  # union at 1 shared (6 B, 6 C, G1-G2), each prompt once
    assert report.cells == [
        "2 shared, strict",
        "2 shared, lenient",
        "1 shared, strict",
        "1 shared, lenient",
    ]

    r1 = report.runs[0]["cells"]
    assert r1["2 shared, strict"]["candidates"] == {"total": 6, "true": 6, "false": 0}
    assert r1["1 shared, strict"]["candidates"] == {"total": 13, "true": 12, "false": 1}
    assert r1["1 shared, lenient"]["judged"]["true_positive"] == 12
    assert r1["1 shared, lenient"]["wrong_joins"] == [
        {"pair": "G1-G2", "clusters": "distinct / distinct", "reason": "G1 and G2."}
    ]
    assert report.summary["1 shared, lenient"]["wrong_joins_per_run"] == [1, 1]
    assert report.summary["2 shared, strict"]["lost_per_run"] == [(12, 0), (12, 0)]

    # 2 shared strict and 1 shared strict tie at 6 of 18; the stricter one is the default.
    assert report.decision["chosen"] == "2 shared, strict"
    assert report.decision["changed"] is False

    saved = json.loads((tmp_path / "extraction_run1.json").read_text(encoding="utf-8"))
    assert saved["model"] == "fake-main" and saved["claims"]["C1"][0]["aspect"] == "amenities"
    md = report.to_markdown()
    assert "| 1 shared, lenient | 1 | 12 / 1 | 12 / 1 |" in md
    assert "- 1 shared, lenient, run 1: G1-G2 (distinct / distinct). G1 and G2." in md
    assert "Judgments at 500 reviews" in md
    report.to_json()


def summary(**cells):
    return {
        name: {"wrong_joins_per_run": wrong, "mean_recall": recall}
        for name, (wrong, recall) in cells.items()
    }


CELLS = [(2, "strict"), (2, "lenient"), (1, "strict"), (1, "lenient")]
NAMES = ["2 shared, strict", "2 shared, lenient", "1 shared, strict", "1 shared, lenient"]


def test_rule_picks_highest_recall_without_a_wrong_join():
    s = summary(
        **dict(zip(NAMES, [([0, 0], 0.25), ([0, 0], 0.40), ([0, 0], 0.60), ([1, 0], 0.90)]))
    )
    d = choose_default(s, CELLS, (2, "strict"))
    assert (d["chosen"], d["changed"], d["min_shared"], d["judge"]) == (
        "1 shared, strict",
        True,
        1,
        "strict",
    )


def test_rule_breaks_ties_toward_the_stricter_setting():
    s = summary(
        **dict(zip(NAMES, [([0, 0], 0.25), ([0, 0], 0.50), ([0, 0], 0.50), ([0, 0], 0.50)]))
    )
    d = choose_default(s, CELLS, (2, "strict"))
    assert d["chosen"] == "2 shared, lenient"
    assert "stricter" in d["reason"]


def test_a_current_default_with_a_wrong_join_is_replaced_by_a_clean_cell():
    s = summary(
        **dict(zip(NAMES, [([0, 0], 0.25), ([0, 0], 0.25), ([0, 0], 0.25), ([1, 0], 0.90)]))
    )
    d = choose_default(s, CELLS, (1, "lenient"))
    assert (d["chosen"], d["changed"]) == ("2 shared, strict", True)


def test_rule_keeps_the_default_when_nothing_beats_it_cleanly():
    s = summary(
        **dict(zip(NAMES, [([0, 0], 0.25), ([0, 1], 0.40), ([0, 0], 0.25), ([2, 2], 0.90)]))
    )
    d = choose_default(s, CELLS, (2, "strict"))
    assert (d["chosen"], d["changed"]) == ("2 shared, strict", False)


def test_a_failed_judgment_stops_the_ablation():
    rows, _ = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    with pytest.raises(CallFailure) as err:
        evaluate_grid(rows, lambda: PromptAwareFake(fail_on="C3"), runs=1)
    assert err.value.stage == "claim extraction"
    assert "ConnectionError: reset by peer" in err.value.failures[0]
