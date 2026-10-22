"""The ablation harness, offline: one extraction per run per set, every cell scored on it,
the same property control, and the fixed decision rule."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.authenticity import Review
from review_synthesizer.evals.claims import CallFailure
from review_synthesizer.evals.claims_ablation import choose_default, evaluate_grid
from review_synthesizer.evals.similarity import LabelledReview, load_paraphrase_set
from review_synthesizer.graph import ModelConfig
from review_synthesizer.graph.prompts import JUDGE_SYSTEM_LENIENT, JUDGE_SYSTEM_STRICT
from review_synthesizer.graph.schemas import ClaimOut, ExtractedClaims, PairVerdict

FIXTURES = Path(__file__).parent / "fixtures"


def control_rows():
    data = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))
    return [
        LabelledReview(Review.from_dict(r), None)
        for r in data["reviews"]
        if r["intent"] == "genuine"
    ]


def claim(aspect):
    return ClaimOut(aspect=aspect, polarity="negative", specific=True, statement=aspect)


def extraction(rid):
    """C shares two specific claims; B, the pair G1 G2, and genuine Harbor 1 and 2 share one."""
    if rid[0] == "C":
        return [claim("amenities"), claim("breakfast")]
    if rid[0] == "B":
        return [claim("service")]
    if rid in ("G1", "G2"):
        return [claim("parking")]
    if rid in ("1", "2"):
        return [claim("location")]
    return []


class PromptAwareFake:
    """Strict judge: B and C pairs are the same. Lenient: also G1-G2 and Harbor 1-2 (wrong)."""

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
        same = a[0] == b[0] and a[0] in "BC"
        if system == JUDGE_SYSTEM_LENIENT:
            same = same or {a, b} in ({"G1", "G2"}, {"1", "2"})
        else:
            assert system == JUDGE_SYSTEM_STRICT
        return PairVerdict(same_claims=same, reason=f"{a} and {b}.")


def test_every_cell_scored_on_one_extraction_per_set_and_the_control_decides(tmp_path):
    rows, _ = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    clients = []

    def make():
        clients.append(PromptAwareFake())
        return clients[-1]

    report = evaluate_grid(rows, control_rows(), make, runs=2, save_dir=tmp_path)

    for c in clients:
        assert sum(x[0] == "ExtractedClaims" for x in c.calls) == 24 + 20
        # Union at 1 shared: paraphrase 13 (6 B, 6 C, G1-G2), control 1 (1-2); each prompt once.
        assert sum(x[0] == "PairVerdict" for x in c.calls) == 2 * (13 + 1)
    assert report.control_pairs == 190

    r1 = report.runs[0]
    assert r1["control_cells"]["1 shared, lenient"]["wrong_joins"] == [
        {"pair": "1-2", "reason": "1 and 2."}
    ]
    assert r1["control_cells"]["1 shared, strict"] == {"candidates": 1, "wrong_joins": []}
    s = report.summary
    assert s["1 shared, lenient"]["control_wrong_joins_per_run"] == [1, 1]
    assert s["1 shared, strict"]["mean_recall"] == pytest.approx(12 / 18)

    d = report.decision
    assert d["qualifying"] == ["2 shared, strict", "2 shared, lenient", "1 shared, strict"]
    assert (d["chosen"], d["min_shared"], d["judge"]) == ("1 shared, strict", 1, "strict")

    assert (tmp_path / "control_extraction_run2.json").exists()
    md = report.to_markdown()
    assert "- 1 shared, lenient, run 1, Harbor Light control: 1-2. 1 and 2." in md
    assert "| 1 shared, lenient | 1 / 1, 1 / 1 | no |" in md
    assert "Zero wrong joins on 190 honest pairs of one property" in md
    assert "Harbor Light share" in md
    report.to_json()


def summary(*cells):
    names = ["2 shared, strict", "2 shared, lenient", "1 shared, strict", "1 shared, lenient"]
    return {
        n: {"wrong_joins_per_run": w, "control_wrong_joins_per_run": c, "mean_recall": r}
        for n, (w, c, r) in zip(names, cells, strict=True)
    }


CELLS = [(2, "strict"), (2, "lenient"), (1, "strict"), (1, "lenient")]


def test_rule_ties_go_to_the_stricter_setting():
    s = summary(
        ([0, 0], [0, 0], 0.3), ([0, 0], [0, 0], 0.3), ([0, 0], [1, 0], 0.6), ([0, 0], [0, 2], 0.7)
    )
    d = choose_default(s, CELLS)
    assert d["chosen"] == "2 shared, strict" and "stricter" in d["reason"]


def test_a_control_wrong_join_in_one_run_disqualifies():
    s = summary(
        ([0, 0], [0, 0], 0.2), ([0, 0], [0, 1], 0.5), ([0, 0], [0, 0], 0.4), ([0, 0], [0, 0], 0.4)
    )
    d = choose_default(s, CELLS)
    assert d["chosen"] == "1 shared, strict"
    assert d["qualifying"] == ["2 shared, strict", "1 shared, strict", "1 shared, lenient"]


def test_no_qualifying_cell_means_no_change():
    s = summary(
        ([1, 0], [0, 0], 0.2), ([0, 0], [0, 1], 0.5), ([0, 1], [0, 0], 0.4), ([2, 2], [3, 3], 0.9)
    )
    d = choose_default(s, CELLS)
    assert d["chosen"] is None and d["changed"] is False
    assert "design change, not a setting" in d["reason"]


def test_a_failed_call_stops_the_ablation():
    rows, _ = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    with pytest.raises(CallFailure) as err:
        evaluate_grid(rows, control_rows(), lambda: PromptAwareFake(fail_on="7"), runs=1)
    assert err.value.stage == "claim extraction"
    assert "ConnectionError: reset by peer" in err.value.failures[0]
