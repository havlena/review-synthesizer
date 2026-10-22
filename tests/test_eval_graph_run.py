"""The end to end run report, offline: golden check, genuine reviews flagged, differences."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.evals.graph_run import summarize
from review_synthesizer.graph import FakeModelClient, Property, approve_all, run_brief
from review_synthesizer.graph.run import AUTO_APPROVED
from review_synthesizer.graph.schemas import (
    ActionList,
    ActionOut,
    AspectScoreOut,
    ClaimOut,
    ExtractedClaims,
    PairVerdict,
    Synthesis,
    ThemeOut,
)

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")


def rows():
    return json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]


def client(*, breakfast_score, theme, action, join_genuine):
    def extract(prompt):
        rid = int(re.search(r'<review id="(\d+)"', prompt).group(1))
        shared = join_genuine and rid in (1, 3)
        return ExtractedClaims(
            claims=[
                ClaimOut(aspect="location", polarity="positive", specific=True, statement="view")
            ]
            if shared
            else []
        )

    return FakeModelClient(
        {
            ExtractedClaims: extract,
            PairVerdict: PairVerdict(same_claims=True, reason="Same harbor view."),
            Synthesis: Synthesis(
                themes=[
                    ThemeOut(name=theme, sentiment="negative", review_ids=["2", "3"], evidence=[])
                ],
                aspect_scores=[
                    AspectScoreOut(
                        aspect="breakfast", score=breakfast_score, review_ids=["2"], evidence="x"
                    )
                ],
            ),
            ActionList: ActionList(
                actions=[
                    ActionOut(title=action, why="w", impact="High", effort="Low", review_ids=["2"])
                ]
            ),
        }
    )


def test_report_checks_golden_flags_genuine_reviews_and_diffs_runs():
    run1 = run_brief(
        rows(),
        HARBOR,
        approve_all,
        client=client(breakfast_score=2, theme="Breakfast", action="Hot items", join_genuine=False),
        run_id="r1",
        approved_by=AUTO_APPROVED,
    )
    run2 = run_brief(
        rows(),
        HARBOR,
        approve_all,
        client=client(breakfast_score=3, theme="Coffee", action="Better coffee", join_genuine=True),
        run_id="r2",
        approved_by=AUTO_APPROVED,
    )
    usage = {"claude-haiku-5-5": {"calls": 10, "input_tokens": 1_000_000, "output_tokens": 0}}
    report = summarize(rows(), [run1.data, run2.data], [usage, {}])

    r1, r2 = report.runs
    assert r1["matches_golden"] and not r2["matches_golden"]
    assert r2["genuine_flagged"] == [
        {"review_id": 1, "signals": ["claim_duplicate"]},
        {"review_id": 3, "signals": ["claim_duplicate"]},
    ]
    assert report.headline["genuine_reviews_flagged"] == [1, 3]
    assert report.headline["all_runs_match_golden"] is False

    d = report.differences
    assert d["flags"] == {"run 1 only": [], "run 2 only": ["claim_duplicate:1,3"]}
    assert d["themes"] == {"run 1 only": ["Breakfast"], "run 2 only": ["Coffee"]}
    assert d["scores"] == [{"aspect": "breakfast", "run 1": 2, "run 2": 3}]
    assert d["actions"] == {"run 1 only": ["Hot items"], "run 2 only": ["Better coffee"]}

    md = report.to_markdown()
    assert "**At least one run does not reproduce the golden ratings.**" in md
    assert "**Genuine reviews were flagged: 1, 3.**" in md
    assert f"- Flag decisions: {AUTO_APPROVED}." in md
    assert f"- Flag decisions: {AUTO_APPROVED}." in run1.markdown
    assert report.tokens["usd"] == pytest.approx(0.10)
    report.to_json()
