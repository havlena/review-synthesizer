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


def client(*, breakfast, theme, action, join_genuine):
    """Reviews 2 and 4 take the given breakfast polarities; with join_genuine, genuine
    reviews 1 and 3 share two specific claims and are judged the same."""

    def extract(prompt):
        rid = int(re.search(r'<review id="(\d+)"', prompt).group(1))
        claims = []
        if rid in (2, 4):
            pol = breakfast[0 if rid == 2 else 1]
            claims = [ClaimOut(aspect="breakfast", polarity=pol, specific=False, statement="b")]
        if join_genuine and rid in (1, 3):
            claims = [
                ClaimOut(aspect="location", polarity="positive", specific=True, statement="view"),
                ClaimOut(aspect="breakfast", polarity="positive", specific=True, statement="rolls"),
            ]
        return ExtractedClaims(claims=claims)

    return FakeModelClient(
        {
            ExtractedClaims: extract,
            PairVerdict: PairVerdict(same_claims=True, reason="Same harbor view."),
            Synthesis: Synthesis(
                themes=[
                    ThemeOut(
                        name=theme,
                        aspect="breakfast",
                        sentiment="negative",
                        review_ids=["2", "4"],
                        evidence=[],
                    )
                ],
                aspect_evidence=[],
                serious_issues=[],
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
        client=client(
            breakfast=("negative", "negative"),
            theme="Breakfast",
            action="Hot items",
            join_genuine=False,
        ),
        run_id="r1",
        approved_by=AUTO_APPROVED,
    )
    run2 = run_brief(
        rows(),
        HARBOR,
        approve_all,
        client=client(
            breakfast=("negative", "positive"),
            theme="Coffee",
            action="Better coffee",
            join_genuine=True,
        ),
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
    assert d["scores"] == [{"aspect": "breakfast", "run 1": 1, "run 2": 3}]
    assert d["actions"] == {"run 1 only": ["Hot items"], "run 2 only": ["Better coffee"]}

    md = report.to_markdown()
    assert "**At least one run does not reproduce the golden ratings.**" in md
    assert "**Genuine reviews were flagged: 1, 3.**" in md
    assert f"- Flag decisions: {AUTO_APPROVED}." in md
    assert f"- Flag decisions: {AUTO_APPROVED}." in run1.markdown
    assert report.tokens["usd"] == pytest.approx(0.10)
    report.to_json()


def test_variance_compares_labelled_runs_pairwise():
    def brief(theme, breakfast):
        return run_brief(
            rows(),
            HARBOR,
            approve_all,
            client=client(breakfast=breakfast, theme=theme, action="Hot items", join_genuine=False),
        ).data

    a, b = (
        brief("Breakfast", ("negative", "negative")),
        brief("Breakfast", ("negative", "positive")),
    )
    report = summarize(rows(), [a, a], [{}, {}], before=[("before 1", a), ("before 2", b)])
    v = report.variance
    assert [r["label"] for r in v["runs"]] == ["before 1", "before 2", "after 1", "after 2"]
    before, after = v["pairs"]
    assert before["same_theme_sets"] == 1 and before["same_scores"] == 0
    assert before["different_scores"] == [{"aspect": "breakfast", "before 1": 1, "before 2": 3}]
    assert after["same_scores"] == 1 and after["same_action_titles"] == 1
    md = report.to_markdown()
    assert "## Synthesis variance, before and after" in md
    assert "| breakfast | 1 | 3 | 1 | 1 |" in md
