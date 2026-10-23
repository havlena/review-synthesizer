"""themes_and_scores and actions: verified reviews only, checked output, fixed ranking, a
scorecard computed from claims, serious issues listed apart.

All model output comes from FakeModelClient; no network, no key.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.graph import Action, Claim, FakeModelClient, Property
from review_synthesizer.graph.batch import validate_batch
from review_synthesizer.graph.nodes import NOTHING_VERIFIED, actions, themes_and_scores
from review_synthesizer.graph.prompts import ACTIONS_SYSTEM, SYNTHESIS_SYSTEM
from review_synthesizer.graph.schemas import (
    ActionList,
    ActionOut,
    AspectEvidenceOut,
    SeriousIssueOut,
    Synthesis,
    ThemeOut,
)
from review_synthesizer.graph.state import SeriousIssue
from review_synthesizer.graph.synthesis import (
    MAX_ACTIONS,
    MAX_THEMES,
    clean_actions,
    clean_serious_issues,
    clean_themes,
)

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")
PLANTED = [21, 22, 23, 24, 25]
VERIFIED = list(range(1, 21))


def harbor_rows():
    return json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]


def harbor_reviews():
    return validate_batch(harbor_rows())


def verified_reviews():
    return harbor_reviews()[:20]


# Claims as extraction would give them for a few verified reviews (and one planted).
CLAIMS = {
    1: [Claim("service", "positive", False, "staff warm and helpful")],
    2: [Claim("cleanliness", "negative", True, "room smelled musty")],
    4: [
        Claim("service", "mixed", False, "friendly but slow check in"),
        Claim("noise", "negative", True, "thin walls"),
    ],
    5: [
        Claim("cleanliness", "negative", True, "mold in the bathroom"),
        Claim("service", "negative", True, "nobody answered the phone"),
    ],
    6: [Claim("cleanliness", "positive", False, "housekeeping thorough")],
    21: [Claim("service", "positive", False, "amazing staff")],
}


def state_with(verified_ids):
    return {
        "property": HARBOR,
        "reviews": harbor_reviews(),
        "verified_ids": verified_ids,
        "claims": CLAIMS,
    }


def theme(name, ids, aspect="service", evidence=()):
    return ThemeOut(
        name=name, aspect=aspect, sentiment="mixed", review_ids=list(ids), evidence=list(evidence)
    )


SYNTHESIS = Synthesis(
    themes=[
        theme("Front desk", ["1", "4"], evidence=["warm and helpful", "a phrase nobody wrote"]),
        theme(
            "Breakfast",
            ["2", "3", "5", "99", "21"],
            aspect="breakfast",
            evidence=["Breakfast was just packaged muffins"],
        ),
        theme("One review only", ["7"]),
        theme("Planted support", ["2", "21", "22"]),
    ],
    aspect_evidence=[
        AspectEvidenceOut(
            aspect="service", evidence="Warm staff, slow phone.", review_ids=["1", "5"]
        ),
        AspectEvidenceOut(aspect="noise", evidence="Ungrounded.", review_ids=["21"]),
        AspectEvidenceOut(aspect="other", evidence="Not scored.", review_ids=["1"]),
    ],
    serious_issues=[
        SeriousIssueOut(kind="health", review_id="5", quote="The bathroom had mold in the corner"),
        SeriousIssueOut(kind="health", review_id="5", quote="The bathroom had mold in the corner"),
        SeriousIssueOut(kind="safety", review_id="3", quote="the stairs were dangerous"),
        SeriousIssueOut(kind="health", review_id="21", quote="Absolutely the best hotel"),
    ],
)


def run_synthesis(client=None):
    return themes_and_scores(
        state_with(VERIFIED), client=client or FakeModelClient({Synthesis: SYNTHESIS})
    )


def test_synthesis_sees_only_verified_reviews_and_uses_the_main_model():
    client = FakeModelClient({Synthesis: SYNTHESIS})
    run_synthesis(client)
    (call,) = client.calls
    assert call.system == SYNTHESIS_SYSTEM and not call.small
    assert "Verified reviews (20):" in call.prompt
    for r in harbor_reviews():
        assert (f'<review id="{r.id}"' in call.prompt) == (r.id not in PLANTED)


def test_themes_need_two_verified_reviews_and_count_only_those():
    out = run_synthesis()
    assert [(t.name, t.aspect, t.count, t.review_ids) for t in out["themes"]] == [
        ("Breakfast", "breakfast", 3, [2, 3, 5]),
        ("Front desk", "service", 2, [1, 4]),
    ]
    assert out["themes"][1].evidence == ["warm and helpful"]
    assert out["notes"][:3] == [
        "4 review reference(s) in the themes were not verified reviews and were left out.",
        "2 theme(s) cited fewer than 2 verified reviews and were left out.",
        (
            "1 quoted phrase(s) did not appear in the cited reviews and were left out of the "
            "theme evidence."
        ),
    ]


def test_at_most_eight_themes_the_largest_kept():
    raw = [theme(f"T{n}", [str(i) for i in range(1, n + 1)]) for n in range(2, 12)]
    themes, notes = clean_themes(raw, verified_reviews())
    assert len(themes) == MAX_THEMES == 8
    assert [t.count for t in themes] == [11, 10, 9, 8, 7, 6, 5, 4]
    assert "2 smaller theme(s) beyond the first 8 were left out." in notes


def test_serious_issues_need_a_verified_review_and_its_exact_words():
    issues, notes = clean_serious_issues(SYNTHESIS.serious_issues, verified_reviews())
    assert issues == [SeriousIssue("health", 5, "The bathroom had mold in the corner")]
    assert notes == [
        "1 review reference(s) in the serious issues were not verified reviews and were left out.",
        (
            "2 serious issue report(s) were left out: the review was not verified or the quote "
            "did not appear in it."
        ),
    ]


def test_scorecard_is_computed_from_claims_with_checked_evidence_and_issues_apart():
    out = run_synthesis()
    by = {s.aspect: s for s in out["aspect_scores"]}
    # Service: 1 positive, 4 mixed, 5 negative -> (1 + 0.5) / 3 = 0.5 -> 3.
    assert (by["service"].score, by["service"].positive, by["service"].mixed) == (3, 1, 1)
    assert by["service"].evidence == "Warm staff, slow phone."
    # Cleanliness: 2 and 5 negative, 6 positive -> 2/3 -> 2. One mold report does not make it 1.
    assert by["cleanliness"].score == 2
    assert (
        by["noise"].score is None and by["noise"].evidence == ""
    )  # one review; evidence ungrounded
    assert "other" not in by
    assert out["serious_issues"] == [
        SeriousIssue("health", 5, "The bathroom had mold in the corner")
    ]
    assert any("aspect evidence line(s) were left out" in n for n in out["notes"])


def test_system_prompt_keeps_the_capstone_rules_and_does_not_score():
    assert "passed the authenticity checks" in SYNTHESIS_SYSTEM
    assert "Never\nset a negative review aside for being brief or blunt" in SYNTHESIS_SYSTEM
    assert "Do not score" in SYNTHESIS_SYSTEM
    assert "SERIOUS ISSUES" in SYNTHESIS_SYSTEM
    assert "Do not suggest responding to, reporting or\nremoving reviews" in ACTIONS_SYSTEM


def test_nothing_verified_means_no_model_call_and_a_note():
    client = FakeModelClient()
    out = themes_and_scores(state_with([]), client=client)
    assert out == {
        "themes": [],
        "aspect_scores": [],
        "serious_issues": [],
        "notes": [NOTHING_VERIFIED],
    }
    assert actions(state_with([]), client=client) == {"actions": []}
    assert client.calls == []


def test_a_failed_synthesis_call_still_leaves_the_computed_scorecard():
    out = run_synthesis(FakeModelClient({Synthesis: TimeoutError("slow")}))
    assert out["themes"] == [] and out["serious_issues"] == []
    assert {s.aspect: s.score for s in out["aspect_scores"]}["cleanliness"] == 2
    assert out["notes"] == [
        ("Theme synthesis failed (TimeoutError); themes and serious issues are empty, and the "
        "scorecard has no evidence lines.")
    ]


def act(title, impact, effort, ids=("1",)):
    return ActionOut(
        title=title, why="Because.", impact=impact, effort=effort, review_ids=list(ids)
    )


def test_actions_ranked_by_impact_then_lower_effort_and_capped():
    raw = [
        act("Low low", "Low", "Low"),
        act("High high", "High", "High"),
        act("High low", "High", "Low"),
        act("Medium low", "Medium", "Low"),
        act("High medium", "High", "Medium"),
        act("Medium high", "Medium", "High"),
        act("Unsupported", "High", "Low", ids=("21",)),
    ]
    result, notes = clean_actions(raw, verified_reviews())
    assert [a.title for a in result] == [
        "High low",
        "High medium",
        "High high",
        "Medium low",
        "Medium high",
    ]
    assert len(result) == MAX_ACTIONS
    assert notes == [
        "1 review reference(s) in the actions were not verified reviews and were left out.",
        "1 action(s) cited no verified review and were left out.",
        "1 lower ranked action(s) were left out to keep the list to 5.",
    ]


def test_action_prompt_carries_themes_counts_issues_and_verified_reviews_only():
    state = {**state_with(VERIFIED), **run_synthesis()}
    client = FakeModelClient(
        {ActionList: ActionList(actions=[act("Fix the mold", "High", "Medium", ["5"])])}
    )
    out = actions(state, client=client)
    (call,) = client.calls
    assert call.system == ACTIONS_SYSTEM
    assert "- Breakfast (breakfast, mixed, 3 reviews: 2, 3, 5)" in call.prompt
    assert "- cleanliness: 2 of 5 (1 positive, 0 mixed, 2 negative)." in call.prompt
    assert "- noise: unscored (0 positive, 0 mixed, 1 negative)." in call.prompt
    assert '- health, review 5: "The bathroom had mold in the corner"' in call.prompt
    assert '<review id="21"' not in call.prompt
    assert out["actions"] == [Action("Fix the mold", "Because.", "High", "Medium", [5])]


def test_after_a_dismissal_the_rerun_synthesis_includes_the_released_reviews(run_graph):
    client = FakeModelClient(
        {
            Synthesis: Synthesis(themes=[], aspect_evidence=[], serious_issues=[]),
            ActionList: ActionList(actions=[]),
        }
    )
    run_graph(
        client,
        {"property": HARBOR, "raw_reviews": harbor_rows()},
        decide=lambda req: {f["key"]: False for f in req["flags"]},
    )
    first, second = [c.prompt for c in client.calls if c.schema == "Synthesis"]
    assert "Verified reviews (20):" in first and '<review id="21"' not in first
    assert "Verified reviews (25):" in second and '<review id="21"' in second
