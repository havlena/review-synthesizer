"""themes_and_scores and actions: verified reviews only, checked output, fixed ranking.

All model output comes from FakeModelClient; no network, no key.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.graph import Action, FakeModelClient, Property
from review_synthesizer.graph.batch import validate_batch
from review_synthesizer.graph.nodes import NOTHING_VERIFIED, actions, themes_and_scores
from review_synthesizer.graph.prompts import ACTIONS_SYSTEM, SYNTHESIS_SYSTEM
from review_synthesizer.graph.schemas import (
    ActionList,
    ActionOut,
    AspectScoreOut,
    Synthesis,
    ThemeOut,
)
from review_synthesizer.graph.synthesis import MAX_ACTIONS, clean_actions

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")
PLANTED = [21, 22, 23, 24, 25]


def harbor_rows():
    return json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]


def harbor_reviews():
    return validate_batch(harbor_rows())


def state_with(verified_ids):
    return {"property": HARBOR, "reviews": harbor_reviews(), "verified_ids": verified_ids}


SYNTHESIS = Synthesis(
    themes=[
        ThemeOut(
            name="Front desk",
            sentiment="mixed",
            review_ids=["1", "4"],
            evidence=["warm and helpful", "check in took a while", "a phrase nobody wrote"],
        ),
        ThemeOut(
            name="Breakfast",
            sentiment="negative",
            review_ids=["2", "3", "5", "99", "21"],
            evidence=["Breakfast was just packaged muffins", "could use more hot options"],
        ),
        ThemeOut(name="Ghost theme", sentiment="positive", review_ids=["42"], evidence=[]),
    ],
    aspect_scores=[
        AspectScoreOut(aspect="noise", score=2, review_ids=["4"], evidence="Thin walls."),
        AspectScoreOut(aspect="service", score=3, review_ids=["1", "4", "5"], evidence="Mixed."),
        AspectScoreOut(aspect="wifi", score=7, review_ids=["6"], evidence="Off the scale."),
        AspectScoreOut(aspect="service", score=5, review_ids=["1"], evidence="Repeat."),
    ],
)


def test_synthesis_sees_only_verified_reviews_and_uses_the_main_model():
    client = FakeModelClient({Synthesis: SYNTHESIS})
    themes_and_scores(state_with(list(range(1, 21))), client=client)
    (call,) = client.calls
    assert call.system == SYNTHESIS_SYSTEM and not call.small
    assert "Verified reviews (20):" in call.prompt
    for r in harbor_reviews():
        assert (f'<review id="{r.id}"' in call.prompt) == (r.id not in PLANTED)
        assert (r.text in call.prompt) == (r.id not in PLANTED)


def test_themes_keep_verified_support_and_exact_quotes_ranked_by_support():
    out = themes_and_scores(
        state_with(list(range(1, 21))), client=FakeModelClient({Synthesis: SYNTHESIS})
    )
    breakfast, desk = out["themes"]
    assert (breakfast.name, breakfast.count, breakfast.review_ids) == ("Breakfast", 3, [2, 3, 5])
    assert breakfast.evidence == [
        "Breakfast was just packaged muffins",
        "could use more hot options",
    ]
    assert (desk.count, desk.evidence) == (2, ["warm and helpful", "check in took a while"])
    assert out["notes"] == [
        "3 review reference(s) in the themes were not verified reviews and were left out.",
        "1 theme(s) had no verified supporting review and were left out.",
        (
            "1 quoted phrase(s) did not appear in the cited reviews and were left out of the "
            "theme evidence."
        ),
        (
            "2 aspect score(s) were left out: off the 1 to 5 scale, repeated, or without a "
            "verified supporting review."
        ),
    ]


def test_scorecard_is_in_fixed_aspect_order_with_counts_from_verified_ids():
    out = themes_and_scores(
        state_with(list(range(1, 21))), client=FakeModelClient({Synthesis: SYNTHESIS})
    )
    assert [(s.aspect, s.score, s.review_count) for s in out["aspect_scores"]] == [
        ("service", 3, 3),
        ("noise", 2, 1),
    ]


def test_system_prompt_keeps_the_capstone_rules():
    assert "passed the authenticity checks" in SYNTHESIS_SYSTEM
    assert "Never\nset a negative review aside for being brief or blunt" in SYNTHESIS_SYSTEM
    assert "Apply the bands the same way every time" in SYNTHESIS_SYSTEM
    assert "Do not suggest responding to, reporting or\nremoving reviews" in ACTIONS_SYSTEM


def test_nothing_verified_means_no_model_call_and_a_note():
    client = FakeModelClient()
    out = themes_and_scores(state_with([]), client=client)
    assert out == {"themes": [], "aspect_scores": [], "notes": [NOTHING_VERIFIED]}
    assert actions(state_with([]), client=client) == {"actions": []}
    assert client.calls == []


def test_a_failed_synthesis_call_leaves_the_parts_empty_with_a_note():
    out = themes_and_scores(
        state_with([1, 2]), client=FakeModelClient({Synthesis: TimeoutError("slow")})
    )
    assert out["themes"] == [] and out["aspect_scores"] == []
    assert out["notes"] == ["Theme and scorecard synthesis failed (TimeoutError); both are empty."]


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
    result, notes = clean_actions(raw, harbor_reviews()[:20])
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


def test_action_prompt_carries_themes_scores_and_verified_reviews_only():
    synth = themes_and_scores(
        state_with(list(range(1, 21))), client=FakeModelClient({Synthesis: SYNTHESIS})
    )
    state = {**state_with(list(range(1, 21))), **synth}
    client = FakeModelClient(
        {
            ActionList: ActionList(
                actions=[act("Add hot breakfast items", "High", "Medium", ["2", "3"])]
            )
        }
    )
    out = actions(state, client=client)
    (call,) = client.calls
    assert call.system == ACTIONS_SYSTEM
    assert "- Breakfast (negative, 3 reviews: 2, 3, 5)" in call.prompt
    assert "- service: 3 of 5 (3 reviews). Mixed." in call.prompt
    assert '<review id="21"' not in call.prompt
    assert out["actions"] == [
        Action("Add hot breakfast items", "Because.", "High", "Medium", [2, 3])
    ]


def test_after_a_dismissal_the_rerun_synthesis_includes_the_released_reviews(run_graph):
    client = FakeModelClient(
        {Synthesis: Synthesis(themes=[], aspect_scores=[]), ActionList: ActionList(actions=[])}
    )
    run_graph(
        client,
        {"property": HARBOR, "raw_reviews": harbor_rows()},
        decide=lambda req: {f["key"]: False for f in req["flags"]},
    )
    first, second = [c.prompt for c in client.calls if c.schema == "Synthesis"]
    assert "Verified reviews (20):" in first and '<review id="21"' not in first
    assert "Verified reviews (25):" in second and '<review id="21"' in second
