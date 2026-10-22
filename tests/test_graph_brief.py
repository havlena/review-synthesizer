"""compose_brief and run_brief on the Harbor Light Inn fixture, end to end on the fake model.

Confirming every flag must give the golden 3.8 displayed to 3.5 verified, with the four
parts of the brief, the authenticity findings and the methods note.
"""

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.graph import (
    BatchError,
    FakeModelClient,
    Property,
    approve_all,
    run_brief,
)
from review_synthesizer.graph.batch import validate_batch
from review_synthesizer.graph.schemas import (
    ActionList,
    ActionOut,
    AspectScoreOut,
    ExtractedClaims,
    Synthesis,
    ThemeOut,
)

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn", region="fictional")


def harbor_rows():
    return json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]


def fake_client(synthesis=None):
    return FakeModelClient(
        {
            ExtractedClaims: ExtractedClaims(claims=[]),
            Synthesis: synthesis
            or Synthesis(
                themes=[
                    ThemeOut(
                        name="Breakfast | coffee",
                        sentiment="negative",
                        review_ids=["2", "3"],
                        evidence=["weak coffee", "could use more hot options"],
                    )
                ],
                aspect_scores=[
                    AspectScoreOut(
                        aspect="breakfast", score=2, review_ids=["2", "3"], evidence="Thin."
                    )
                ],
            ),
            ActionList: ActionList(
                actions=[
                    ActionOut(
                        title="Add two hot breakfast items",
                        why="Guests call the spread thin.",
                        impact="Medium",
                        effort="Medium",
                        review_ids=["2", "3"],
                    )
                ]
            ),
        }
    )


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_confirming_every_flag_gives_the_golden_brief():
    requests = []

    def approve(request):
        requests.append(request)
        return approve_all(request)

    brief = run_brief(harbor_rows(), HARBOR, approve, client=fake_client(), run_id="golden")
    data = brief.data

    assert len(requests) == 1 and len(requests[0]["flags"]) == 2
    assert data["rating"] == {"total": 25, "removed": 5, "displayed": 3.8, "verified": 3.5}
    assert data["authenticity"]["held_review_ids"] == [21, 22, 23, 24, 25]
    assert data["authenticity"]["verified_review_ids"] == list(range(1, 21))
    assert [f["decision"] for f in data["authenticity"]["flags"]] == ["confirmed", "confirmed"]
    assert data["themes"][0]["review_ids"] == [2, 3]
    assert data["actions"][0]["title"] == "Add two hot breakfast items"
    assert data["methods"] == {
        "signals": ["duplicate_phrasing", "rating_burst", "claim_duplicate"],
        "thresholds": {
            "duplicate_phrasing": "Jaccard word overlap at or above 0.55",
            "rating_burst": "3 or more same rating reviews within 4 days",
            "claim_duplicate": "at least 2 shared specific claims, then a model judgment per pair",
        },
        "models": {"main": "fake-main", "small": "fake-small"},
        "run_id": "golden",
    }
    json.dumps(data)  # plain JSON throughout, burst dates included


def test_markdown_has_the_four_parts_in_order_and_the_methods_note():
    md = run_brief(harbor_rows(), HARBOR, approve_all, client=fake_client(), run_id="md").markdown
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Verified rating",
        "## 1. Themes",
        "## 2. Aspect scorecard",
        "## 3. Actions",
        "## 4. Authenticity findings",
        "## Methods",
    ]
    assert md.startswith("# Operator brief: Harbor Light Inn\n")
    assert "**3.5** verified, from 20 of 25 reviews (displayed rating 3.8)" in md
    assert "| Breakfast \\| coffee | negative | 2 (2, 3) |" in md
    assert "1. **Add two hot breakfast items** Impact Medium, effort Medium." in md
    assert "- **Confirmed**, duplicate_phrasing, reviews 21, 22, 23, 24, 25, high" in md
    assert "signal for review, not a verdict" in md
    assert "- Models: main fake-main, small fake-small." in md


def test_dismissing_every_flag_restores_the_displayed_rating_and_says_so():
    dismiss = lambda req: {f["key"]: False for f in req["flags"]}
    brief = run_brief(harbor_rows(), HARBOR, dismiss, client=fake_client())
    assert brief.data["rating"]["verified"] == 3.8 and brief.data["rating"]["removed"] == 0
    assert [f["decision"] for f in brief.data["authenticity"]["flags"]] == [
        "dismissed",
        "dismissed",
    ]
    assert "- **Dismissed**, rating_burst" in brief.markdown


def test_a_note_repeated_by_the_rerun_appears_once():
    client = fake_client()
    client.responses[Synthesis] = TimeoutError("slow")
    dismiss = lambda req: {f["key"]: False for f in req["flags"]}
    brief = run_brief(harbor_rows(), HARBOR, dismiss, client=client)
    failure = "Theme and scorecard synthesis failed (TimeoutError); both are empty."
    assert sum(c.schema == "Synthesis" for c in client.calls) == 2
    assert brief.data["notes"].count(failure) == 1
    assert "## Notes on this run" in brief.markdown
    assert "No themes produced; see notes." in brief.markdown


def test_checkpoints_rebuild_only_registered_state_types(caplog):
    # LangGraph logs, and will later refuse, rebuilding a type nobody registered.
    with caplog.at_level(logging.WARNING):
        run_brief(harbor_rows(), HARBOR, approve_all, client=fake_client())
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]


def test_accepts_review_objects_and_a_property_name():
    reviews = validate_batch(harbor_rows())
    brief = run_brief(reviews, "Harbor Light Inn", approve_all, client=fake_client())
    assert brief.data["property"]["name"] == "Harbor Light Inn"
    assert brief.data["rating"]["verified"] == 3.5
    assert len(brief.data["run_id"]) == 12


def test_invalid_input_raises_before_any_model_call():
    client = fake_client()
    with pytest.raises(BatchError):
        run_brief(
            [{"id": 1, "text": "", "rating": 9, "date": "x"}], HARBOR, approve_all, client=client
        )
    assert client.calls == []
