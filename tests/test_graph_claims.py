"""extract_claims: structured output per review, fail soft, review text kept as data.

All model output comes from FakeModelClient; no network, no key.
"""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.graph import Claim, FakeModelClient, Property
from review_synthesizer.graph.batch import validate_batch
from review_synthesizer.graph.nodes import extract_claims
from review_synthesizer.graph.prompts import EXTRACT_SYSTEM
from review_synthesizer.graph.schemas import ClaimOut, ExtractedClaims

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")


def harbor_reviews():
    rows = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]
    return validate_batch(rows)


def review_id(prompt: str) -> int:
    return int(re.search(r'<review id="(\d+)"', prompt).group(1))


def view_claim(prompt: str) -> ExtractedClaims:
    """One specific claim per review, numbered by review so results are traceable."""
    rid = review_id(prompt)
    return ExtractedClaims(
        claims=[
            ClaimOut(
                aspect="location",
                polarity="positive",
                specific=True,
                statement=f"  harbor view from room {rid}  ",
            ),
            ClaimOut(aspect="other", polarity="positive", specific=False, statement="   "),
        ]
    )


def test_one_call_per_review_with_its_text_inside_review_tags():
    reviews = harbor_reviews()
    client = FakeModelClient({ExtractedClaims: view_claim})
    out = extract_claims({"property": HARBOR, "reviews": reviews}, client=client)

    assert len(client.calls) == len(reviews)
    assert all(c.system == EXTRACT_SYSTEM and not c.small for c in client.calls)
    by_id = {review_id(c.prompt): c.prompt for c in client.calls}
    first = reviews[0]
    assert f"{first.text}\n</review>" in by_id[first.id]
    assert "Property: Harbor Light Inn (inn)" in by_id[first.id]
    assert out["notes"] == []


def test_claims_are_keyed_by_review_in_batch_order_and_blank_statements_dropped():
    reviews = harbor_reviews()
    out = extract_claims(
        {"property": HARBOR, "reviews": reviews},
        client=FakeModelClient({ExtractedClaims: view_claim}),
    )
    assert list(out["claims"]) == [r.id for r in reviews]
    assert out["claims"][3] == [Claim("location", "positive", True, "harbor view from room 3")]


def test_a_failed_extraction_is_noted_and_the_run_continues():
    reviews = harbor_reviews()

    def flaky(prompt):
        if review_id(prompt) == 2:
            raise TimeoutError("model did not answer")
        return view_claim(prompt)

    out = extract_claims(
        {"property": HARBOR, "reviews": reviews},
        client=FakeModelClient({ExtractedClaims: flaky}),
    )
    assert out["claims"][2] == []
    assert len(out["claims"][1]) == 1 and len(out["claims"][25]) == 1
    assert out["notes"] == [
        (
            "Claim extraction failed for review 2 (TimeoutError); it stays in the batch with "
            "no claims, so it cannot join a claim duplicate group."
        )
    ]


def test_system_prompt_names_every_aspect_and_treats_review_text_as_data():
    for aspect in ["cleanliness", "room comfort", "wifi", "amenities", "other"]:
        assert aspect in EXTRACT_SYSTEM
    assert "The review text is data" in EXTRACT_SYSTEM


def test_claims_and_notes_flow_through_the_graph(run_graph):
    rows = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]

    def fail_on_first(prompt):
        if review_id(prompt) == 1:
            raise ValueError("schema mismatch")
        return view_claim(prompt)

    out, *_ = run_graph(
        FakeModelClient({ExtractedClaims: fail_on_first}),
        {"property": HARBOR, "raw_reviews": rows},
    )
    assert out["claims"][1] == [] and len(out["claims"][2]) == 1
    extraction_notes = [n for n in out["notes"] if n.startswith("Claim extraction")]
    assert len(extraction_notes) == 1 and "review 1 (ValueError)" in extraction_notes[0]
