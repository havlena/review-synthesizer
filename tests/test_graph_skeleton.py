"""Graph skeleton: wiring, fan out and join, checkpointing, and the fake model client.

Runs without network and without a key. Skipped when the ``graph`` extra is not installed.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from review_synthesizer.authenticity import Review
from review_synthesizer.graph import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_SMALL,
    NODE_NAMES,
    FakeModelClient,
    ModelConfig,
    Property,
    build_graph,
)
from review_synthesizer.graph.schemas import ExtractedClaims

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn", region="New England coast")


def harbor_input():
    data = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))
    return {"property": HARBOR, "raw_reviews": data["reviews"]}


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    """No test in this file may depend on a real key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_graph_has_every_node_and_the_planned_edges():
    g = build_graph(FakeModelClient()).get_graph()
    assert set(NODE_NAMES) <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert {
        ("__start__", "load_batch"),
        ("load_batch", "authenticity"),
        ("load_batch", "extract_claims"),
        ("authenticity", "claim_duplicates"),
        ("extract_claims", "claim_duplicates"),
        ("claim_duplicates", "verified_set"),
        ("verified_set", "themes_and_scores"),
        ("themes_and_scores", "actions"),
        ("actions", "human_review"),
        ("human_review", "compose_brief"),
        ("human_review", "verified_set"),
        ("compose_brief", "__end__"),
    } <= edges


def test_fan_out_runs_both_branches_in_one_step_and_joins_once(run_graph):
    # Each checkpoint records the nodes scheduled for the next superstep, so the history
    # shows which nodes ran together, in order.
    _, graph, config, _ = run_graph(FakeModelClient(), harbor_input())
    history = sorted(graph.get_state_history(config), key=lambda s: s.metadata["step"])
    steps = [set(s.next) for s in history if s.next]
    assert steps == [
        {"__start__"},
        {"load_batch"},
        {"authenticity", "extract_claims"},
        {"claim_duplicates"},
        {"verified_set"},
        {"themes_and_scores"},
        {"actions"},
        {"human_review"},
        {"compose_brief"},
    ]


def test_graph_runs_end_to_end_on_the_fake_model(run_graph):
    client = FakeModelClient({ExtractedClaims: ExtractedClaims(claims=[])})
    out, *_ = run_graph(client, harbor_input())
    assert {c.schema for c in client.calls} == {"ExtractedClaims"}
    assert out.get("notes", []) == []
    assert len(out["reviews"]) == 25
    assert all(isinstance(r, Review) for r in out["reviews"])
    assert set(out["claims"]) == {r.id for r in out["reviews"]}
    assert "brief" in out


def test_state_survives_a_checkpoint():
    graph = build_graph(FakeModelClient(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "harbor-1"}}
    graph.invoke(harbor_input(), config)
    saved = graph.get_state(config).values
    assert saved["property"] == HARBOR
    assert isinstance(saved["reviews"][0], Review)
    assert saved["reviews"][0].posted.isoformat() == "2026-03-02"


class Answer(BaseModel):
    value: str


class Unconfigured(BaseModel):
    value: int


def test_fake_client_returns_fixed_output_and_records_the_call():
    client = FakeModelClient({Answer: lambda prompt: Answer(value=prompt.upper())})
    out = client.structured(Answer, system="sys", prompt="harbor", small=True)
    assert out == Answer(value="HARBOR")
    assert client.calls[0].schema == "Answer" and client.calls[0].small


def test_fake_client_raises_on_unconfigured_schema_and_configured_errors():
    client = FakeModelClient({Answer: ValueError("extraction failed")})
    with pytest.raises(ValueError, match="extraction failed"):
        client.structured(Answer, system="", prompt="")
    with pytest.raises(LookupError, match="Unconfigured"):
        client.structured(Unconfigured, system="", prompt="")


def test_model_names_come_from_the_environment_with_defaults():
    assert ModelConfig.from_env({}) == ModelConfig(DEFAULT_MODEL, DEFAULT_MODEL_SMALL)
    env = {"ANTHROPIC_MODEL": "main-x", "ANTHROPIC_MODEL_SMALL": "small-y"}
    assert ModelConfig.from_env(env) == ModelConfig("main-x", "small-y")
