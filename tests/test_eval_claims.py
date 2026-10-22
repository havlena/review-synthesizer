"""The claim duplicate evaluation harness, run offline on the fake model.

Checks the scoring, not the model: stages are counted separately, run to run agreement is
reported as pair sets, and a failed call stops the run instead of being scored.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from review_synthesizer.evals.claims import CallFailure, evaluate
from review_synthesizer.evals.similarity import load_paraphrase_set
from review_synthesizer.graph import FakeModelClient
from review_synthesizer.graph.schemas import ClaimOut, ExtractedClaims, PairVerdict

FIXTURES = Path(__file__).parent / "fixtures"


def two_specific(prompt):
    """Clusters B and C get two shared specific claims; A and the distinct reviews get none,
    except G1 and G2, which share two with each other (a false candidate)."""
    rid = re.search(r'<review id="([^"]+)"', prompt).group(1)
    aspects = {
        "B": ("service", "value"),
        "C": ("amenities", "breakfast"),
        "G": ("parking", "noise"),
    }
    if rid[0] in "BC" or rid in ("G1", "G2"):
        claims = [
            ClaimOut(aspect=a, polarity="negative", specific=True, statement=f"{rid[0]} {a}")
            for a in aspects[rid[0]]
        ]
    else:
        claims = [ClaimOut(aspect="other", polarity="positive", specific=False, statement="great")]
    return ExtractedClaims(claims=claims)


def judge(same_pairs):
    def answer(prompt):
        a, b = re.findall(r'<review id="([^"]+)"', prompt)
        return PairVerdict(same_claims=f"{a}-{b}" in same_pairs, reason=f"{a} vs {b}.")

    return answer


B_AND_C = {f"{c}{i}-{c}{j}" for c in "BC" for i in range(1, 5) for j in range(i + 1, 5)}


def make_clients(*same_sets):
    it = iter(same_sets)
    return lambda: FakeModelClient({ExtractedClaims: two_specific, PairVerdict: judge(next(it))})


def test_stages_are_scored_separately_and_agreement_is_pair_sets():
    rows, clusters = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    run1 = B_AND_C - {"C3-C4"}
    run2 = B_AND_C | {"G1-G2"}
    report = evaluate(rows, clusters, make_clients(run1, run2), runs=2)

    r1, r2 = report.runs
    assert report.truth_pairs == 18 and report.negative_pairs == 258
    assert r1["candidates"] == {"total": 13, "true": 12, "false": 1, "recall": 12 / 18}
    assert (r1["judged"]["true_positive"], r1["judged"]["false_positive"]) == (11, 0)
    assert (r2["judged"]["true_positive"], r2["judged"]["false_positive"]) == (12, 1)
    assert r2["false_positives"] == [
        {"pair": "G1-G2", "clusters": "distinct / distinct", "reason": "G1 vs G2."}
    ]

    stages = {m["pair"]: m["stage"] for m in r1["missed"]}
    assert stages["C3-C4"] == "judged different"
    assert stages["A1-A2"] == "not a candidate"
    assert len(r1["missed"]) == 18 - 11

    assert report.agreement["judged_only_in"] == {"1": [], "2": ["C3-C4", "G1-G2"]}
    assert len(report.agreement["judged_in_all"]) == 11
    assert report.agreement["candidates_only_in"] == {"1": [], "2": []}
    assert [c["candidates"] for c in report.per_cluster] == [[0, 0], [6, 6], [6, 6]]
    assert report.tokens["usd"] is None  # the fake reports no usage


def test_markdown_shows_both_stages_agreement_and_methods():
    rows, clusters = load_paraphrase_set(FIXTURES / "paraphrase_set.json")
    report = evaluate(
        rows,
        clusters,
        make_clients(B_AND_C, B_AND_C),
        runs=2,
        settings={"Temperature": "not sent"},
        limits=["Small synthetic set."],
    )
    md = report.to_markdown()
    assert "| True pairs that became candidates (of 18) | 12 | 12 |" in md
    assert "| True candidates judged the same | 12 of 12 | 12 of 12 |" in md
    assert "- Judged the same in run 2 only: 0 (none)." in md
    assert "| A1-A2 | not a candidate | 0 |  |" in md
    assert "- Temperature: not sent" in md
    assert "- Small synthetic set." in md
    report.to_json()


def test_a_failed_call_stops_the_run_and_names_the_review():
    rows, clusters = load_paraphrase_set(FIXTURES / "paraphrase_set.json")

    def flaky(prompt):
        if '<review id="B2"' in prompt:
            raise TimeoutError("read timed out")
        return two_specific(prompt)

    def make():
        return FakeModelClient({ExtractedClaims: flaky, PairVerdict: judge(set())})

    with pytest.raises(CallFailure) as err:
        evaluate(rows, clusters, make, runs=1)
    assert err.value.stage == "claim extraction"
    assert err.value.failures == ["ExtractedClaims for review(s) B2: TimeoutError: read timed out"]
