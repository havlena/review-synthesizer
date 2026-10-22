"""The fictional properties behind the evaluation fixtures, and the control set loader.

Evaluations must send the model the same prompt the graph sends, property line included:
the Harbor Light control once used a placeholder property name while the graph used the
real one, and the judge's verdict on the same pair differed between the two.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..authenticity import Review
from ..graph.state import Property
from .similarity import LabelledReview

HARBOR_PROPERTY = Property(
    name="Harbor Light Inn", type="inn", region="fictional, Northern Michigan"
)
"""The house example, as ``scripts/run_graph.py`` runs it (tests/fixtures/harbor_light_inn.json)."""

PARAPHRASE_PROPERTY = Property(
    name="Birch Point Hotel", type="hotel", region="fictional, Northern Michigan"
)
"""A fictional property for the paraphrase set, which names none; its reviews mention
Northern Michigan, so the region matches them."""

PLACEHOLDER_PROPERTY = Property(name="Lakeshore hotel (fictional)", type="hotel", region="")
"""The placeholder the claim evaluations used for both sets until the prompt fix; kept so
reports made with it can say so."""


def harbor_control(path: str | Path) -> list[LabelledReview]:
    """The genuine Harbor Light reviews (the fixture's intent labels) as a negatives only set."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        LabelledReview(Review.from_dict(r), None)
        for r in data["reviews"]
        if r.get("intent") == "genuine"
    ]
