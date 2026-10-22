"""The LangGraph graph that turns a batch of reviews into an operator brief.

Nodes, in order: ``load_batch``; then ``authenticity`` (deterministic detector) and
``extract_claims`` (model) in parallel; ``claim_duplicates`` joining both;
``verified_set``, ``themes_and_scores``, ``actions``; ``human_review`` (the operator
confirms or dismisses each flag); ``compose_brief``. See ``docs/build-plan.md``, step 5.

Needs the ``graph`` extra (``pip install -e .[graph]``). The authenticity package does
not depend on this one. Model calls go through :class:`ModelClient`, so tests run the
whole graph with :class:`FakeModelClient` and no key.

Nothing here posts, removes or accuses anything: every flag is a signal for the operator,
and no brief is final before ``human_review`` has run.
"""

from .batch import BatchError, validate_batch
from .build import NODE_NAMES, build_graph, memory_checkpointer, state_serializer
from .config import DEFAULT_MODEL, DEFAULT_MODEL_SMALL, ModelConfig
from .model import AnthropicModelClient, FakeModelClient, ModelCall, ModelClient
from .run import approve_all, run_brief
from .state import (
    Action,
    AspectScore,
    Brief,
    BriefState,
    Claim,
    PairJudgment,
    Property,
    Theme,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_SMALL",
    "NODE_NAMES",
    "Action",
    "AnthropicModelClient",
    "AspectScore",
    "BatchError",
    "Brief",
    "BriefState",
    "Claim",
    "FakeModelClient",
    "ModelCall",
    "ModelClient",
    "ModelConfig",
    "PairJudgment",
    "Property",
    "Theme",
    "approve_all",
    "build_graph",
    "memory_checkpointer",
    "run_brief",
    "state_serializer",
    "validate_batch",
]
