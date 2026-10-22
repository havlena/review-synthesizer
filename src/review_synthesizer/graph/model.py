"""The one door through which the graph calls a language model.

Every model call asks for a Pydantic schema and gets back a validated instance of it, so
raw model text never reaches the graph state. Two implementations:

* :class:`AnthropicModelClient` calls Claude through ``langchain-anthropic`` using native
  structured outputs. It needs ``ANTHROPIC_API_KEY`` and the network, so it is only used by
  the live scripts.
* :class:`FakeModelClient` returns fixed outputs per schema. Tests use it; it never touches
  the network or the environment.

What this cannot do: guarantee the same output twice. The API no longer accepts a
temperature setting, so run to run stability is measured by the live evaluation, not
assumed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import MAX_TOKENS, ModelConfig

T = TypeVar("T", bound=BaseModel)


class ModelClient(Protocol):
    """Structured call: a system prompt, a user prompt, and the schema of the answer."""

    config: ModelConfig
    """Model names, recorded on flags and in the brief's methods note."""

    def structured(self, schema: type[T], *, system: str, prompt: str, small: bool = False) -> T:
        """Return an instance of ``schema``. ``small`` selects the cheap model."""
        ...  # pragma: no cover - protocol


@dataclass(frozen=True)
class ModelCall:
    """One recorded call, for tests to inspect what a node asked for."""

    schema: str
    system: str
    prompt: str
    small: bool


Response = BaseModel | Callable[[str], BaseModel] | Exception


@dataclass
class FakeModelClient:
    """Returns fixed structured outputs, keyed by schema class.

    A response may be an instance, a function of the prompt (to answer per review), or an
    exception to raise (to test fail soft paths). A call for a schema with no response
    configured raises, so an unexpected model call fails the test instead of passing quietly.
    """

    responses: dict[type[BaseModel], Response] = field(default_factory=dict)
    calls: list[ModelCall] = field(default_factory=list)
    config: ModelConfig = field(default_factory=lambda: ModelConfig("fake-main", "fake-small"))

    def structured(self, schema: type[T], *, system: str, prompt: str, small: bool = False) -> T:
        self.calls.append(ModelCall(schema.__name__, system, prompt, small))
        if schema not in self.responses:
            raise LookupError(f"FakeModelClient has no response for {schema.__name__}")
        response = self.responses[schema]
        if isinstance(response, Exception):
            raise response
        out = response(prompt) if callable(response) else response
        return schema.model_validate(out.model_dump())


@dataclass
class Usage:
    """Tokens reported by the API for one model. Output includes thinking tokens."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicModelClient:
    """Calls Claude via ``langchain-anthropic``. Imports and connects on first use only.

    Records the token usage the API reports, per model, in :attr:`usage`. Effort and
    thinking are left at the API defaults; no sampling parameter is sent.
    """

    def __init__(self, config: ModelConfig | None = None, max_tokens: int = MAX_TOKENS) -> None:
        self.config = config or ModelConfig.from_env()
        self.max_tokens = max_tokens
        self.usage: dict[str, Usage] = {}
        self._chat: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _model(self, name: str) -> Any:
        with self._lock:
            if name not in self._chat:
                from langchain_anthropic import ChatAnthropic

                # No temperature: the Messages API rejects sampling parameters.
                self._chat[name] = ChatAnthropic(model=name, max_tokens=self.max_tokens)
            return self._chat[name]

    def _record(self, name: str, raw: Any) -> None:
        meta = getattr(raw, "usage_metadata", None) or {}
        with self._lock:
            u = self.usage.setdefault(name, Usage())
            u.calls += 1
            u.input_tokens += int(meta.get("input_tokens", 0))
            u.output_tokens += int(meta.get("output_tokens", 0))

    def structured(self, schema: type[T], *, system: str, prompt: str, small: bool = False) -> T:
        name = self.config.small if small else self.config.main
        # json_schema uses native structured outputs; function_calling would force a tool
        # choice, which current models reject.
        runnable = self._model(name).with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        result = runnable.invoke([("system", system), ("human", prompt)])
        self._record(name, result.get("raw"))
        if result.get("parsing_error") is not None:
            raise result["parsing_error"]
        parsed = result["parsed"]
        if parsed is None:
            raise ValueError(f"{name} returned no structured output for {schema.__name__}")
        return schema.model_validate(
            parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
        )
