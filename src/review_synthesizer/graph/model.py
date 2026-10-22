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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import MAX_TOKENS, ModelConfig

T = TypeVar("T", bound=BaseModel)


class ModelClient(Protocol):
    """Structured call: a system prompt, a user prompt, and the schema of the answer."""

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

    def structured(self, schema: type[T], *, system: str, prompt: str, small: bool = False) -> T:
        self.calls.append(ModelCall(schema.__name__, system, prompt, small))
        if schema not in self.responses:
            raise LookupError(f"FakeModelClient has no response for {schema.__name__}")
        response = self.responses[schema]
        if isinstance(response, Exception):
            raise response
        out = response(prompt) if callable(response) else response
        return schema.model_validate(out.model_dump())


class AnthropicModelClient:
    """Calls Claude via ``langchain-anthropic``. Imports and connects on first use only."""

    def __init__(self, config: ModelConfig | None = None, max_tokens: int = MAX_TOKENS) -> None:
        self.config = config or ModelConfig.from_env()
        self.max_tokens = max_tokens
        self._chat: dict[str, Any] = {}

    def _model(self, name: str) -> Any:
        if name not in self._chat:
            from langchain_anthropic import ChatAnthropic

            # No temperature: the Messages API rejects sampling parameters.
            self._chat[name] = ChatAnthropic(model=name, max_tokens=self.max_tokens)
        return self._chat[name]

    def structured(self, schema: type[T], *, system: str, prompt: str, small: bool = False) -> T:
        name = self.config.small if small else self.config.main
        # json_schema uses native structured outputs; function_calling would force a tool
        # choice, which current models reject.
        runnable = self._model(name).with_structured_output(schema, method="json_schema")
        result = runnable.invoke([("system", system), ("human", prompt)])
        return schema.model_validate(result)
