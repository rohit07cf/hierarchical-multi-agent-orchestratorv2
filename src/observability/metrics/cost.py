"""Token accounting + cost estimation.

LLM cost is the line item finance asks about and the one most AI systems
fly blind on. We make it a first-class signal:

- ``estimate_tokens`` — a cheap heuristic (~4 chars/token) used when the
  provider does not return usage (mock mode, streaming, or non-OpenAI
  backends). It is explicitly approximate; real usage from the API
  response always wins when present.
- ``PRICING`` — per-model $/1K tokens. Kept as plain data so updating a
  price is a one-line change and adding a model is trivial. Unknown
  models price at $0 (and we tag the model so the gap is visible) rather
  than guessing.

This turns "why did the bill spike?" into a PromQL query sliced by model
and operation instead of a forensic log dig.
"""

from __future__ import annotations

from dataclasses import dataclass

_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1K tokens."""

    input_per_1k: float
    output_per_1k: float


# $/1K tokens. Update as provider pricing changes.
PRICING: dict[str, ModelPrice] = {
    "gpt-4.1-nano": ModelPrice(0.0001, 0.0004),
    "gpt-4.1-mini": ModelPrice(0.0004, 0.0016),
    "gpt-4.1": ModelPrice(0.002, 0.008),
    "gpt-4o": ModelPrice(0.0025, 0.01),
    "gpt-4o-mini": ModelPrice(0.00015, 0.0006),
}

_DEFAULT_PRICE = ModelPrice(0.0, 0.0)


def estimate_tokens(text: str | None) -> int:
    """Heuristic token count for when the provider omits usage."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price a call from token counts. Unknown models cost $0 (and stay visible)."""
    price = PRICING.get(model, _DEFAULT_PRICE)
    return (
        input_tokens / 1000.0 * price.input_per_1k
        + output_tokens / 1000.0 * price.output_per_1k
    )
