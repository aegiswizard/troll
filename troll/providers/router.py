"""
troll/providers/router.py

Provider abstraction + router for Troll 🧌.

Supported: OpenAI, Anthropic, OpenRouter (covers Gemini, DeepSeek,
Groq, Fireworks, Mistral, etc.), with a Gemini-direct stub.

The router picks provider/model from tier or explicit override.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from troll.core.budgets import BudgetTracker, StepCost


# ---------------------------------------------------------------------------
# Completion result
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_id: str
    provider: str
    latency_s: float


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> CompletionResult:
        ...

    @abstractmethod
    def model_id(self) -> str:
        ...


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseProvider):
    name = "openai"

    # cost per 1k tokens (approximate, varies by model)
    _COSTS: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini":            {"in": 0.000150, "out": 0.000600},
        "gpt-4o":                 {"in": 0.002500, "out": 0.010000},
        "o4-mini":                {"in": 0.001100, "out": 0.004400},
        "o3":                     {"in": 0.010000, "out": 0.040000},
    }
    _TIERS: Dict[str, str] = {
        "budget": "gpt-4o-mini",
        "mid":    "gpt-4o",
        "max":    "o3",
    }

    def __init__(self, model_tier: str = "mid", model_id: Optional[str] = None) -> None:
        try:
            import openai as _openai  # type: ignore[import]
            self._client = _openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("pip install openai")
        self._model = model_id or self._TIERS.get(model_tier, "gpt-4o")

    def model_id(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)

        kwargs: Dict[str, Any] = dict(model=self._model, messages=all_msgs, max_tokens=max_tokens)
        # o-series models use reasoning
        if self._model.startswith("o") and thinking:
            kwargs["reasoning_effort"] = "high"
        else:
            kwargs["temperature"] = temperature

        resp = self._client.chat.completions.create(**kwargs)
        lat = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        inp, out = usage.prompt_tokens, usage.completion_tokens
        costs = self._COSTS.get(self._model, {"in": 0.002, "out": 0.008})
        cost = (inp * costs["in"] + out * costs["out"]) / 1000
        return CompletionResult(
            text=text, input_tokens=inp, output_tokens=out,
            cost_usd=cost, model_id=self._model, provider=self.name, latency_s=lat,
        )


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseProvider):
    name = "anthropic"

    _COSTS: Dict[str, Dict[str, float]] = {
        "claude-haiku-4-5-20251001":  {"in": 0.000800, "out": 0.004000},
        "claude-sonnet-4-6":           {"in": 0.003000, "out": 0.015000},
        "claude-opus-4-6":             {"in": 0.015000, "out": 0.075000},
    }
    _TIERS: Dict[str, str] = {
        "budget": "claude-haiku-4-5-20251001",
        "mid":    "claude-sonnet-4-6",
        "max":    "claude-opus-4-6",
    }

    def __init__(self, model_tier: str = "mid", model_id: Optional[str] = None) -> None:
        try:
            import anthropic as _anthropic  # type: ignore[import]
            self._client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._model = model_id or self._TIERS.get(model_tier, "claude-sonnet-4-6")

    def model_id(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
            kwargs.pop("temperature", None)
        else:
            kwargs["temperature"] = temperature

        resp = self._client.messages.create(**kwargs)
        lat = time.perf_counter() - t0

        text = "".join(
            b.text for b in resp.content
            if hasattr(b, "text")
        )
        inp = resp.usage.input_tokens
        out = resp.usage.output_tokens
        costs = self._COSTS.get(self._model, {"in": 0.003, "out": 0.015})
        cost = (inp * costs["in"] + out * costs["out"]) / 1000
        return CompletionResult(
            text=text, input_tokens=inp, output_tokens=out,
            cost_usd=cost, model_id=self._model, provider=self.name, latency_s=lat,
        )


# ---------------------------------------------------------------------------
# OpenRouter provider  (Gemini, DeepSeek, Groq, Fireworks, Mistral, etc.)
# ---------------------------------------------------------------------------

class OpenRouterProvider(BaseProvider):
    name = "openrouter"

    _TIERS: Dict[str, str] = {
        "budget": "google/gemini-flash-1.5-8b",
        "mid":    "google/gemini-2.0-flash-001",
        "max":    "deepseek/deepseek-r1",
    }

    def __init__(self, model_tier: str = "mid", model_id: Optional[str] = None) -> None:
        try:
            import openai as _openai  # type: ignore[import]
            self._client = _openai.OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        except ImportError:
            raise ImportError("pip install openai  # OpenRouter uses the OpenAI SDK")
        self._model = model_id or self._TIERS.get(model_tier, "google/gemini-2.0-flash-001")

    def model_id(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=all_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        lat = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0
        # OpenRouter reports cost in usage sometimes; approximate if not
        cost_approx = (inp * 0.001 + out * 0.003) / 1000
        return CompletionResult(
            text=text, input_tokens=inp, output_tokens=out,
            cost_usd=cost_approx, model_id=self._model, provider=self.name, latency_s=lat,
        )


# ---------------------------------------------------------------------------
# Provider router
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, type] = {
    "openai":     OpenAIProvider,
    "anthropic":  AnthropicProvider,
    "openrouter": OpenRouterProvider,
    # aliases
    "gemini":     OpenRouterProvider,
    "deepseek":   OpenRouterProvider,
    "groq":       OpenRouterProvider,
}


class ProviderRouter:
    """
    Instantiates the correct provider and model for a run.
    Records all calls to the BudgetTracker.
    """

    def __init__(
        self,
        provider_name: str,
        model_tier: str = "mid",
        model_id: Optional[str] = None,
        budget: Optional[BudgetTracker] = None,
    ) -> None:
        cls = PROVIDER_REGISTRY.get(provider_name)
        if cls is None:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Choices: {list(PROVIDER_REGISTRY)}"
            )
        self._provider: BaseProvider = cls(model_tier=model_tier, model_id=model_id)  # type: ignore
        self._budget = budget

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        thinking: bool = False,
        role: str = "general",
        step: int = 0,
    ) -> CompletionResult:
        result = self._provider.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
        )
        if self._budget:
            self._budget.record(StepCost(
                step=step,
                provider=result.provider,
                model=result.model_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                latency_s=result.latency_s,
                role=role,
            ))
        return result

    @property
    def model_id(self) -> str:
        return self._provider.model_id()

    @property
    def provider_name(self) -> str:
        return self._provider.name
