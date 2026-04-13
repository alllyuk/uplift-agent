from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

class LLMClient(ABC):
    """Unified LLM client interface."""

    def __init__(self, *, model: str, api_key: Optional[str], base_url: Optional[str]):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def _build_llm(self, *, temperature: float, top_p: Optional[float], json_mode: bool=True, max_tokens: Optional[int], seed: Optional[int]) -> ChatOpenAI:
        model_kwargs = {}
        """
        Create a ChatOpenAI client with optional JSON response_format.

        Args:
            model: OpenAI model name.
            temperature: Sampling temperature.
            api_key: API key or None to rely on env.
            seed: Optional seed for deterministic behavior when supported.
            top_p: Optional nucleus sampling parameter.
            json_mode: If True, set response_format to JSON object.

        Returns:
            ChatOpenAI instance.
        """
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens

        common = dict(
            model=self.model,
            temperature=temperature,
            api_key=self.api_key,
            seed=seed,
        )
        if self.base_url:
            common["base_url"] = self.base_url
        if top_p is not None:
            common["top_p"] = top_p
        if model_kwargs:
            common["model_kwargs"] = model_kwargs

        return ChatOpenAI(**common)  # type: ignore[arg-type]
    
    @abstractmethod
    def invoke_with_fallback(
        self,
        messages: Any,
        *,
        temperature: float,
        top_p: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
    ) -> Tuple[str, Any, bool]:
        """Return (text, raw, used_json)."""
        ...