from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

import tiktoken

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
    
    def _truncate_messages(
        self,
        messages: List[BaseMessage],
        *,
        max_tokens: int = 2000,
        encoding_name: str = "gpt2",   # универсально для gpt-совместимых
    ) -> List[BaseMessage]:
        """Жёстко обрезает суммарный контент сообщений до max_tokens.

        Создаёт *копии* сообщений через pydantic .copy(update=...),
        чтобы не трогать исходные объекты.
        """
        enc = tiktoken.get_encoding(encoding_name)

        def get_text(msg: BaseMessage) -> str:
            c = msg.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                # LangChain иногда хранит контент как список частей
                parts = []
                for part in c:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                return "\n".join(parts)
            return str(c)

        total = 0
        out: List[BaseMessage] = []
        for m in messages:
            text = get_text(m)
            toks = enc.encode(text)
            n = len(toks)

            if total + n <= max_tokens:
                out.append(m)        # можно безопасно оставить оригинал
                total += n
                continue

            # Нужно резать текущий месседж
            remain = max_tokens - total
            if remain <= 0:
                break
            cut_text = enc.decode(toks[:remain])
            out.append(m.copy(update={"content": cut_text}))   # <-- ВАЖНО: copy(update=...)
            break

        return out

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