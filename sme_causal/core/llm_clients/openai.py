from __future__ import annotations
from typing import Any, Optional, Tuple


from .base import LLMClient

class OpenAIClient(LLMClient):
    """OpenAI Client(base_url=None -> api.openai.com)."""

    def invoke_with_fallback(
        self,
        messages: Any,
        *,
        temperature: float,
        top_p: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
    ) -> Tuple[str, Any, bool]:
        # JSON mode
        try:
            llm_json = self._build_llm(temperature=temperature, top_p=top_p, json_mode=True, max_tokens=max_tokens, seed=seed)
            msg = llm_json.invoke(messages)
            return msg.content, msg, True
        except Exception:
            llm_txt = self._build_llm(temperature=temperature, top_p=top_p, json_mode=False, max_tokens=max_tokens, seed=seed)
            msg = llm_txt.invoke(messages)
            return msg.content, msg, False
