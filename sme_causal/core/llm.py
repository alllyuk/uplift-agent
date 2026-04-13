from __future__ import annotations

"""
Lightweight helpers to construct and invoke ChatOpenAI consistently.

Provides:
- build_llm(..., json_mode=True): returns configured ChatOpenAI client.
- invoke_with_fallback(...): tries JSON mode first, then falls back to text mode.

These helpers keep construction consistent across modules and centralize
minor differences in LangChain/OpenAI kwargs.
"""

from typing import Optional, Tuple, Any

from sme_causal.core.llm_clients.factory import get_llm_client
"""Invoke ChatOpenAI in JSON mode, falling back to text mode on failure.

    Args:
        messages: Formatted messages for LangChain `invoke`.
        model: Model name.
        temperature: Temperature value.
        api_key: API key or None to rely on env.
        seed: Optional seed.
        top_p: Optional top_p.

    Returns:
        Tuple of (content, raw_message, used_json_mode).
    """

def invoke_with_fallback(
    messages: Any,
    *,
    model: str,
    temperature: float,
    api_key: Optional[str],   # can be ignored and taken from cfg
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[str, Any, bool]:
    client = get_llm_client()  # made once with LLM_PROVIDER
    if model and model != client.model:
        # optionally override model on each call
        client.model = model
    return client.invoke_with_fallback(
        messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

