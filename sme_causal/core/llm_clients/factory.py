from __future__ import annotations


from sme_causal.core.config import get_config

from .openai import OpenAIClient
from .local import LocalClient
from .base import LLMClient

_client_singleton: LLMClient | None = None

def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    cfg = get_config()
    provider = cfg.effective_llm_provider
    model = cfg.llm.model_name
    api_key = cfg.effective_llm_api_key
    base_url = cfg.effective_llm_base_url  # None for openai

    if provider == "local":
        _client_singleton = LocalClient(model=model, api_key=api_key, base_url=base_url)
    else:
        _client_singleton = OpenAIClient(model=model, api_key=api_key, base_url=None)

    return _client_singleton

def reset_llm_client() -> None:
    """Если в UI поменяли провайдера — сбросить синглтон и создать заново."""
    global _client_singleton
    _client_singleton = None
