from __future__ import annotations
from typing import Any, Optional, Tuple, Iterable
from langchain_openai import ChatOpenAI, OpenAI  
from .base import LLMClient

def _messages_to_prompt(messages: Iterable[Any]) -> str:
    # Простейшая склейка истории в один prompt; можно улучшить по вкусу
    parts = []
    for m in messages:
        # LangChain может отдавать AIMessage/UserMessage/SystemMessage
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        content = getattr(m, "content", str(m))
        if role == "system":
            parts.append(f"[system]\n{content}\n")
        elif role == "ai" or role == "assistant":
            parts.append(f"[assistant]\n{content}\n")
        else:
            parts.append(f"[user]\n{content}\n")
    return "\n".join(parts).strip()

class LocalClient(LLMClient):
    def _build_text_llm(
        self,
        *,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> OpenAI:
        kwargs = dict(
            model=self.model,
            api_key=self.api_key,
            temperature=temperature,
        )
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # seed у OpenAI(text) может игнорироваться, не критично
        return OpenAI(**kwargs)  # type: ignore[arg-type]

    def invoke_with_fallback(
        self,
        messages: Any,
        *,
        temperature: float,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, Any, bool]:
        # 1) пробуем чат JSON
        try:
            llm_json = self._build_llm(
                temperature=temperature,
                top_p=top_p,
                json_mode=True,
                max_tokens=max_tokens,
                seed=seed,
            )
            # по дефолту максимальное окно контекста 32k токенов, но можно редактировать в docker-compose
            # в классе существует функция обрезки промпта в токенах truncate_messages, 
            # на случай уменьшения контекста/увеличения промпта
            msg = llm_json.invoke(messages)

            return msg.content, msg, True
        except Exception as e_json:
            # 2) если упали из-за chat template — сразу уходим в completions
            if "chat template" in str(e_json).lower():
                prompt = _messages_to_prompt(messages)
                llm_text = self._build_text_llm(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                text = llm_text.invoke(prompt)

                # raw можно отдать словарём для логов
                return (text if isinstance(text, str) else str(text)), {"via": "completions"}, False

            # 3) иначе ещё попробуем чат БЕЗ json_mode
            try:
                prompt = _messages_to_prompt(messages)
                llm_text = self._build_llm(
                    temperature=temperature,
                    top_p=top_p,
                    json_mode=False,
                    max_tokens=max_tokens,
                    seed=seed,
                )

                msg = llm_text.invoke(prompt)

                return msg.content, msg, False
            except Exception as e_txt:
                if "chat template" in str(e_txt).lower():
                    prompt = _messages_to_prompt(messages)
                    llm_text = self._build_text_llm(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=seed,
                    )
                    
                    text = llm_text.invoke(prompt)

                    return (text if isinstance(text, str) else str(text)), {"via": "completions"}, False
                raise  # другие ошибки дальше по стеку
