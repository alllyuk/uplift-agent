"""
RAGAS eval (no ground truth) + local embeddings intfloat/multilingual-e5-base
TF disabled. LLM adapter supports async (RAGAS awaits it).
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset
from loguru import logger

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from sentence_transformers import SentenceTransformer

from sme_causal.agent.agent_service import CausalAgent
from sme_causal.rag.rag_pipeline import RAG
from sme_causal.core.config import get_config


def _prompt_to_str(prompt: Any) -> str:
    if prompt is None:
        return ""
    if hasattr(prompt, "text"):
        t = getattr(prompt, "text")
        if isinstance(t, str):
            return t
    if isinstance(prompt, dict) and "text" in prompt and isinstance(prompt["text"], str):
        return prompt["text"]
    return str(prompt)


# --- минимальные структуры под ожидания RAGAS (LLMResult/Generation-like) ---

class _Gen:
    def __init__(self, text: str):
        self.text = text

class _LLMResult:
    def __init__(self, generations: List[List[_Gen]]):
        self.generations = generations


class AgentLLMAdapter:
    """
    Твоя версия RAGAS делает: await llm.generate(...)
    и ожидает вернуть LLMResult-подобный объект с .generations.
    """

    def __init__(self, agent: CausalAgent):
        self.agent = agent

    def _one_sync(self, prompt_str: str) -> str:
        msgs = [
            {"role": "system", "content": "You are an evaluator for RAG outputs."},
            {"role": "user", "content": prompt_str},
        ]
        return self.agent._invoke_with_fallback(msgs, context_hint="ragas_evaluator")

    async def generate(self, prompt: Any, n: int = 1, **kwargs: Any) -> _LLMResult:
        prompt_str = _prompt_to_str(prompt)

        if n is None or n <= 1:
            text = await asyncio.to_thread(self._one_sync, prompt_str)
            return _LLMResult(generations=[[ _Gen(text) ]])

        gens: List[_Gen] = []
        for _ in range(int(n)):
            text = await asyncio.to_thread(self._one_sync, prompt_str)
            gens.append(_Gen(text))
        return _LLMResult(generations=[gens])



class LocalE5Embeddings:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str = "cpu"):
        logger.info(f"Loading embeddings: {model_name} device={device}")
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([f"query: {text}"])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode([f"passage: {t}" for t in texts])


@dataclass
class RagAnswer:
    answer: str
    contexts: List[str]


def _chunk_to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("text", "content", "chunk", "page_content"):
            v = x.get(key)
            if isinstance(v, str):
                return v
        return str(x)
    for attr in ("text", "content", "page_content"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, str):
                return v
    return str(x)


class SimpleRAGPipeline:
    def __init__(self, cfg, agent: CausalAgent, top_k: int = 5):
        self.cfg = cfg
        self.agent = agent
        self.top_k = top_k
        self.rag = RAG()

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        ctx_block = "\n\n---\n\n".join(contexts) if contexts else "(no contexts)"
        prompt = (
            "Answer the question using ONLY the provided contexts.\n"
            "If the contexts do not contain enough information, say you don't know.\n\n"
            f"Question:\n{question}\n\n"
            f"Contexts:\n{ctx_block}\n\n"
            "Answer:"
        )
        msgs = [
            {"role": "system", "content": "You are a helpful assistant answering strictly from context."},
            {"role": "user", "content": prompt},
        ]
        return self.agent._invoke_with_fallback(msgs, context_hint="rag_answer_generation")

    def answer_with_contexts(self, question: str) -> RagAnswer:
        results = self.rag.perform_query(question, top_k=self.top_k)
        contexts = [_chunk_to_text(r).strip() for r in results]
        contexts = [c for c in contexts if c]
        answer = self._generate_answer(question, contexts)
        return RagAnswer(answer=answer, contexts=contexts)


def build_ragas_dataset(rag: SimpleRAGPipeline, questions: List[str]) -> Dataset:
    rows: List[Dict] = []
    for q in questions:
        out = rag.answer_with_contexts(q)
        rows.append({"question": q, "answer": out.answer, "contexts": out.contexts})
    return Dataset.from_list(rows)


def main() -> None:
    cfg = get_config()
    logger.info("Starting RAGAS evaluation (LOCAL E5, TF disabled)")

    agent = CausalAgent(graph_method=None)
    llm_adapter = AgentLLMAdapter(agent)

    rag = SimpleRAGPipeline(cfg=cfg, agent=agent, top_k=5)

    questions = [
        "Как изменение кредитного лимита влияет на клиента и его активность?",
        "Как предложение нового продукта эквайринг влияет на клиента и его активность?",
        "Как предложение льготного тарифа влияет на клиента и его активность?",
    ]

    dataset = build_ragas_dataset(rag, questions)
    logger.info(f"Dataset size: {len(dataset)}")

    embeddings = LocalE5Embeddings("intfloat/multilingual-e5-base", device="cpu")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm_adapter,
        embeddings=embeddings,
    )

    print("\n=== RAGAS SCORES ===")
    print(result)

    try:
        print("\n=== DETAILS ===")
        print(result.to_pandas())
    except Exception:
        pass


if __name__ == "__main__":
    main()
