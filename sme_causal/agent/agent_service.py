from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from sme_causal.core.config import get_config
from sme_causal.core.columns import (
    CONTEXT_FIELDS as CORE_CONTEXT_FIELDS,
    INTERVENTIONS_RANGES,
    CLIENT_ID,
)
from sme_causal.core.llm import invoke_with_fallback
from sme_causal.rag.rag_pipeline import RAG
from sme_causal.core.utils import create_query, parse_json_obj_from_text


# Поля, которые формируют контекст клиента (совпадают с UI)
# Re-export canonical context fields for backward compatibility
CONTEXT_FIELDS: List[str] = CORE_CONTEXT_FIELDS

PREDICT_NUMBER_PHRASE = (
    "ДОПОЛНИТЕЛЬНОЕ ТРЕБОВАНИЕ К ОТВЕТУ:\n"
    "• Обязательно оцени ориентировочное числовое значение целевой метрики "
    "Revenue_Growth_Rate (в п.п. или процентах) на основе переданных данных "
    "и явно запиши это число в поле expected_effect, указав кратко источник "
    "(только из реально переданных источников: PSM/RAG/GRAPH/PROFILE)."
)


GRAPH_RESPONSE_RULES = (
    "Правила использования графа в ответе:\n"
    "• Используйте ТОЛЬКО рёбра из блока [GRAPH_DSL]; не придумывайте новые рёбра.\n"
    "• Поле sign/conf используйте только внутренне: sign задаёт направление эффекта, conf задаёт доверие.\n"
    "• В пользовательском ответе НЕ пишите технические маркеры 'sign', 'conf' и шаблоны вида '(sign:+, conf=0.7)'.\n"
    "• Пользовательская формулировка для графа: 'По причинному графу: A положительно/отрицательно влияет на B; доверие высокое/среднее/низкое: ...'.\n"
    "• Если нет указанного ребра или короткой цепочки от интервенции к целевой метрике, пишите: 'В графе нет подтверждённого пути от A к B'.\n"
    "• Не начинайте отсутствующую связь словами 'На основании ребра A -> B'. Так можно писать только для реально переданного ребра.\n"
    "• Не используйте формулировку 'связь между A и B': она скрывает направление причинной связи.\n"
)


def _confidence_label(raw_conf: str) -> str:
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        return "не указано"
    if conf >= 0.75:
        return "высокое"
    if conf >= 0.5:
        return "среднее"
    return "низкое"


def _effect_label(sign: str) -> str:
    if sign == "+":
        return "положительно влияет на"
    if sign == "-":
        return "отрицательно влияет на"
    return "направленно связан с"


def _clean_public_graph_terms(text: str) -> str:
    """Hide internal graph DSL tokens from user-facing explanations."""
    if not text:
        return text

    def replace_edge_reference(match: re.Match) -> str:
        source, target, sign, conf = match.groups()
        return (
            f"По причинному графу: {source} {_effect_label(sign)} {target}; "
            f"доверие {_confidence_label(conf)}:"
        )

    cleaned = re.sub(
        r"На основании(?:\s+ребра)?\s+([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*"
        r"\(\s*sign\s*:\s*([+\-?])\s*,\s*conf\s*[:=]\s*([0-9.]+)\s*\)\s*:",
        replace_edge_reference,
        text,
    )
    cleaned = re.sub(
        r"\s*\(\s*conf\s*[:=]?\s*[0-9.]+\s*\)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*\(\s*sign\s*:\s*[+\-?]\s*,\s*conf\s*[:=]\s*[0-9.]+\s*\)",
        "",
        cleaned,
    )
    return cleaned

FEATURES_DESCRIPTION = (
    "Client Profile (Macro)\n"
    "- Industry: Сектор деятельности (OKVED). Влияет на риски и потребности.\n"
    "- Region: Регион работы. Экономические условия и политика региона могут влиять на результаты.\n"
    "- Business_Size: Размер компании (микро, малый). Определяет ресурсы и устойчивость.\n"
    "- Years_in_Operation: Возраст бизнеса в годах. Ориентиры: 0-2 = молодой, 3-10 = зрелый, 11+ = долгий срок.\n\n"
    "Bank–Client Relationship\n"
    "- Client_Tenure: Срок работы с банком в годах. Ориентиры: 0-1 = короткий, 2-6 = средний, 7+ = долгий.\n"
    "- Num_Products: Количество продуктов у клиента. Больше продуктов = более глубокие отношения.\n"
    "- Has_Card / Has_Deposit / Has_Acquiring / Has_Loan / Has_Payroll: Булевые признаки наличия ключевых банковских продуктов (карта, депозит, эквайринг, кредит, зарплатный проект). Показывают глубину отношений.\n"
    "- Total_Bank_Profit: Совокупная прибыль от клиента (LTV). Важен при решении, как активно удерживать или развивать. Включает комиссионный доход.\n\n"
    "Transactional Behavior\n"
    "- Avg_Monthly_Inflow: Средний месячный приток средств (обороты клиента). Прокси выручки.\n"
    "- Avg_Monthly_Outflow: Средние расходы. Показывают структуру затрат.\n"
    "- Net_Cashflow: Разница между притоком и расходами. Положительное значение = избыточная ликвидность, отрицательное = возможный дефицит.\n"
    "- Monthly_Transaction_Count: Количество транзакций в месяц. Отражает интенсивность деятельности.\n"
    "- Avg_Account_Balance: Средний баланс на счёте. Признак ликвидности.\n"
    "- Balance_Bucket: Категория баланса (высокий/средний/низкий). Упрощённая характеристика уровня ликвидности.\n"
    "- Activity_Bucket: Категория активности (высокая/средняя/низкая). Упрощённая характеристика транзакционной активности.\n\n"
    "Interventions\n"
    "- New_Product_Offer: Новый продукт/услуга, предложенные клиенту.\n"
    "- Credit_Limit_Change: Изменение кредитного лимита. Влияет на доступ к средствам.\n"
    "- Tariff_Discount: Скидка или временное снижение тарифов/комиссий (например, на РКО или эквайринг). Может стимулировать рост транзакций и использования продуктов.\n\n"
    "Outcomes, Target\n"
    "- Revenue_Growth_Rate / Avg_Account_Balance / etc.: Изменение ключевых показателей. Целевая метрика для оценки эффекта интервенций.\n"
)


@dataclass
class Explanation:
    diagnosis: str
    drivers_pos: List[str]
    drivers_neg: List[str]
    recommendations: List[str]
    expected_effect: str
    raw_text: str  # оригинальный ответ LLM (для логов/дебага)

    # Context for evaluation purposes (captured during generation)
    full_prompt: Optional[str] = None  # Full formatted prompt sent to LLM
    graph_context: Optional[str] = None  # Graph DSL if used
    rag_context: Optional[str] = None  # RAG enrichment if used
    psm_summary: Optional[str] = None  # PSM metrics if used
    base_context: Optional[Dict] = None  # Client profile context
    delta_changes: Optional[Dict] = None  # What-if intervention changes

    def to_dict(self, include_debug: bool = False) -> Dict:
        """Serialize explanation to a plain dict.

        Returns:
            Dict representation. Debug fields are omitted unless requested.
        """
        data = asdict(self)
        if include_debug:
            return data
        for key in (
            "raw_text",
            "full_prompt",
            "graph_context",
            "rag_context",
            "psm_summary",
            "base_context",
            "delta_changes",
        ):
            data.pop(key, None)
        return data

    def to_pretty_string(self) -> str:
        """Render a human-readable, multiline string for console output.

        Returns:
            str: Nicely formatted text with structured fields when available.
                 Falls back to pretty-printed JSON from ``raw_text`` or the raw
                 string if JSON parsing fails.
        """
        lines: List[str] = []
        has_struct: bool = any(
            [
                bool(self.diagnosis),
                bool(self.drivers_pos),
                bool(self.drivers_neg),
                bool(self.recommendations),
                bool(self.expected_effect),
            ]
        )

        if has_struct:
            if self.diagnosis:
                lines.append(f"Диагноз: {self.diagnosis}")

            if self.drivers_pos:
                lines.append("Драйверы роста (+):")
                lines.extend([f"  - {item}" for item in self.drivers_pos])

            if self.drivers_neg:
                lines.append("Сдерживающие факторы (−):")
                lines.extend([f"  - {item}" for item in self.drivers_neg])

            if self.recommendations:
                lines.append("Рекомендации:")
                lines.extend(
                    [
                        f"  {idx + 1}. {rec}"
                        for idx, rec in enumerate(self.recommendations)
                    ]
                )

            if self.expected_effect:
                lines.append(f"Ожидаемый эффект: {self.expected_effect}")

            return "\n".join(lines)

        # Fallback: pretty print the raw JSON if possible, otherwise raw text
        try:
            parsed = json.loads(self.raw_text)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return self.raw_text.strip()

    def __str__(self) -> str:
        """String representation used by ``print()``.

        Returns:
            str: Human-readable text built by :meth:`to_pretty_string`.
        """
        return self.to_pretty_string()


class CausalAgent:
    """Сервис объяснений и what‑if.

    Supports deterministic generation via optional ``seed``.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        seed: int | None = None,
        top_p: float | None = None,
        api_key: Optional[str] = None,
        graph_method: Optional[str] = None,
    ):
        # Defaults from central config when args are not provided
        cfg = get_config()
        self.model_name = model or cfg.llm.model_name
        self.confidence_threshold = float(cfg.llm.confidence_threshold)
        self.temperature = (
            float(temperature)
            if temperature is not None
            else float(cfg.llm.temperature)
        )
        self.seed = seed if seed is not None else int(cfg.data_generation.seed)
        self.top_p = top_p
        # Prefer explicitly provided key; otherwise use effective config/env
        self.api_key = api_key or cfg.effective_openai_api_key

        # Optional graph path (JSON from LLM edge inference step)
        if graph_method == "hybrid" and getattr(cfg, "full_artifacts_dir", ""):
            self.graph_path = cfg.full_artifacts_dir / "hybrid_edges.json"
        elif graph_method == "algo" and getattr(cfg, "algo_edges_path", ""):
            self.graph_path = Path(cfg.algo_edges_path)
        elif graph_method == "llm" and getattr(cfg, "llm_edges_path", ""):
            self.graph_path = Path(cfg.llm_edges_path)
        elif graph_method == "algo_llm" and getattr(cfg, "algorithmic_dir", ""):
            self.graph_path = cfg.full_algorithmic_dir / "algo_llm_edges.json"
        else:
            self.graph_path = None
        self._graph_dsl_cache: Optional[str] = None  # кэш строк DSL

        # Текущий режим (для логгирования в метаданных); фактический вызов идет через общий fallback-хелпер
        self._json_mode = True

        self._features_description = FEATURES_DESCRIPTION

        # Пример JSON-схемы — БЕЗ фигурных скобок в самом шаблоне!
        self._schema_example = (
            "{\n"
            '  "drivers_pos": ["Фактор роста: кратко (<=15 слов). Если RAG-документы ПЕРЕДАНЫ - используйте хотя бы 1 RAG-аргумент с doc_id и короткой цитатой. Если граф ПЕРЕДАН и релевантен - пишите по-человечески: По причинному графу: A положительно влияет на B; доверие высокое: ..."],\n'
            '  "drivers_neg": ["Сдерживающий фактор: кратко (<=15 слов). Если RAG-документы ПЕРЕДАНЫ - используйте хотя бы 1 RAG-аргумент с doc_id и короткой цитатой. Если граф ПЕРЕДАН и релевантен - пишите по-человечески: По причинному графу: C отрицательно влияет на B; доверие среднее: ..."],\n'
            '  "expected_effect": "Ожидаемый эффект от интервенции. Используйте только реально переданные источники и не выводите sign/conf."\n'
            "}"
        )

        self._base_schema_example = (
            "{\n"
            '  "drivers_pos": ["Фактор роста: кратко (<=15 слов). Если граф ПЕРЕДАН и релевантен - пишите: По причинному графу: A положительно влияет на B; доверие высокое: ..."],\n'
            '  "drivers_neg": ["Сдерживающий фактор: кратко (<=15 слов). Если граф ПЕРЕДАН и релевантен - пишите: По причинному графу: C отрицательно влияет на B; доверие среднее: ..."],\n'
            '  "recommendations": ["Краткая практическая рекомендация. Если граф ПЕРЕДАН и релевантен - объясните через причинный граф без sign/conf."],\n'
            "}"
        )

        # Базовое объяснение
        self._prompt_base = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Вы — старший аналитик ММБ банка с экспертизой в области клиентского анализа и продуктового развития.\n\n"
                    "ВАШИ ЗАДАЧИ:\n"
                    "1. Проанализировать профиль клиента и его финансовые показатели\n"
                    "2. Выявить ключевые драйверы роста и риски\n"
                    "3. Предложить конкретные рекомендации для развития клиента\n\n"
                    "МАТЕРИАЛЫ ДЛЯ АНАЛИЗА:\n"
                    "• Базовый профиль клиента\n"
                    "• Описание признаков\n"
                    "• (Опионально) причинно-следственный граф\n"
                    "ТРЕБОВАНИЯ К ОТВЕТУ:\n"
                    "• Отвечайте СТРОГО на русском языке\n"
                    "• Используйте ТОЛЬКО предложенную информацию, не придумывайте новых связей.\n"
                    "• Если граф НЕ передан — не ссылайтесь на рёбра и не употребляйте формулы вида 'На основании A -> B'.\n"
                    "• Используйте ТОЛЬКО JSON-формат по схеме ниже\n"
                    "• Будьте краткими и конкретными (до 15 слов на пункт)\n"
                    "ЕСЛИ ПЕРЕДАН ГРАФ (блок [GRAPH_DSL]):\n"
                    f"{GRAPH_RESPONSE_RULES}"
                    "• Фокусируйтесь на практических аспектах реализации\n\n"
                    "СХЕМА ОТВЕТА:\n"
                    "{base_schema_example}",
                ),
                (
                    "user",
                    "{graph_dsl_block}\n\n"
                    "ПРОФИЛЬ КЛИЕНТА И МЕТРИКИ:\n{base_json}\n\n"
                    "ОПИСАНИЕ ПРИЗНАКОВ:\n{features_description}\n\n"
                    "{target_metric_info}\n"
                    "ПРОАНАЛИЗИРУЙТЕ и дайте 2-3 конкретные рекомендации для развития клиента.",
                ),
            ]
        )

        # Сценарий WHAT_IF (сравнение)
        self._prompt_whatif = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Вы — старший аналитик ММБ банка, специализирующийся на сценарном анализе и оценке влияния изменений.\n\n"
                    "ВАША ЗАДАЧА:\n"
                    "Сравнить базовый сценарий клиента с предложенными изменениями (WHAT_IF) и оценить их влияние.\n\n"
                    "Вы обязаны учитывать переданные источники: профиль, PSM-оценку, RAG-чанки, граф (если они есть).\n"
                    "Главная цель — показать, что модель умеет ДИСКОНТИРОВАТЬ источники: выбирать, чему доверять — PSM или RAG (если есть RAG и PSM).\n"
                    "АНАЛИЗ ДОЛЖЕН ВКЛЮЧАТЬ:\n"
                    "• Оценку влияния изменений на ключевые показатели\n"
                    "• Выявление новых возможностей и рисков\n"
                    "• Прогноз ожидаемых результатов\n\n"
                    "МАТЕРИАЛЫ ДЛЯ АНАЛИЗА:\n"
                    "• Базовый профиль клиента\n"
                    "• Описание признаков\n"
                    "• (Опионально) причинно-следственный граф\n"
                    "• (Опционально) PSM-оценка влияния интервенции\n"
                    "• (Опционально) контекст, предоставленный RAG-системой\n"
                    "ЖЁСТКИЕ ТРЕБОВАНИЯ К ОТВЕТУ:\n"
                    "• Отвечайте СТРОГО на русском языке\n"
                    "• Если переданы и граф, и RAG-контекст — УЧИТЫВАЙТЕ ОБА. Игнорирование RAG считается ошибкой, если RAG передан в запрос.\n"
                    "• Если ПЕРЕДАН PSM: не просто укажи PSM-числа, но и сделай короткий вывод на их основе\n"
                    "• Используйте ТОЛЬКО предложенную информацию, не придумывайте новых связей.\n"
                    "• Для КАЖДОГО пункта, основанного на RAG, пишите: «На основании документа <(doc_id)>: ...»,\n"
                    "  и добавляйте В СКОБКАХ КОРОТКУЮ ЦИТАТУ из чанка (≤20 слов), подсказавшую вывод. Сделай короткий вывод на основании цитаты (3-7 слов).\n"
                    "• Если контекстные документы (RAG) НЕ переданы, НЕ используйте их в ответе, НЕ ссылайтесь на них и не пишите 'документ n/a'.\n"
                    "• В разделах drivers_pos и drivers_neg при наличии RAG — минимум по 1 пункту, опирающемуся на документ.\n"
                    "• В drivers_pos и drivers_neg: КАЖДЫЙ пункт, который содержит ЧИСЛА, должен заканчиваться хвостом:«(источник: <PSM|RAG|GRAPH|PROFILE>; объяснение: <2–10 слов>)».\n"
                    "Для PSM-цифр: в ОДНОМ из пунктов drivers_pos/neg сделайте микро-вывод на основе ATE/ATT и n_treated/n_control: пример: «… (источник: PSM; объяснение: ATT=0.003, n_treated=300, n_control=300 → эффект слабый)»\n"
                    "• Если граф НЕ передан — не ссылайтесь на рёбра и не употребляйте формулы вида 'На основании A -> B'.\n"
                    "• В expected_effect после 'Приоритет:' перечисляйте только реально переданные источники: RAG только если есть непустой RAG-контекст, GRAPH только если есть [GRAPH_DSL], PSM только если есть блок PSM.\n"
                    "• Если ПЕРЕДАНЫ и RAG, и PSM — в expected_effect ЯВНО укажите, на какой источник опираются ЧИСЛА и ПОЧЕМУ:\n"
                    "пример: «Приоритет: PSM → ожидаем +0.4 п.п.; цифры из PSM (ATT=0.004, n_treated=320, n_control=320), RAG вторично: подтверждает контекст».\n"
                    "пример: «Приоритет: RAG → ожидаем рост транзакций ~2–3%; цифры из RAG (doc_12, doc_09), PSM ≈ 0».\n\n"
                    "ТРЕБОВАНИЯ К ДИСКОНТИРОВАНИЮ:\n"
                    "• primary_effect_pp = ATT, если ATT доступен для matched sample; ATE — только наивный ориентир и не может быть primary для персонального what-if.\n"
                    "\n"
                    "ПРИОРИТЕТ PSM, ЕСЛИ ВСЕ ВЕРНО:\n"
                    "1) psm_reliable=true\n"
                    "2) |primary_effect_pp| > 0.001  // эффект отличается от нуля\n"
                    "3) n_treated ≥ 100 и n_control ≥ 100\n"
                    "ПРИОРИТЕТ RAG, ЕСЛИ ВСЕ ВЕРНО:\n"
                    "1) |primary_effect_pp| ≤ 0.001  // эффект около нуля\n"
                    "2) ≥ 2 согласованных RAG-документов с объяснимыми числовыми фактами\n"
                    "3) Эти факты применимы к профилю клиента\n"
                    "ИНАЧЕ:\n"
                    '• decision = "neutral" и объяснить, каких данных не хватает\n\n'
                    "• Используйте ТОЛЬКО JSON-формат по схеме ниже\n"
                    "• Каждую цифру объяснить: что это, откуда, как интерпретировать\n"
                    "• Для RAG-ссылок: указывать doc_id и краткую цитату ≤ 20 слов\n"
                    "• Для PSM-чисел: ясно указать ATE/ATT, n_treated, n_control\n"
                    "• Будьте краткими и конкретными (до 15 слов на пункт)\n"
                    "ЕСЛИ ПЕРЕДАН ГРАФ (блок [GRAPH_DSL]):\n"
                    f"{GRAPH_RESPONSE_RULES}"
                    "• При интерпретации A -> B считайте, что A влияет на B. Не допускайте формулировок, где B влияет на A.\n"
                    "• Если граф предполагает наличие направленной цепочки из 2-3 ребер от A к B, можно использовать эту цепочку в ответе. "
                    "Но если связь длинная, или нет явной цепочки зависимостей, не используй эту цепочку связей.\n"
                    "• Фокусируйтесь на практических аспектах реализации\n\n"
                    "{predict_number_block}\n"
                    "СХЕМА ОТВЕТА:\n"
                    "{schema_example}",
                ),
                (
                    "user",
                    "{graph_dsl_block}{rag_context_block}{source_availability_block}"
                    # Блок для информации о неточном совпадении
                    "{match_info_block}\n" "БАЗОВЫЙ ПРОФИЛЬ КЛИЕНТА:\n{base_json}\n\n"
                    # Заголовок и логика меняются в зависимости от наличия delta
                    "{what_if_block}"
                    "{psm_summary}"
                    "ОПИСАНИЕ ПРИЗНАКОВ:\n{features_description}\n\n"
                    # Формулировка задачи меняется в зависимости от наличия delta
                    "{task_instructions}",
                ),
            ]
        )

    # -------------------- публичный API --------------------
    def build_context_for_client(self, df: pd.DataFrame, client_id: str) -> Dict:
        """Build LLM context for a specific client ID.

        Raises ValueError if the client_id is not present in the DataFrame.
        """
        dff = df[df[CLIENT_ID] == client_id]
        if dff.empty:
            raise ValueError(f"Client_ID not found: {client_id}")
        row = dff.iloc[0].to_dict()
        return {k: row[k] for k in CONTEXT_FIELDS if k in row}

    def explain_client(
        self,
        client_ctx: Dict,
        use_graph: bool = False,
        min_conf: float = None,
        target_metric: Optional[str] = None,
    ) -> Explanation:
        graph_dsl_block = ""

        effective_target_metric = target_metric or "Revenue_Growth_Rate"

        if use_graph:
            min_conf = min_conf if min_conf is not None else self.confidence_threshold
            graph_dsl = self._load_graph_dsl(min_conf=min_conf)
            if graph_dsl:
                # Можно явно указать outcome, если обнаружен узел 'Revenue_Growth_Rate'
                outcome_hint = (
                    effective_target_metric
                    if effective_target_metric in graph_dsl
                    else None
                )
                header = "[GRAPH_DSL]\n"
                if outcome_hint:
                    header = f"Целевая метрика (outcome): {outcome_hint}\n" + header
                guidance = (
                    f"{GRAPH_RESPONSE_RULES}\n"
                )
                graph_dsl_block = header + guidance + graph_dsl + "\n\n"

        target_metric_info = (
            f"ОСНОВНАЯ ЦЕЛЕВАЯ МЕТРИКА ДЛЯ АНАЛИЗА: {effective_target_metric}\n\n"
            "КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ДЛЯ АНАЛИЗА:\n"
            "• Отрасль и размер бизнеса (Industry, Business_Size)\n"
            "• Финансовые показатели (Avg_Account_Balance, Avg_Monthly_Inflow/Outflow)\n"
            "• Продуктовая линейка (Num_Products, Product_Types)\n"
            f"• Целевая метрика ({effective_target_metric})\n"
        )

        msgs = self._prompt_base.format_messages(
            base_schema_example=self._base_schema_example,
            base_json=json.dumps(client_ctx, ensure_ascii=False, sort_keys=True),
            graph_dsl_block=graph_dsl_block,
            features_description=self._features_description,
            target_metric_info=target_metric_info,
        )

        # Build full prompt text for evaluation purposes
        full_prompt_parts = []
        for msg in msgs:
            if hasattr(msg, 'content'):
                full_prompt_parts.append(msg.content)
            elif isinstance(msg, dict) and 'content' in msg:
                full_prompt_parts.append(msg['content'])
        full_prompt_text = "\n\n".join(full_prompt_parts) if full_prompt_parts else None

        resp = self._invoke_with_fallback(msgs, context_hint="explain_client")

        return self._parse_explanation(
            resp,
            full_prompt=full_prompt_text,
            graph_context=graph_dsl_block if use_graph else None,
            base_context=client_ctx,
        )

    def explain_what_if(
        self,
        base_ctx: Dict,
        delta_changes: Dict,
        psm_metrics: Optional[Dict[str, object]] = None,
        use_graph: bool = False,
        use_rag: bool = False,
        min_conf: float = None,
        rag_query_text: Optional[str] = None,
        match_info: Optional[Dict] = None,
        target_metric: Optional[str] = None,
        predict_concrete_target: bool = False,
    ) -> Explanation:
        # Только реально изменившиеся поля
        delta = {k: v for k, v in delta_changes.items() if base_ctx.get(k) != v}

        # Блок для информации о неточных совпадениях
        match_info_block = ""
        if match_info:
            similar_matches = []
            for col, info in match_info.items():
                if info.get("status") == "similar":
                    similar_matches.append(
                        f"- Поле `{col}` было подобрано как наиболее похожее на запрос «{info.get('query_phrase', '')}». Это приблизительное совпадение."
                    )
            if similar_matches:
                match_info_block = (
                    "ВАЖНОЕ ЗАМЕЧАНИЕ О НЕТОЧНОСТИ ЗАПРОСА:\n"
                    "Следующие интервенции были определены не по точному совпадению, а по смысловой близости. "
                    "Отнеситесь к выводам по ним с меньшей уверенностью, так как интерпретация может быть не совсем верной.\n"
                    + "\n".join(similar_matches)
                    + "\n\n"
                )

        # Динамическое формирование блоков промпта в зависимости от наличия delta
        what_if_block = ""
        task_instructions = ""
        if delta:
            # Сценарий с найденной интервенцией (хотя бы похожей по смыслу)
            what_if_block = f"ПРЕДЛАГАЕМЫЕ ИЗМЕНЕНИЯ (WHAT_IF):\n{json.dumps(delta, ensure_ascii=False, sort_keys=True)}\n\n"
            task_instructions = (
                "ПРОАНАЛИЗИРУЙТЕ:\n"
                "• Как изменения повлияют на текущую ситуацию клиента\n"
                "• Какие новые возможности откроются\n"
                "• Какие риски могут возникнуть\n"
                "• Что нужно сделать для успешной реализации\n"
                "• Какой результат ожидается"
            )
        else:
            # Сценарий без конкретной интервенции (не распознана даже похожая интервенция)
            what_if_block = (
                "КОНКРЕТНАЯ ИНТЕРВЕНЦИЯ НЕ РАСПОЗНАНА.\n"
                f"Исходный запрос пользователя для анализа: «{rag_query_text}»\n\n"
            )
            task_instructions = (
                "ЗАДАЧА:\n"
                "Так как конкретная интервенция не была распознана, ваша задача — дать оценку и рекомендации для клиента.\n"
                "Используйте исходный запрос пользователя как контекст его интересов.\n"
                "Опирайтесь только на RAG-контекст (если он передан) для поиска релевантных идей и предложений.\n"
                "Можете также использовать свои знания законов рынка и экономики а также банковской деятельности, в которых уверены.\n"
                "Сформулируйте 2-3 наиболее вероятные гипотезы или рекомендации, которые соответствуют интересам пользователя."
            )

        def _format_ate(value) -> str:
            if value is None:
                return "нет данных"
            try:
                num = float(value)
            except (TypeError, ValueError):
                return str(value)
            if math.isnan(num):
                return "нет данных"
            return f"{num:.4f}"

        def _format_count(value) -> str:
            if value is None:
                return "нет данных"
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return str(value)

        psm_summary = ""
        # PSM-анализ запускается только если есть delta из запроса пользователя
        if psm_metrics and delta:
            att = psm_metrics.get("att")
            if att is None and "ate_matched" in psm_metrics:
                att = psm_metrics.get("ate_matched")

            ate = psm_metrics.get("ate")
            if ate is None and "ate_naive" in psm_metrics:
                ate = psm_metrics.get("ate_naive")
            if ate is None and "naive_ate" in psm_metrics:
                ate = psm_metrics.get("naive_ate")

            if att is not None or ate is not None:
                outcome_col_name = psm_metrics.get(
                    "outcome_col", target_metric or "Revenue_Growth_Rate"
                )
                n_pairs = psm_metrics.get("n_pairs")
                n_treated = psm_metrics.get("n_treated")
                n_control = psm_metrics.get("n_control")
                psm_reliable = bool(psm_metrics.get("psm_reliable"))
                psm_reason = psm_metrics.get("psm_reason") or (
                    "PSM can be primary only when ATT is available and matched groups are large enough."
                )
                psm_summary = (
                    f"РЕЗУЛЬТАТЫ PSM-АНАЛИЗА (целевая метрика: {outcome_col_name}):\n"
                    f"• ATT (matched sample): {_format_ate(att)}\n"
                    f"• ATE (naive baseline): {_format_ate(ate)}\n\n"
                    f"• n_pairs: {_format_count(n_pairs)}\n"
                    f"• n_treated: {_format_count(n_treated)}\n"
                    f"• n_control: {_format_count(n_control)}\n"
                    f"• psm_reliable: {str(psm_reliable).lower()}\n"
                    f"• reliability_reason: {psm_reason}\n\n"
                    "Если psm_reliable=false, не выбирайте PSM как приоритетный источник: "
                    "используйте ATE только как слабый наивный ориентир и явно объясните ограничение.\n\n"
                )
        effective_target_metric = target_metric or "Revenue_Growth_Rate"

        graph_dsl_block = ""
        if use_graph:
            min_conf = min_conf if min_conf is not None else self.confidence_threshold
            graph_dsl = self._load_graph_dsl(min_conf=min_conf)
            if graph_dsl:
                outcome_hint = (
                    effective_target_metric
                    if effective_target_metric in graph_dsl
                    else None
                )
                header = "[GRAPH_DSL]\n"
                if outcome_hint:
                    header = f"Целевая метрика (outcome): {outcome_hint}\n" + header
                guidance = (
                    f"{GRAPH_RESPONSE_RULES}\n"
                )
                graph_dsl_block = header + guidance + graph_dsl + "\n\n"

        if use_rag:
            rag = RAG()
            if rag_query_text:
                logger.info(f"Using original query for RAG: '{rag_query_text}'")
                delta_txt = rag_query_text
            else:
                delta_txt = create_query(delta)
                logger.info(f"Using constructed query for RAG: '{delta_txt}'")

            rag_ctx = rag.perform_query(delta_txt, top_k=3)
            rag_ctx = "\n\n".join(rag_ctx)
        else:
            rag_ctx = ""
        rag_context_block = f"RAG-КОНТЕКСТ:\n{rag_ctx}\n\n" if rag_ctx else ""
        source_availability_block = (
            "ПЕРЕДАННЫЕ ИСТОЧНИКИ:\n"
            "• PROFILE: да\n"
            f"• GRAPH: {'да' if graph_dsl_block else 'нет'}\n"
            f"• PSM: {'да' if psm_summary else 'нет'}\n"
            f"• RAG: {'да' if rag_ctx else 'нет'}\n"
            "В expected_effect после 'Приоритет:' используйте только источники со значением 'да'.\n\n"
        )

        predict_number_block = (
            PREDICT_NUMBER_PHRASE if predict_concrete_target else ""
        )

        msgs = self._prompt_whatif.format_messages(
            schema_example=self._schema_example,
            predict_number_block=predict_number_block,
            base_json=json.dumps(base_ctx, ensure_ascii=False, sort_keys=True),
            match_info_block=match_info_block,
            what_if_block=what_if_block,
            task_instructions=task_instructions,
            psm_summary=psm_summary,
            graph_dsl_block=graph_dsl_block,
            features_description=self._features_description,
            rag_context_block=rag_context_block,
            source_availability_block=source_availability_block,
        )

        # Build full prompt text for evaluation purposes
        full_prompt_parts = []
        for msg in msgs:
            if hasattr(msg, 'content'):
                full_prompt_parts.append(msg.content)
            elif isinstance(msg, dict) and 'content' in msg:
                full_prompt_parts.append(msg['content'])
        full_prompt_text = "\n\n".join(full_prompt_parts) if full_prompt_parts else None

        resp = self._invoke_with_fallback(msgs, context_hint="explain_what_if")
        explanation = self._parse_explanation(
            resp,
            full_prompt=full_prompt_text,
            graph_context=graph_dsl_block if use_graph else None,
            rag_context=rag_ctx if use_rag else None,
            psm_summary=psm_summary,
            base_context=base_ctx,
            delta_changes=delta_changes,
        )

        missing_rag_sections = self._missing_rag_sections(explanation, rag_ctx)
        if use_rag and rag_ctx and missing_rag_sections:
            missing = ", ".join(missing_rag_sections)
            logger.warning(
                f"RAG answer missing document references in: {missing}. Retrying once."
            )
            correction = HumanMessage(
                content=(
                    "Исправь предыдущий JSON-ответ и верни только JSON по той же схеме. "
                    f"В секциях {missing} добавь минимум один пункт с опорой на RAG: "
                    "формат 'На основании документа <doc_id>: ... (\"цитата <=20 слов\")'. "
                    "Используй только RAG-КОНТЕКСТ, уже переданный выше; не добавляй новых фактов."
                )
            )
            retry_prompt_text = (
                f"{full_prompt_text}\n\n[RAG correction retry]\n{correction.content}"
                if full_prompt_text
                else correction.content
            )
            retry_resp = self._invoke_with_fallback(
                [*msgs, correction],
                context_hint="explain_what_if_rag_retry",
            )
            explanation = self._parse_explanation(
                retry_resp,
                full_prompt=retry_prompt_text,
                graph_context=graph_dsl_block if use_graph else None,
                rag_context=rag_ctx,
                psm_summary=psm_summary,
                base_context=base_ctx,
                delta_changes=delta_changes,
            )

        return explanation

    # -------------------- утилиты --------------------
    @staticmethod
    def _missing_rag_sections(explanation: Explanation, rag_ctx: str) -> List[str]:
        if not rag_ctx:
            return []

        doc_ids = set(re.findall(r"\[DOC_ID\]\s*([^\s]+)", rag_ctx))

        def has_doc_ref(items: List[str]) -> bool:
            text = " ".join(str(item) for item in items)
            if doc_ids:
                return any(doc_id in text for doc_id in doc_ids)
            return "документ" in text.lower()

        missing: List[str] = []
        if not has_doc_ref(explanation.drivers_pos):
            missing.append("drivers_pos")
        if not has_doc_ref(explanation.drivers_neg):
            missing.append("drivers_neg")
        return missing

    @staticmethod
    def _parse_explanation(
        llm_content: str,
        full_prompt: Optional[str] = None,
        graph_context: Optional[str] = None,
        rag_context: Optional[str] = None,
        psm_summary: Optional[str] = None,
        base_context: Optional[Dict] = None,
        delta_changes: Optional[Dict] = None,
    ) -> Explanation:
        """Extract and normalize Explanation schema from LLM text.

        Args:
            llm_content: Raw LLM response text.
            full_prompt: Optional full formatted prompt (for evaluation).
            graph_context: Optional graph DSL context (for evaluation).
            rag_context: Optional RAG enrichment (for evaluation).
            psm_summary: Optional PSM metrics (for evaluation).
            base_context: Optional client profile context (for evaluation).
            delta_changes: Optional intervention changes (for evaluation).

        Returns:
            Explanation with optional evaluation context attached.
        """
        from sme_causal.core.utils import extract_explanation_from_text

        obj: Dict = extract_explanation_from_text(llm_content)
        drivers_pos = [
            _clean_public_graph_terms(str(item))
            for item in list(obj.get("drivers_pos", []) or [])
        ]
        drivers_neg = [
            _clean_public_graph_terms(str(item))
            for item in list(obj.get("drivers_neg", []) or [])
        ]
        recommendations = [
            _clean_public_graph_terms(str(item))
            for item in list(obj.get("recommendations", []) or [])
        ]
        return Explanation(
            diagnosis=_clean_public_graph_terms(str(obj.get("diagnosis", ""))),
            drivers_pos=drivers_pos,
            drivers_neg=drivers_neg,
            recommendations=recommendations,
            expected_effect=_clean_public_graph_terms(
                str(obj.get("expected_effect", ""))
            ),
            raw_text=str(obj.get("raw_text", llm_content)),
            full_prompt=full_prompt,
            graph_context=graph_context,
            rag_context=rag_context,
            psm_summary=psm_summary,
            base_context=base_context,
            delta_changes=delta_changes,
        )

    def _log_response_meta(self, message, context: str) -> None:
        """Log system_fingerprint and basic response metadata, if available.

        Best-effort across LangChain versions by probing common attributes.
        """
        try:
            meta = getattr(message, "response_metadata", {}) or {}
            extra = getattr(message, "additional_kwargs", {}) or {}
            fp = (
                meta.get("system_fingerprint")
                or extra.get("system_fingerprint")
                or (meta.get("openai_response") or {}).get("system_fingerprint")
            )
            model_name = meta.get("model_name", self.model_name)
            logger.info(
                f"LLM meta [{context}]: model={model_name}, fingerprint={fp}, json_mode={self._json_mode}, seed={self.seed}"
            )
        except Exception:
            # Ignore logging failures
            pass

    # -------------------- внутренние утилиты --------------------

    def filter_graph(self, min_conf: float) -> str:
        """
        Вернуть версию DSL-графа, отфильтрованную по conf >= min_conf.
        Если кэша нет — сначала загрузим полный граф (_load_graph_dsl),
        затем отфильтруем строки (без перезаписи кэша).
        """
        # Нормализуем порог
        try:
            min_conf = float(min_conf)
        except Exception:
            logger.warning(
                f"filter_graph: invalid min_conf={min_conf}, fallback to default"
            )
            min_conf = float(self.confidence_threshold)

        # Убедимся, что базовый кэш заполнен
        if self._graph_dsl_cache is None:
            _ = self._load_graph_dsl()  # создаст кэш или пустую строку

        dsl = self._graph_dsl_cache or ""
        if not dsl.strip():
            return ""

        lines = dsl.splitlines()
        header = (
            lines[0]
            if lines and lines[0].startswith("# Causal Graph")
            else "# Causal Graph (DSL) v1"
        )

        kept = []
        for line in lines[1:]:
            # ищем токен вида "conf:0.78"
            conf_val = None
            for part in line.split("|"):
                p = part.strip()
                if p.lower().startswith("conf:"):
                    try:
                        conf_val = float(p.split(":", 1)[1].strip())
                    except Exception:
                        conf_val = None
                    break

            if conf_val is not None and conf_val >= min_conf:
                kept.append(line)

        if not kept:
            return ""

        return "\n".join([header, *kept])

    def _load_graph_dsl(self, min_conf: float = None) -> str:
        """Читает JSON-граф (cfg.llm_edges_path) и возвращает DSL-строки вида:
        'A -> B | sign:+ | conf:0.78 | note:"..."'
        Если файла нет — возвращает пустую строку.
        """
        min_conf = min_conf if min_conf is not None else self.confidence_threshold

        # Кэш
        if self._graph_dsl_cache is not None:
            if min_conf == self.confidence_threshold:
                return self._graph_dsl_cache
            else:
                return self.filter_graph(min_conf)

        # Нет пути — ничего не подмешиваем
        if not self.graph_path:
            self._graph_dsl_cache = ""
            return self._graph_dsl_cache

        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # Поддержка двух форматов: список рёбер ИЛИ {"edges": [...]}
            edges = raw.get("edges") if isinstance(raw, dict) else raw
            if not isinstance(edges, list):
                logger.warning("LLM graph JSON: expected list or dict with 'edges'.")
                self._graph_dsl_cache = ""
                return self._graph_dsl_cache

            lines: List[str] = ["# Causal Graph (DSL) v1"]
            for e in edges:
                s = e.get("source")
                t = e.get("target")
                if not isinstance(s, str) or not isinstance(t, str):
                    continue
                s = s.strip()
                t = t.strip()
                if not s or not t or s == t:
                    # минимальная защита от битых/самозацикленных рёбер
                    continue

                sign = e.get("sign") or e.get("polarity") or "?"
                conf = e.get("confidence", e.get("robustness_score", None))
                note = e.get("interpretation") or e.get("rationale") or ""
                if not note and e.get("robustness_label"):
                    note = (
                        f"Algorithmic consensus: {e.get('robustness_label')}; "
                        f"support_ge_tau={e.get('support_ge_tau', 'n/a')}; "
                        f"mean_freq={e.get('mean_freq', 'n/a')}"
                    )

                parts = [f"{s} -> {t}", f"sign:{sign}"]

                # conf печатаем, только если это число
                try:
                    if conf is not None:
                        parts.append(f"conf:{float(conf):.2f}")
                except Exception:
                    pass

                if note:
                    safe_note = str(note).replace('"', '\\"')
                    parts.append(f'note:"{safe_note}"')

                lines.append(" | ".join(parts))

            self._graph_dsl_cache = "\n".join(lines)
            logger.info(
                f"Loaded causal graph DSL: {len(lines) - 1} edges from {self.graph_path}"
            )
            return self._graph_dsl_cache

        except FileNotFoundError:
            logger.warning(f"No graph file at {self.graph_path}")
        except Exception as ex:
            logger.warning(f"Failed to load graph DSL: {ex}")

        self._graph_dsl_cache = ""
        return self._graph_dsl_cache

    def _invoke_with_fallback(self, msgs, context_hint: str) -> str:
        """Invoke current LLM, falling back to non-JSON mode on failure.

        Also logs response metadata when available.
        """
        content, raw_msg, used_json = invoke_with_fallback(
            msgs,
            model=self.model_name,
            temperature=float(self.temperature),
            api_key=self.api_key,
            seed=self.seed,
            top_p=self.top_p,
        )
        self._json_mode = used_json
        self._log_response_meta(
            raw_msg, context=context_hint if used_json else f"{context_hint}:fallback"
        )
        return content


# Датакласс для структурированного ответа парсера
@dataclass
class ParsedQuery:
    action_type: str
    delta: Dict[str, Any]
    label: str
    info_text: str
    match_info: Optional[Dict[str, Any]] = None
    target_metric: Optional[str] = "Revenue_Growth_Rate"


class QueryParser:
    """
    Преобразует открытый текстовый запрос пользователя в структурированную команду для CausalAgent.
    Умеет находить похожие на запрос колонки и сообщать об этом.
    """

    def __init__(
        self, model: str = None, temperature: float = 0.2, api_key: Optional[str] = None
    ):
        cfg = get_config()
        self.model_name = model or cfg.llm.model_name
        self.temperature = temperature
        self.api_key = api_key or cfg.effective_openai_api_key

        # Список всех доступных колонок для помощи модели в маппинге, а также их диапазоны
        self.available_columns = str(INTERVENTIONS_RANGES)

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Вы — экспертная система, которая анализирует запросы на естественном языке и преобразует их в структурированный JSON-формат для дальнейшей обработки. "
                    "Вы работаете в крупном и авторитетном банковской сфере, ваши клиенты входят в сферу малого и микро бизнеса."
                    "Ваш пользователь - опытный аналитик банка, который хочет исследовать, как определенные вмешательства (интервенции) повлияют на клиентов банка."
                    "Ваша задача — точно определить намерение пользователя, извлечь параметры и сопоставить их с известными полями системы.\n\n"
                    f"Описания доступных полей и метрик клиента: {FEATURES_DESCRIPTION}."
                    "ТИПЫ ЗАПРОСОВ:\n"
                    "1. 'what_if': Пользователь хочет проверить гипотезу, т.е. оценить эффект от одной или нескольких конкретных интервенций. Например: 'Что будет, если предложить клиенту ...?'\n"
                    "2. 'optimize': Пользователь ищет наилучшую интервенцию для достижения цели. Например: 'Какую интервенцию лучше применить для роста дохода?'\n\n"
                    "ПРАВИЛА ПРЕОБРАЗОВАНИЯ:\n"
                    "- Определите тип желаемого действия `action_type` ('what_if' или 'optimize').\n"
                    "- Для 'what_if', сформируйте словарь `delta` с изменениями. Ключами `delta` должны быть ПОЛЯ из списка ДОСТУПНЫХ ПОЛЕЙ.\n"
                    "- Важное правило поиска: если в запросе спрашивается про одну интервенцию, нужно найти одно поле, если в запросе спрашивается про несколько интервенций - нужно найти столько же полей, сколько в запросе есть желаемых интервенций.\n"
                    "  - Сначала попробуйте найти уверенное смысловое совпадение (ты сильно уверен в нем) для каждой интервенции из запроса. Например, для 'предложить скидку на тариф' уверенное поле - `Tariff_Discount`.\n"
                    "  - Если близкого совпадения нет, найдите возможное совпадение (ты средне уверен в нем) для каждой интервенции из запроса. Например, для запроса 'снизить комиссию' возможное похожее поле - `Tariff_Discount`.\n"
                    "  - **Если не удалось найти даже возможное по смыслу поле (ты слабо уверен в нем), оставьте `delta` и `match_info` пустыми (`{{}}`). Не выдумывайте несуществующие в списке ДОСТУПНЫХ ПОЛЕЙ интервенции!**\n"
                    "  - Интервенции должны принимать ТОЛЬКО ЗНАЧЕНИЯ ИЗ ДИАПАЗОНОВ, указанных в списке ДОСТУПНЫХ ПОЛЕЙ. Если интервенция из запроса не находится в этом диапазоне и не похожа на возможные значения диапазона, не добавляй ее в итоговый ответ. "
                    "- `match_info`: Этот словарь ОБЯЗАТЕЛЕН для `what_if` с непустым `delta`. Для каждой пары ключ-значение в `delta` добавьте информацию о совпадении:\n"
                    "  - `status`: 'confident' (уверенное совпадение) или 'similar' (похожее).\n"
                    "  - `query_phrase`: фрагмент текста из запроса, который соответствует этому полю.\n"
                    "- `target_metric`: Если пользователь в запросе явно указывает, какую метрику он хочет улучшить (например, 'увеличить средний баланс', 'для роста оборотов'), извлеките название этой метрики, используя описания доступных полей и метрик. Если целевая метрика не указана, оставьте это поле `null`.\n"
                    "- `label` и `info_text`: Краткое и детальное описание интервенции.\n"
                    "- Если запрос типа 'optimize', поля `delta`, `label`, `info_text`, `match_info` должны быть пустыми.\n\n"
                    "ДОСТУПНЫЕ ПОЛЯ ДЛЯ `delta` И ДИАПАЗОНЫ ИХ ВОЗМОЖНЫХ ЗНАЧЕНИЙ:\n{available_columns}\n\n"
                    "ИНТЕРВЕНЦИИ В ТВОЕМ ОТВЕТЕ МОГУТ БЫТЬ ТОЛЬКО В ДИАПАЗОНЕ ВОЗМОЖНЫХ ЗНАЧЕНИЙ: если в диапазоне передан список - строго быть одним из значений в списке, если передана строка - строго соответствовать требованиям из строки.\n"
                    "Если в диапазоне возможных значений интервенций есть похожее значение на желаемую интервенцию из запроса - работай с ним по алгоритму выше. Если похожих нет, игнорируй эту запрошенную интервенцию из запроса и не выводи ее в ответ."
                    "СТРОГО СЛЕДУЙТЕ ФОРМАТУ ВЫВОДА JSON. ПОЛЕ `match_info` ОБЯЗАТЕЛЬНО, ЕСЛИ `delta` НЕ ПУСТОЙ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "(what_if или optimize)",\n'
                    '  "delta": {{...}},\n'
                    '  "label": "(краткое название)",\n'
                    '  "info_text": "(описание)",\n'
                    '  "target_metric": "(название метрики или null)",\n'
                    '  "match_info": {{ "Имя_колонки_из_delta": {{ "status": "confident|similar", "query_phrase": "фраза из запроса" }} }}\n'
                    "}}\n"
                    "```\n\n"
                    "НЕСКОЛЬКО ПРИМЕРОВ ДЛЯ ОБУЧЕНИЯ (FEW-SHOT EXAMPLES):\n\n"
                    "ПРИМЕР 1: Запрос с несколькими интервенциями (уверенное совпадение)\n"
                    "ЗАПРОС: 'что если мы предложим клиенту зарплатный проект и эквайринг?'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "what_if",\n'
                    '  "delta": {{\n'
                    '    "New_Product_Offer": 1,\n'
                    '    "New_Product_Offer_Type": "payroll, acquiring"\n'
                    "  }},\n"
                    '  "label": "Предложение зарплатного проекта и эквайринга",\n'
                    '  "info_text": "Offer a package with payroll project and acquiring services.",\n'
                    '  "target_metric": null,\n'
                    '  "match_info": {{\n'
                    '    "New_Product_Offer": {{ "status": "confident", "query_phrase": "предложим клиенту" }},\n'
                    '    "New_Product_Offer_Type": {{ "status": "confident", "query_phrase": "зарплатный проект и эквайринг" }}\n'
                    "  }}\n"
                    "}}\n"
                    "```\n\n"
                    "ПРИМЕР 2: Запрос на изменение лимита (похожее совпадение)\n"
                    "ЗАПРОС: 'как повлияет на клиента увеличение кредитной линии на 25%?'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "what_if",\n'
                    '  "delta": {{\n'
                    '    "Credit_Limit_Change": 25.0\n'
                    "  }},\n"
                    '  "label": "Увеличение кредитной линии на 25%",\n'
                    '  "info_text": "Increase the credit line for the client by 25%.",\n'
                    '  "target_metric": null,\n'
                    '  "match_info": {{\n'
                    '    "Credit_Limit_Change": {{ "status": "similar", "query_phrase": "увеличение кредитной линии на 25%" }}\n'
                    "  }}\n"
                    "}}\n"
                    "```\n\n"
                    "ПРИМЕР 3: Нечеткий/неизвестный запрос (нет совпадений)\n"
                    "ЗАПРОС: 'оценить эффект от подключения услуги бизнес-ассистент'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "what_if",\n'
                    '  "delta": {{}},\n'
                    '  "label": "Анализ услуги бизнес-ассистент,\n'
                    '  "info_text": "Analyze the effect of connecting the business assistant service",\n'
                    '  "target_metric": null,\n'
                    '  "match_info": {{}}\n'
                    "}}\n"
                    "```\n\n"
                    "ПРИМЕР 4: Запрос на поиск оптимального решения ('optimize')\n"
                    "ЗАПРОС: 'подбери оптимальное предложение для роста оборотов клиента'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "optimize",\n'
                    '  "delta": {{}},\n'
                    '  "label": "",\n'
                    '  "info_text": "",\n'
                    '  "target_metric": "Avg_Monthly_Inflow",\n'
                    '  "match_info": {{}}\n'
                    "}}\n"
                    "```\n\n"
                    "ПРИМЕР 5: Запрос с одной интервенцией (уверенное совпадение)\n"
                    "ЗАПРОС: 'открыть эквайринг'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "what_if",\n'
                    '  "delta": {{\n'
                    '    "New_Product_Offer": 1,\n'
                    '    "New_Product_Offer_Type": "acquiring"\n'
                    "  }},\n"
                    '  "label": "Предложение эквайринга",\n'
                    '  "info_text": "Offer acquiring.",\n'
                    '  "target_metric": null,\n'
                    '  "match_info": {{\n'
                    '    "New_Product_Offer": {{ "status": "confident", "query_phrase": "открыть" }},\n'
                    '    "New_Product_Offer_Type": {{ "status": "confident", "query_phrase": "эквайринг" }}\n'
                    "  }}\n"
                    "}}\n"
                    "```"
                    "ПРИМЕР 6: Запрос с целевой метрикой\n"
                    "ЗАПРОС: 'что если дать скидку на РКО для увеличения среднего баланса на счете?'\n"
                    "ОТВЕТ:\n"
                    "```json\n"
                    "{{\n"
                    '  "action_type": "what_if",\n'
                    '  "delta": {{\n'
                    '    "Tariff_Discount": 1\n'
                    "  }},\n"
                    '  "label": "Скидка на РКО для роста баланса",\n'
                    '  "info_text": "Apply a discount on cash and settlement services",\n'
                    '  "target_metric": "Avg_Account_Balance",\n'
                    '  "match_info": {{\n'
                    '    "Tariff_Discount": {{ "status": "confident", "query_phrase": "дать скидку на РКО" }}\n'
                    "  }}\n"
                    "}}\n"
                    "```\n\n",
                ),
                ("user", "Проанализируй запрос: '{query_text}'"),
            ]
        )

    def parse(self, query_text: str) -> Optional[ParsedQuery]:
        """
        Выполняет парсинг запроса.

        Args:
            query_text: Текст запроса от пользователя.

        Returns:
            Объект ParsedQuery со структурированными данными или None в случае ошибки.
        """
        logger.info(f"Parsing open query: '{query_text}'")

        msgs = self._prompt_template.format_messages(
            query_text=query_text, available_columns=self.available_columns
        )

        content, _, _ = invoke_with_fallback(
            msgs,
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            seed=42,  # Используем seed для более стабильного парсинга
        )

        try:
            parsed_obj = parse_json_obj_from_text(content)

            if "action_type" not in parsed_obj:
                raise ValueError("Missing 'action_type' in parsed object")

            logger.info(f"Successfully parsed query: {parsed_obj}")
            return ParsedQuery(
                action_type=parsed_obj.get("action_type"),
                delta=parsed_obj.get("delta", {}),
                label=parsed_obj.get("label", ""),
                info_text=parsed_obj.get("info_text", ""),
                match_info=parsed_obj.get("match_info"),
                target_metric=parsed_obj.get("target_metric"),
            )

        except Exception as e:
            logger.error(
                f"Error parsing LLM response for query '{query_text}'. Error: {e}\nResponse: {content}"
            )
            return None
