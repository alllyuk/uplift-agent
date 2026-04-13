from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re

import numpy as np
import pandas as pd
from loguru import logger

from sme_causal.core.config import LoggingConfig


def configure_logging(
    log_path: Path,
    logging_cfg: LoggingConfig,
    add_stdout: bool = False,
    stdout_format: str | None = None,
) -> None:
    """Configure loguru logger consistently across entry points.

    Args:
        log_path: Path to the file log.
        logging_cfg: Logging configuration (rotation/retention/level/format).
        add_stdout: Whether to also log to stdout.
        stdout_format: Optional, override format for stdout sink.
    """
    # Remove any prior sinks to avoid duplicates
    logger.remove()
    # File sink
    logger.add(
        log_path,
        rotation=logging_cfg.rotation,
        retention=logging_cfg.retention,
        level=logging_cfg.level,
        format=logging_cfg.format,
    )
    # Optional stdout sink
    if add_stdout:
        logger.add(
            sys.stdout,
            level=logging_cfg.level,
            format=stdout_format or logging_cfg.format,
        )


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Generate statistical summary of numeric columns.

    Returns a DataFrame with columns: mean, std, min, q50, max.
    """
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc = desc[["mean", "std", "min", "50%", "max"]]
    desc = desc.rename(columns={"50%": "q50"})
    return desc.round(3)


def parse_json_obj_from_text(text: str) -> Dict[str, Any]:
    """Best-effort parse a JSON object from raw LLM text.

    Tries direct json.loads, otherwise extracts the first {...} block.
    Returns an empty dict if parsing fails.
    """
    import json

    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            frag = text[start : end + 1]
            try:
                return json.loads(frag)
            except Exception:
                return {}
        return {}


def extract_edges_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse and lightly validate an edge list from LLM text.

    Expects a JSON object with key "edges": [ {source, target, ...} ].
    Filters to known fields and drops entries without source/target.
    """
    obj = parse_json_obj_from_text(text) or {}
    raw = obj.get("edges", []) if isinstance(obj, dict) else []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    allowed = {"source", "target", "relation", "polarity", "confidence", "rationale"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        src, dst = item.get("source"), item.get("target")
        if not (isinstance(src, str) and isinstance(dst, str)):
            continue
        e: Dict[str, Any] = {k: item.get(k) for k in allowed if k in item}
        # type coercions
        if "confidence" in e:
            try:
                e["confidence"] = float(e["confidence"])  # type: ignore[assignment]
            except Exception:
                e.pop("confidence", None)
        out.append(
            {
                "source": src,
                "target": dst,
                **{k: v for k, v in e.items() if k not in {"source", "target"}},
            }
        )
    return out


def extract_explanation_from_text(text: str) -> Dict[str, Any]:
    """Parse and normalize an Explanation-like dict from LLM text.

    Keeps only expected keys and coerces lists to list[str].
    """
    obj = parse_json_obj_from_text(text) or {}
    if not isinstance(obj, dict):
        return {}
    keep = {
        "diagnosis": "",
        "drivers_pos": [],
        "drivers_neg": [],
        "recommendations": [],
        "expected_effect": "",
        "raw_text": text,
    }
    res: Dict[str, Any] = {k: obj.get(k, v) for k, v in keep.items() if k != "raw_text"}
    # Coerce list fields to list[str]
    for key in ("drivers_pos", "drivers_neg", "recommendations"):
        vals = res.get(key, [])
        if not isinstance(vals, list):
            vals = [vals]
        res[key] = [str(x).strip() for x in vals if str(x).strip()]
    res["diagnosis"] = str(res.get("diagnosis", "")).strip()
    res["expected_effect"] = str(res.get("expected_effect", "")).strip()
    res["raw_text"] = text
    return res


def sanity_checks(
    ctx: Dict[str, object], delta: Dict[str, object]
) -> Dict[str, object]:
    """
    Возвращает:
    {
        "blocked": bool,               # если True — аплифт = 0, и LLM/PSM можно не вызывать
        "reasons": [str, ...],         # объяснения
        "notes": Dict[str, object],    # дополнительные факты для объяснения
    }
    """
    reasons = []
    notes = {}

    # New Product Offer
    new_offer_flag = int(delta.get("New_Product_Offer", 0))

    # Product offer types:
    has_acq = int(ctx.get("Has_Acquiring", ctx.get("has_acquiring", 0)) or 0)
    has_loan = int(ctx.get("Has_Loan", ctx.get("has_loan", 0)) or 0)
    has_payroll = int(ctx.get("Has_Payroll", ctx.get("has_payroll", 0)) or 0)
    has_card = int(ctx.get("Has_Card", ctx.get("has_card", 0)) or 0)
    has_deposit = int(ctx.get("Has_Deposit", ctx.get("has_deposit", 0)) or 0)

    # Credit Limit Change
    curr_rate = ctx.get("Credit_Limit_Change", 0)

    # Tariff Discount
    curr_discount = ctx.get("Tariff_Discount", 0)

    # 1) Предлагаем продукт клиенту, у которого уже есть
    if str(delta.get("New_Product_Offer_Type")).lower() in {"acquiring"}:
        if new_offer_flag == 1 and has_acq == 1:
            product = "acquiring"
            reasons.append(
                f"У клиента уже есть этот продукт ({product}) — повторное предложение не создаёт ценности."
            )
            notes["has_acquiring"] = True

    if str(delta.get("New_Product_Offer_Type")).lower() in {"loan"}:
        if new_offer_flag == 1 and has_loan == 1:
            product = "loan"
            reasons.append(
                f"У клиента уже есть этот продукт ({product}) — повторное предложение не создаёт ценности."
            )
            notes["has_loan"] = True

    if str(delta.get("New_Product_Offer_Type")).lower() in {"payroll"}:
        if new_offer_flag == 1 and has_payroll == 1:
            product = "payroll"
            reasons.append(
                f"У клиента уже есть этот продукт ({product}) — повторное предложение не создаёт ценности."
            )
            notes["has_payroll"] = True

    if str(delta.get("New_Product_Offer_Type")).lower() in {"deposit"}:
        if new_offer_flag == 1 and has_deposit == 1:
            product = "deposit"
            reasons.append(
                f"У клиента уже есть этот продукт ({product}) — повторное предложение не создаёт ценности."
            )
            notes["has_deposit"] = True

    if str(delta.get("New_Product_Offer_Type")).lower() in {"card"}:
        if new_offer_flag == 1 and has_card == 1:
            product = "card"
            reasons.append(
                f"У клиента уже есть этот продукт ({product}) — повторное предложение не создаёт ценности."
            )
            notes["has_card"] = True

    # 2) Предложенная ставка не лучше текущей
    proposed_rate = None
    k = "Credit_Limit_Change"
    if k in delta:
        proposed_rate = delta[k]

    if proposed_rate is not None and curr_rate is not None:
        try:
            pr = float(proposed_rate)
            cr = float(curr_rate)
            # Если предлагаемая ставка >= текущей — аплифт = 0
            if cr >= pr:
                reasons.append(
                    f"Предложенная ставка {pr:.2f}% не лучше текущей {cr:.2f}%."
                )
                notes["rate_no_improvement"] = {"proposed": pr, "current": cr}
        except Exception:
            pass

    # 2) У клиента и так льготные условия
    proposed_discount = None
    d = "Tariff_Discount"
    if d in delta:
        proposed_discount = delta[d]

    if proposed_rate is not None and curr_discount is not None:
        try:
            pd = int(proposed_discount)
            cd = int(curr_discount)

            if cd >= pd:
                reasons.append(f"Уже предложены льготные условия.")
                notes["discount_no_improvement"] = {"proposed": pr, "current": cr}
        except Exception:
            pass

    return {
        "blocked": len(reasons) > 0,
        "reasons": reasons,
        "notes": notes,
    }


def create_query(delta):
    translate = {
        "new_product_offer": "предложение нового продукта",
        "new_product_offer_type": "тип нового предложения продукта",
        "credit_limit_change": "изменение кредитного лимита",
        "tariff_discount": "применение льготного тарифа",
        "acquiring": "эквайринг",
        "payroll": "зарплатный проект",
    }
    interventions = []
    try:
        for k, v in delta.items():
            key_lower = k.lower() if isinstance(k, str) else k
            val_lower = v.lower() if isinstance(v, str) else v
            translated_key = translate.get(key_lower, k)
            translated_value = translate.get(val_lower, v)
            temp1 = ""
            if key_lower == "new_product_offer":
                temp1 = f"{translated_key}"
            if key_lower == "new_product_offer_type":
                if temp1 == "":
                    temp1 += f"{translate['new_product_offer']} {translated_value}"
                else:
                    temp1 += f"{translated_value}"
                temp = temp1
                interventions.append(temp)
            if key_lower == "credit_limit_change":
                temp = f"{translated_key} на {translated_value}"
                interventions.append(temp)
            if key_lower == "tariff_discount":
                temp = f"{translated_key}"
                interventions.append(temp)
            query_tmp = ", ".join(interventions)
            query = f"Как {query_tmp} повлияет на клиента и его активность?"
    except Exception as e:
        delta_str = ", ".join([f"{k}={v}" for k, v in delta.items()])
        query = f"Как {delta_str} повлияет на клиента и его активность?"
    return query


def parse_client_id_and_intent(query_text: str) -> Tuple[Optional[str], str]:
    """
    Извлекает ID клиента (формат CXXXXXX) из текста запроса с помощью регулярного выражения.

    Args:
        query_text: Исходный текст запроса.

    Returns:
        Кортеж из двух элементов:
        - Найденный client_id или None, если ID не найден.
        - Текст запроса (intent) без ID клиента, с удаленными лишними пробелами.
    """
    # Паттерн для поиска "C" и 6 цифр как отдельного слова
    pattern = r"\b(C\d{6})\b"

    match = re.search(pattern, query_text)

    if match:
        client_id = match.group(1)
        cleaned_text = re.sub(pattern, "", query_text, count=1).strip()
        # Убираем двойные пробелы, которые могли остаться после удаления
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        logger.info(
            f"Regex found client_id: {client_id}. Cleaned query: '{cleaned_text}'"
        )
        return client_id, cleaned_text
    else:
        return None, query_text
