import json
from pathlib import Path

import pandas as pd


def test_summarize_numeric_basic():
    from sme_causal.core.utils import summarize_numeric

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10.0, 12.0, 14.0, 16.0],
            "cat": ["x", "y", "x", "z"],
        }
    )
    out = summarize_numeric(df)
    # Should include only numeric columns with expected summary headers
    assert list(out.columns) == ["mean", "std", "min", "q50", "max"]
    assert set(out.index.tolist()) == {"a", "b"}
    # Spot-check rounding
    assert out.loc["a", "mean"] == 2.5
    assert out.loc["b", "min"] == 10.0


def test_summarize_numeric_no_numeric():
    from sme_causal.core.utils import summarize_numeric

    df = pd.DataFrame({"cat": ["x", "y"]})
    out = summarize_numeric(df)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_parse_json_obj_from_text_variants():
    from sme_causal.core.utils import parse_json_obj_from_text

    obj = {"a": 1, "b": [1, 2]}
    # Direct JSON
    assert parse_json_obj_from_text(json.dumps(obj)) == obj
    # JSON embedded in text
    txt = f"noise... {json.dumps(obj)} ...more noise"
    assert parse_json_obj_from_text(txt) == obj
    # Invalid JSON
    assert parse_json_obj_from_text("not json at all") == {}


def test_extract_edges_from_text_validation_and_coercion():
    from sme_causal.core.utils import extract_edges_from_text

    payload = {
        "edges": [
            {"source": "A", "target": "B", "confidence": "0.9", "extra": 1},  # valid, confidence -> float, drop extra
            "not a dict",  # ignore
            {"source": "C"},  # missing target -> drop
            {"source": "D", "target": "E", "confidence": "oops"},  # drop bad confidence
        ]
    }
    res = extract_edges_from_text(json.dumps(payload))
    assert isinstance(res, list)
    assert len(res) == 2
    e1 = res[0]
    assert e1["source"] == "A" and e1["target"] == "B"
    assert e1.get("confidence") == 0.9
    assert "extra" not in e1
    e2 = res[1]
    assert e2["source"] == "D" and e2["target"] == "E"
    assert "confidence" not in e2


def test_extract_explanation_from_text_normalization():
    from sme_causal.core.utils import extract_explanation_from_text

    payload = {
        "diagnosis": "  Growth driver  ",
        "drivers_pos": "inflow",
        "drivers_neg": [" cost  ", 123],
        "recommendations": "increase limit",
        "expected_effect": 0.12,
        "irrelevant": "ignored",
    }
    text = json.dumps(payload)
    out = extract_explanation_from_text(text)
    assert out["diagnosis"] == "Growth driver"
    assert out["drivers_pos"] == ["inflow"]
    assert out["drivers_neg"] == ["cost", "123"]
    assert out["recommendations"] == ["increase limit"]
    assert out["expected_effect"] == "0.12"
    assert out["raw_text"] == text


def test_configure_logging_creates_file(tmp_path: Path):
    from loguru import logger
    from sme_causal.core.utils import configure_logging
    from sme_causal.core.config import LoggingConfig

    log_file = tmp_path / "test.log"
    cfg = LoggingConfig()
    configure_logging(log_file, cfg, add_stdout=False)
    # Emit a test log line
    logger.info("hello")
    # File should be created and non-empty
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") != ""

