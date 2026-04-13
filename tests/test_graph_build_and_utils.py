import networkx as nx
import json


def test_edges_to_digraph_and_graph_to_dict_roundtrip():
    from sme_causal.graph.graph_utils import graph_to_dict, edges_to_digraph

    edges = [
        {"source": "A", "target": "B", "confidence": 0.7},  # defaults applied
        {
            "source": "B",
            "target": "C",
            "relation": "causal",
            "polarity": "-",
            "confidence": 0.9,
            "rationale": "test",
        },
    ]
    G = edges_to_digraph(edges)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_edges() == 2
    # Check attributes and defaults
    d = G["A"]["B"]
    assert d["relation"] == "causal"
    assert d["polarity"] == "+"
    # Confidence provided in input should be preserved
    assert isinstance(d["confidence"], float) and d["confidence"] == 0.7
    assert d["rationale"] == ""

    dto = graph_to_dict(G)
    assert set(dto.keys()) == {"nodes", "edges"}
    assert set(dto["nodes"]) == {"A", "B", "C"}
    # Edge entries include attributes
    rec = next(e for e in dto["edges"] if e["source"] == "B" and e["target"] == "C")
    assert rec["polarity"] == "-" and rec["confidence"] == 0.9


def test_create_edge_comparison_table_columns():
    from sme_causal.graph.evaluate_graphs import create_edge_comparison_table

    gt = [
        {"source": "A", "target": "B", "sign": "+"},
        {"source": "X", "target": "Y", "sign": "-"},
    ]
    preds = [
        {"source": "A", "target": "B", "polarity": "+", "confidence": 0.8},
        {"source": "B", "target": "C", "polarity": "-", "confidence": 0.2},
    ]
    df = create_edge_comparison_table(preds, gt)

    assert list(df.columns) == [
        "From",
        "To",
        "Sign",
        "In_Ground_Truth",
        "Found_In_Graph",
    ]
    assert len(df) == 3
    gt_row = df[(df["From"] == "A") & (df["To"] == "B")].iloc[0]
    assert gt_row["In_Ground_Truth"] == "Yes"
    assert gt_row["Found_In_Graph"] == "Yes"
    fp_row = df[(df["From"] == "B") & (df["To"] == "C")].iloc[0]
    assert fp_row["Sign"] == "-"
    assert fp_row["In_Ground_Truth"] == "No"
    assert fp_row["Found_In_Graph"] == "Yes"

def test_evaluate_graph_metrics():
    from sme_causal.graph.evaluate_graphs import evaluate_graph

    tiny_gt = [
        {"source": "A", "target": "B", "sign": "+"},
        {"source": "B", "target": "C", "sign": "-"},
    ]
    preds = [
        {"source": "A", "target": "B", "polarity": "+", "confidence": 0.9},  # TP signed
        {"source": "B", "target": "C", "polarity": "+", "confidence": 0.6},  # wrong sign
        {"source": "A", "target": "C", "polarity": "+", "confidence": 0.2},  # FP
    ]

    metrics = evaluate_graph(preds, tiny_gt)

    for k in [
        "precision",
        "recall",
        "f1",
        "cpf",
        "tp",
        "fp",
        "fn",
        "rp",
        "total_gt_paths",
        "reproduced_paths",
    ]:
        assert k in metrics

    assert metrics["tp"] == 1
    assert metrics["fp"] == 2
    assert metrics["fn"] == 1
    assert metrics["rp"] == 0
    assert abs(metrics["precision"] - (1 / 3)) < 1e-9
    assert abs(metrics["recall"] - 0.5) < 1e-9
    assert abs(metrics["f1"] - 0.4) < 1e-9
    assert metrics["total_gt_paths"] == 1
    assert metrics["reproduced_paths"] == 1


def test_explanation_to_dict_excludes_debug_by_default():
    from sme_causal.agent.agent_service import Explanation

    expl = Explanation(
        diagnosis="ok",
        drivers_pos=["A"],
        drivers_neg=["B"],
        recommendations=["C"],
        expected_effect="D",
        raw_text="raw",
        full_prompt="prompt",
        graph_context="graph",
        rag_context="rag",
    )

    public_payload = expl.to_dict()
    assert public_payload == {
        "diagnosis": "ok",
        "drivers_pos": ["A"],
        "drivers_neg": ["B"],
        "recommendations": ["C"],
        "expected_effect": "D",
    }

    debug_payload = expl.to_dict(include_debug=True)
    assert debug_payload["raw_text"] == "raw"
    assert debug_payload["full_prompt"] == "prompt"
    assert debug_payload["graph_context"] == "graph"
    assert debug_payload["rag_context"] == "rag"


def test_explanation_parser_hides_internal_graph_tokens():
    from sme_causal.agent.agent_service import CausalAgent

    payload = {
        "drivers_pos": [
            "На основании ребра Avg_Monthly_Inflow -> Revenue_Growth_Rate (sign:+, conf=0.82): рост выручки."
        ],
        "drivers_neg": [
            "На основании ребра Avg_Monthly_Outflow -> Net_Cashflow (sign:-, conf=0.70): давление на поток."
        ],
        "recommendations": [],
        "expected_effect": (
            "На основании ребра Has_Payroll -> Avg_Monthly_Inflow "
            "(sign:+, conf=0.79): ожидаем рост. "
            "Avg_Monthly_Inflow -> Revenue_Growth_Rate (conf 0.76)."
        ),
    }

    expl = CausalAgent._parse_explanation(json.dumps(payload, ensure_ascii=False))

    public_text = "\n".join(
        [*expl.drivers_pos, *expl.drivers_neg, expl.expected_effect]
    )
    assert "sign" not in public_text
    assert "conf" not in public_text
    assert "Avg_Monthly_Inflow" in public_text
    assert "Revenue_Growth_Rate" in public_text
