## SME Causal Graph Inference (LLM + Synthetic Data)

End‑to‑end pipeline to:

- Generate synthetic SME client data with a known causal DAG.
- Infer causal edges from tabular features using an LLM (LangChain + OpenAI) or algorithm approach.
- Build and export a directed graph (JSON/GEXF/GraphML) and an interactive HTML view.
- Validate inferred edges against ground truth from the synthetic generator.
- Explore everything in an interactive Streamlit UI, including per‑client explanations and simple what‑if scenarios.


**Key Modules**

- `sme_causal/app/main.py`: Orchestrates full pipeline (data → graph building → LLM inference → export → evaluation).
- `sme_causal/app/streamlit_app.py`: Interactive UI to run the pipeline, visualize graphs, and get explanations.
- `sme_causal/app/build_and_visualize_graph.py`: CLI example to build/export/visualize without Streamlit.
- `sme_causal/app/run.py`: CLI example to use full pipeline without Streamlit.
- `sme_causal/app/build_rag.py`: Build full RAG pipeline with text corpus from cfg (rag_data/document_corpus for example), chunks, embedds and indeces FAISS.
- `sme_causal/data/synth_data.py`: Synthetic data generator and ground‑truth causal edges.
- `sme_causal/inference/llm_graph.py`: LLM‑based inference of causal edges with layer constraints.
- `sme_causal/graph/build_llm_graph.py`: Build causal graph based on LLM request.
- `sme_causal/graph/build_algo_graph.py`: Build causal graph based on algorithmic approaches (consensus of GES, HC, MMHC)
- `sme_causal/graph/build_algo_llm_graph.py`: Build causal graph based on algorithmic approaches with LLM edges validation.
- `sme_causal/graph/build_hybrid_graph.py`: Build causal graph based on LLM and Algo graphs with LLM as a judge for combination.
- `sme_causal/graph/graph_utils.py`: Helper functions for graph building and using.
- `sme_causal/graph/evaluate_graph.py`: Evaluate calculated graph against Ground Truth, possible to use bootstrap to measure confidence.
- `sme_causal/graph/graph_viz.py`: Reusable PyVis HTML graph rendering helpers.
- `sme_causal/agent/agent_service.py`: LLM agent for per‑client explanations and what‑if.
- `sme_causal/core/config.py`: Central configuration (Pydantic Settings) + env var overrides.



## Quickstart

1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Configure environment (auto‑loaded by `env.py` via `python-dotenv`)

Create `.env` in the project root (auto‑loaded) with at least:

```env

VLLM_DISABLE_CUDA_GRAPH=1
TORCH_CUDA_ARCH_LIST=8.0


#For OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=your-actual-api-key
LLM_MODEL=your-actual-llm-model
```

Optional: override LLM/data/paths/logging via env vars; see defaults in `sme_causal/core/config.py`.

3. Run the full pipeline (CLI).
Specify argument `--graph-method=` from `["llm", "algo", "algo_llm", "hybrid"]` if need (default `llm`)

```bash
python -m sme_causal.app.main
```

Artifacts land under `artifacts/` for LLM-based and Hydrid graph, and also under `causal_outputs/` for algorithm-based graphs.

**Pipeline flow:** `intake → context → policy_check → estimation (PSM + RAG + Graph in parallel) → synthesize → critic (L1 rules + L2 LLM) → [retry if needed] → persist to SQLite`

All three evidence sources (PSM, Graph, RAG) are **enabled by default**. Disable with `--no-psm`, `--no-graph`, `--no-rag`. If all requested sources fail, the pipeline aborts with `no_evidence`. Cases are persisted to `artifacts/cases.db`.

**Cooldown:** pipeline запоминает выполненные кейсы в SQLite. Если для того же клиента и того же типа интервенции уже есть завершённый кейс за последние 30 дней, повторный запуск будет заблокирован (`policy_blocked`). При последовательном запуске примеров ниже для одного клиента это может сработать. Для сброса: `sqlite3 artifacts/cases.db "DELETE FROM cases WHERE status='done'"`.

3.1 Run various what-if scenarios
```bash
python -m sme_causal.app.run --client-id "C000005" --what-if "New_Product_Offer=1,New_Product_Offer_Type=acquiring"

python -m sme_causal.app.run --json --what-if "Credit_Limit_Change=15.0,Tariff_Discount=1"

python -m sme_causal.app.run --what-if "Credit_Limit_Change=25.0"
```

3.2 Disable specific sources with `--no-psm`, `--no-graph`, `--no-rag` flags. Custom PSM settings:
```bash
python -m sme_causal.app.run --client-id "C000005" --what-if "New_Product_Offer=1,New_Product_Offer_Type=acquiring" --outcome-col Revenue_Growth_Rate --covariates "Industry,Region,Business_Size,Avg_Account_Balance,Avg_Monthly_Inflow,Avg_Monthly_Outflow,Num_Products"

python -m sme_causal.app.run --json --what-if "Credit_Limit_Change=15.0"

python -m sme_causal.app.run --what-if "Credit_Limit_Change=25.0" --no-psm --no-rag

python -m sme_causal.app.run --what-if "Credit_Limit_Change=25.0" --graph-method algo

python -m sme_causal.app.run --what-if "Tariff_Discount=1"
```

3.3 Use free query format in natural language: ask about any intervention with any target metric. Add `--query` or `-q` argument to use this feature.

```bash
python -m sme_causal.app.run --query "Оцените эффект от предложения зарплатного проекта клиенту C000005"

python -m sme_causal.app.run --client-id "C000005" -q "Как изменится баланс клиента клиента, если предложить ему скидку на тариф"

python -m sme_causal.app.run --client-id "C000005" -q "Что если поднять кредитный лимит клиенту на 20%" --outcome-col "Avg_Monthly_Inflow"
```

4. Explore via Streamlit UI

```bash
PYTHONPATH=. streamlit run sme_causal/app/streamlit_app.py
```

Provide `OPENAI_API_KEY` via env/.env or in the UI sidebar. Use the controls to generate data, run LLM inference, visualize the graph, and compare with ground truth.

5. Build and visualize graph (non‑UI)

```bash
python -m sme_causal.app.build_and_visualize_graph --min_conf 0.45
```

This exports `graph_merged.(json|gexf|graphml)` and an interactive `graph_merged.html` to `artifacts/`.

6. Run tests

```bash
# run all tests
pytest -q
```

## Artifacts

By default, files are written under `artifacts/` (configurable via `PATHS_ARTIFACTS_DIR`). Key outputs:

- `synthetic_clients.csv`: Generated dataset.
- `ground_truth_edges.json`: Ground‑truth DAG edges from the generator.
- `llm_edges.json`: Edges inferred by the LLM.
- `hybrid_edges.json`: Edges inferred by the LLM and Algo approaches.
- `edge_report.csv`: Per‑edge comparison vs ground truth (for synthetic data).
- `graph_merged.json|gexf|graphml`: Exported graph in common formats.
- `graph_merged.html`: Interactive PyVis graph (from `build_and_visualize_graph.py`).
- `pipeline.log`, `streamlit.log`: Logs with structured output (Loguru).

## Artifacts

Artifacts from algorithmic graphs are written under `causal_outputs/` (configurable in `config.py`). Key outputs:
- `algo_llm_edges.json`: Edges inferred by the Algorithm approach with LLM validation.
- `graph_consensus.json`: Edges inferred by the Algorithm approach.


## Metrics and reports
Also, every run with evaluation creates a folder in `reports/`. This folders contain metrics report and graph edge-be-edge comparison table.

## RAG data


- `document_corpus/`: directory with txt-docs corpus and metadata.csv
- `chunks.parquet`: file with information in fields about chunk_id, doc_id, text
- `embeddings.parquet`: file with embedds and chunks_id
- `index.faiss`: binary index faiss file


## Configuration

Defaults live in `sme_causal/core/config.py`. Override via environment variables or `.env` (e.g., `LLM_MODEL`, `LLM_TEMPERATURE`, `DATA_N_CLIENTS`, `PATHS_ARTIFACTS_DIR`, logging options). `OPENAI_API_KEY` is required for LLM calls.

## Project Layout

```text
sme_causal/
  agent/
    agent_service.py
  app/
    main.py
    run.py
    build_and_visualize_graph.py
    build_rag.py
    streamlit_app.py
  core/
    llm_clients/
      base.py
      factory.py
      local.py
      openai.py
    columns.py
    config.py
    env.py
    utils.py
    data_io.py
    llm.py
    constants.py
    types.py
  data/
    synth_data.py
  graph/
    build_llm_graph.py
    build_algo_graph.py
    build_algo_llm_graph.py
    build_hybrid_graph.py
    evaluate_graphs.py
    graph_utils.py
    graph_viz.py
  inference/
    llm_graph.py
    psm.py
```

## Notes

- The dataset is synthetic and the causal “ground truth” reflects the generator’s assumptions; it is intended for demos/tests only.
- Network calls require a valid OpenAI API key and the selected model to be available to your account.
