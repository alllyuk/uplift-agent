"""
Automatic evaluation of LLM responses using Pollux framework.

This module evaluates agent answers about causal inference and uplift effects
using the Pollux judge model with task-specific criteria from Pollux criteria dataset.

## Key Components

1. **PolluxEvaluator**: Wrapper around Pollux model for scoring LLM responses
2. **UpliftAgentEvaluator**: Main orchestrator that runs complete evaluations
3. **PromptCapture**: Captures full context (prompts, enrichments) for evaluation
4. **Explanation**: Dataclass with embedded evaluation context

## Usage Examples

### Direct CLI Usage

# Evaluate a specific client with default settings
python -m sme_causal.app.evaluate_pollux \\
    --client-id C000001 \\
    --output results.csv \\
    --device cuda

# Evaluate first 100 clients from dataset
python -m sme_causal.app.evaluate_pollux \\
    --num-clients 100 \\
    --what-if "Tariff_Discount=1,Credit_Limit_Change=20" \\
    --use-graph --use-rag \\
    --output batch_results.csv

# With custom what-if intervention and all enrichments enabled
python -m sme_causal.app.evaluate_pollux \\
    --client-id C000001 \\
    --what-if "Tariff_Discount=1,New_Product_Offer=2" \\
    --use-graph --use-rag --use-psm \\
    --output detailed_results.csv

# Evaluate natural language query for first 5 clients
python -m sme_causal.app.evaluate_pollux \\
    --query "Should we offer acquiring to client C000001?" \\
    --use-graph --use-rag \\
    --output query_results.csv


### Programmatic Usage

from sme_causal.app.evaluate_pollux import UpliftAgentEvaluator
from sme_causal.agent.agent_service import CausalAgent
from sme_causal.core.config import get_config
from sme_causal.core.columns import CLIENT_ID
import pandas as pd

# Initialize
evaluator = UpliftAgentEvaluator(device="cuda")
agent = CausalAgent(graph_method="llm")
cfg = get_config()

df = pd.read_csv(cfg.synthetic_clients_path)


# Evaluate single client
baseline_eval, whatif_eval = evaluator.evaluate_agent_run(
    agent=agent,
    df=df,
    client_id="C000001",
    delta={"Tariff_Discount": 1},
    use_graph=True,
    use_rag=True,
)

# Save results
evaluator.save_results([baseline_eval, whatif_eval], "results.csv")

# Evaluate multiple clients programmatically
all_evaluations = []
client_ids = df[CLIENT_ID].iloc[:100].tolist()  # First 100 clients

for client_id in client_ids:
    baseline_eval, whatif_eval = evaluator.evaluate_agent_run(
        agent=agent,
        df=df,
        client_id=client_id,
        delta={"Tariff_Discount": 1, "Credit_Limit_Change": 20},
        use_graph=True,
        use_rag=True,
        use_psm=True,
    )
    all_evaluations.extend([baseline_eval, whatif_eval])

# Save all results in one file
evaluator.save_results(all_evaluations, "batch_results.csv")


### Evaluate Existing Explanations
from sme_causal.agent.agent_service import CausalAgent

# After generating explanations with embedded context
agent = CausalAgent()
base_ctx = agent.build_context_for_client(df, "C000001")
explanation = agent.explain_what_if(base_ctx, {"Tariff_Discount": 1})

# Evaluate using embedded context
evaluation = evaluator.evaluate_from_explanation(
    client_id="C000001",
    scenario_name="what_if",
    explanation=explanation,  # Automatically uses embedded context
)
```

## Output Format

Results are saved as CSV with columns:
- client_id: Client identifier
- scenario: "baseline" or "what_if"
- criteria: Evaluation criterion name
- score: Score (0, 1, 2 from Pollux) or None if failed
- error: Error message if evaluation failed
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

from sme_causal.agent.agent_service import CausalAgent, Explanation, QueryParser, ParsedQuery
from sme_causal.core.config import get_config
from sme_causal.core.columns import CLIENT_ID
from sme_causal.core.utils import configure_logging, parse_client_id_and_intent
from sme_causal.app.run import _run_psm
from sme_causal.data.pollux_criteria import (
    CriteriaItem,
    get_all_criteria,
)


MODEL_PATH = "ai-forever/pollux-judge-7b"


@dataclass
class EvaluationResult:
    """Result of evaluating a single answer."""
    client_id: str
    scenario_name: str  # "baseline" or "what_if"
    criteria_name: str
    score: Optional[float]  # 0, 1, 2 or None if parsing failed
    raw_response: str
    error: Optional[str] = None


@dataclass
class ScenarioEvaluation:
    """Complete evaluation for one scenario."""
    client_id: str
    scenario_name: str
    instruction: str  # Full prompt sent to LLM
    answer: str  # Raw LLM response
    results: List[EvaluationResult]


def extract_score(text: str) -> Optional[float]:
    """Extract numeric score from Pollux response.

    Looks for patterns like "[RESULT] 0.5" or "[RESULT] 2".

    Args:
        text: Raw response from Pollux model.

    Returns:
        Float score or None if not found.
    """
    res = re.search("(?<=\[RESULT\] )\s*\d+\.\d+", text)
    if res is not None:
        return float(res.group(0).strip())
    else:
        res = re.search("(?<=\[RESULT\] )\s*\d", text)
        if res is not None:
            return float(res.group(0).strip())
        else:
            return None


PROMPT_TEMPLATE_NO_REFERENCE = """### Задание для оценки:
{instruction}

### Ответ для оценки:
{answer}

### Критерий оценки:
{criteria_name}

### Шкала оценивания по критерию:
{criteria_rubrics}
"""


class PolluxEvaluator:
    """Wrapper around Pollux model for evaluating uplift agent responses."""

    def __init__(self, device: str = "auto"):
        """Initialize Pollux model and tokenizer.

        Args:
            device: Device to load model on ("auto", "cuda", "cpu", etc).
        """
        logger.info(f"Loading Pollux model from {MODEL_PATH}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        logger.info("Pollux model loaded successfully")

    def generate_score(self, prompt: str, max_tokens: int = 4096) -> str:
        """Generate evaluation score from Pollux model.

        Args:
            prompt: Formatted prompt with instruction, answer, and criteria.
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw text response from model.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        sequence_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, sequence_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def evaluate(
        self,
        instruction: str,
        answer: str,
        criteria_item: CriteriaItem,
    ) -> Tuple[Optional[float], str]:
        """Evaluate single answer against one criterion.

        Args:
            instruction: The instruction/prompt sent to LLM.
            answer: The LLM's answer to evaluate.
            criteria_item: Criteria specification.

        Returns:
            Tuple of (score, raw_response). Score is None if parsing failed.
        """
        prompt = PROMPT_TEMPLATE_NO_REFERENCE.format(
            instruction=instruction,
            answer=answer,
            criteria_name=criteria_item.criteria_name,
            criteria_rubrics=criteria_item.rubrics,
        )

        try:
            response = self.generate_score(prompt)
            score = extract_score(response)
            return score, response
        except Exception as e:
            logger.error(f"Error during Pollux evaluation: {e}")
            return None, str(e)


class PromptCapture:
    """Captures full context: instruction prompt, LLM response, enrichment sources."""

    def __init__(self):
        self.instruction: str = ""
        self.answer: str = ""
        self.graph_context: str = ""
        self.rag_context: str = ""
        self.psm_summary: str = ""
        self.base_context: Dict = {}
        self.delta_changes: Dict = {}

    def build_full_instruction(self) -> str:
        """Reconstruct the full instruction sent to LLM.

        Returns:
            Complete instruction including all enrichments.
        """
        return self._build_instruction(
            instruction=self.instruction,
            graph_context=self.graph_context,
            rag_context=self.rag_context,
            psm_summary=self.psm_summary,
            base_context=self.base_context,
            delta_changes=self.delta_changes,
        )

    @staticmethod
    def _build_instruction(
        instruction: str,
        graph_context: str = "",
        rag_context: str = "",
        psm_summary: str = "",
        base_context: Optional[Dict] = None,
        delta_changes: Optional[Dict] = None,
    ) -> str:
        """Build full instruction from components.

        This is the single source of truth for instruction assembly.

        Args:
            instruction: Base instruction text.
            graph_context: Graph DSL context.
            rag_context: RAG enrichment.
            psm_summary: PSM metrics.
            base_context: Client profile context.
            delta_changes: What-if intervention changes.

        Returns:
            Complete instruction including all enrichments.
        """
        parts = []

        if graph_context:
            parts.append(f"[GRAPH_DSL]\n{graph_context}\n")

        if rag_context:
            parts.append(f"RAG-КОНТЕКСТ:\n{rag_context}\n")

        if psm_summary:
            parts.append(f"РЕЗУЛЬТАТЫ PSM-АНАЛИЗА:\n{psm_summary}\n")

        if base_context:
            parts.append(f"БАЗОВЫЙ ПРОФИЛЬ КЛИЕНТА:\n{json.dumps(base_context, ensure_ascii=False, indent=2)}\n")

        if delta_changes:
            parts.append(f"ПРЕДЛАГАЕМЫЕ ИЗМЕНЕНИЯ (WHAT_IF):\n{json.dumps(delta_changes, ensure_ascii=False, indent=2)}\n")

        parts.append(instruction)

        return "\n".join(parts)


class UpliftAgentEvaluator:
    """Main class for evaluating uplift agent answers with Pollux."""

    def __init__(self, device: str = "auto"):
        """Initialize evaluator.

        Args:
            device: Device for Pollux model.
        """
        self.evaluator = PolluxEvaluator(device=device)
        self.criteria_dict = get_all_criteria()

        if not self.criteria_dict:
            logger.warning("No criteria loaded. Evaluation may be limited.")

    def evaluate_from_explanation(
        self,
        client_id: str,
        scenario_name: str,
        explanation: Explanation,
    ) -> ScenarioEvaluation:
        """Evaluate an Explanation object that has embedded context.

        This is the recommended way when using CausalAgent, as it automatically
        captures all enrichment sources (RAG, PSM, graph).

        Args:
            client_id: Client identifier.
            scenario_name: Scenario name.
            explanation: Explanation object with embedded context.

        Returns:
            ScenarioEvaluation with results for all criteria.
        """
        # Use the full prompt if available, otherwise construct it
        full_instruction = (
            explanation.full_prompt
            if explanation.full_prompt
            else self._reconstruct_instruction(explanation)
        )

        return self._evaluate_all_criteria(
            client_id=client_id,
            scenario_name=scenario_name,
            full_instruction=full_instruction,
            answer=explanation.raw_text,
        )

    def _evaluate_all_criteria(
        self,
        client_id: str,
        scenario_name: str,
        full_instruction: str,
        answer: str,
    ) -> ScenarioEvaluation:
        """Helper method to evaluate answer against all criteria.

        Eliminates code duplication between evaluate_from_explanation and evaluate_explanation.

        Args:
            client_id: Client identifier.
            scenario_name: Scenario name.
            full_instruction: Complete instruction including all enrichments.
            answer: LLM response to evaluate.

        Returns:
            ScenarioEvaluation with results for all criteria.
        """
        results = []
        for criteria_name, criteria_item in self.criteria_dict.items():
            logger.info(
                f"Evaluating {scenario_name}::{client_id} against '{criteria_name}'..."
            )

            score, raw_response = self.evaluator.evaluate(
                instruction=full_instruction,
                answer=answer,
                criteria_item=criteria_item,
            )

            error = None if score is not None else "Failed to parse score"

            result = EvaluationResult(
                client_id=client_id,
                scenario_name=scenario_name,
                criteria_name=criteria_name,
                score=score,
                raw_response=raw_response,
                error=error,
            )
            results.append(result)

        return ScenarioEvaluation(
            client_id=client_id,
            scenario_name=scenario_name,
            instruction=full_instruction,
            answer=answer,
            results=results,
        )

    @staticmethod
    def _reconstruct_instruction(explanation: Explanation) -> str:
        """Reconstruct full instruction from Explanation embedded context.

        Uses the shared instruction builder from PromptCapture.
        """
        return PromptCapture._build_instruction(
            instruction="",  # Instructions are embedded in other fields
            graph_context=explanation.graph_context or "",
            rag_context=explanation.rag_context or "",
            psm_summary=explanation.psm_summary or "",
            base_context=explanation.base_context,
            delta_changes=explanation.delta_changes,
        )

    def evaluate_explanation(
        self,
        client_id: str,
        scenario_name: str,
        instruction: str,
        answer: str,
        graph_context: str = "",
        rag_context: str = "",
        psm_summary: str = "",
        base_context: Optional[Dict] = None,
        delta_changes: Optional[Dict] = None,
    ) -> ScenarioEvaluation:
        """Evaluate a single explanation against all relevant criteria.

        Can also accept an Explanation object directly with embedded context.

        Args:
            client_id: Client identifier.
            scenario_name: Scenario name (e.g., "baseline", "what_if").
            instruction: Base instruction template (or can be full prompt if explanation has it).
            answer: Raw LLM response (or Explanation.raw_text).
            graph_context: Optional graph DSL context.
            rag_context: Optional RAG enrichment.
            psm_summary: Optional PSM analysis summary.
            base_context: Optional base client context.
            delta_changes: Optional intervention changes.

        Returns:
            ScenarioEvaluation with results for all criteria.
        """
        # Always build full instruction to ensure all context is properly assembled
        full_instruction = PromptCapture._build_instruction(
            instruction=instruction,
            graph_context=graph_context,
            rag_context=rag_context,
            psm_summary=psm_summary,
            base_context=base_context,
            delta_changes=delta_changes,
        )

        return self._evaluate_all_criteria(
            client_id=client_id,
            scenario_name=scenario_name,
            full_instruction=full_instruction,
            answer=answer,
        )

    def evaluate_agent_run(
        self,
        agent: CausalAgent,
        df: pd.DataFrame,
        client_id: str,
        delta: Dict[str, object],
        use_graph: bool = True,
        use_rag: bool = True,
        use_psm: bool = True,
        rag_query_text: Optional[str] = None,
    ) -> Tuple[ScenarioEvaluation, ScenarioEvaluation]:
        """Run complete evaluation for baseline and what-if scenarios.

        Uses the CausalAgent to generate explanations with embedded context,
        then evaluates them with Pollux.

        Args:
            agent: Initialized CausalAgent.
            df: Dataset with client data.
            client_id: Client to evaluate.
            delta: Intervention changes.
            use_graph: Whether to include graph context.
            use_rag: Whether to include RAG context.
            use_psm: Whether to include PSM context.
            rag_query_text: Optional natural language query text for RAG enrichment.

        Returns:
            Tuple of (baseline_eval, whatif_eval).
        """
        # Build context
        base_ctx = agent.build_context_for_client(df, client_id)

        # Baseline evaluation
        logger.info(f"Evaluating baseline for client {client_id}...")
        baseline_expl = agent.explain_client(
            client_ctx=base_ctx,
            use_graph=use_graph,
        )

        baseline_eval = self.evaluate_from_explanation(
            client_id=client_id,
            scenario_name="baseline",
            explanation=baseline_expl,
        )

        psm_metrics_for_llm = None
        if use_psm:
            psm_result = _run_psm(
                df,
                delta,
            )

            if psm_result.get("ok"):
                psm_metrics_for_llm = {
                    "att": psm_result.get("att"),
                    "ate": psm_result.get("ate"),
                    "n_pairs": psm_result.get("n_pairs"),
                    "n_treated": psm_result.get("n_treated"),
                    "n_control": psm_result.get("n_control"),
                    "treatment_col": psm_result.get("treatment_col"),
                    "outcome_col": psm_result.get("outcome_col"),
                    "threshold": psm_result.get("threshold"),
                    "caliper": psm_result.get("caliper"),
                    "covariates": psm_result.get("covariates"),
                }

        # What-if evaluation
        logger.info(f"Evaluating what-if scenario for client {client_id}...")
        whatif_expl = agent.explain_what_if(
            base_ctx=base_ctx,
            delta_changes=delta,
            use_graph=use_graph,
            use_rag=use_rag,
            rag_query_text=rag_query_text,
            psm_metrics=psm_metrics_for_llm
        )

        whatif_eval = self.evaluate_from_explanation(
            client_id=client_id,
            scenario_name="what_if",
            explanation=whatif_expl,
        )

        return baseline_eval, whatif_eval

    def save_results(
        self,
        evaluations: List[ScenarioEvaluation],
        output_path: Path,
    ) -> None:
        """Save evaluation results to CSV file.

        Args:
            evaluations: List of scenario evaluations.
            output_path: Path to save results (CSV format).
        """
        results_data = []

        for eval in evaluations:
            for result in eval.results:
                results_data.append({
                    "client_id": result.client_id,
                    "scenario": result.scenario_name,
                    "criteria": result.criteria_name,
                    "score": result.score,
                    "error": result.error,
                })

        df_results = pd.DataFrame(results_data)
        output_path = Path(output_path)  # Ensure Path object
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Print summary statistics
        if not df_results.empty:
            print("\n=== Evaluation Summary ===")
            for scenario in df_results["scenario"].unique():
                subset = df_results[df_results["scenario"] == scenario]
                valid_scores = subset[subset["score"].notna()]["score"]
                if len(valid_scores) > 0:
                    avg_score = valid_scores.mean()
                    print(f"{scenario}: avg_score={avg_score:.2f}, n={len(valid_scores)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate uplift agent responses using Pollux framework"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Specific Client ID to evaluate (overrides --num-clients if provided)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        help="Number of clients to evaluate (default: 1, ignored if --client-id is set)",
    )
    parser.add_argument(
        "--what-if",
        type=str,
        default="Tariff_Discount=1,Credit_Limit_Change=20",
        help="What-if intervention (comma-separated key=value pairs)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Natural language query (e.g. 'Should we offer acquiring to C000123?').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for Pollux model (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Include causal graph in evaluation",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Include RAG context in evaluation",
    )
    parser.add_argument(
        "--use-psm",
        action="store_true",
        help="Include PSM metrics in evaluation",
    )

    args = parser.parse_args()

    # Setup logging
    cfg = get_config()
    configure_logging(
        cfg.pipeline_log_path,
        cfg.logging,
        add_stdout=True,
        stdout_format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.info("Initializing Pollux-based evaluation framework...")

    # Load config and data
    csv_path = cfg.synthetic_clients_path
    if not csv_path.exists():
        logger.error(f"Dataset not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} clients from {csv_path}")

    # Determine clients to evaluate
    if args.client_id:
        # Single specific client
        if args.client_id not in df[CLIENT_ID].values:
            logger.error(f"Client {args.client_id} not found in dataset")
            sys.exit(1)
        client_ids = [args.client_id]
        logger.info(f"Evaluating specific client: {args.client_id}")
    else:
        # Multiple clients
        num_clients = min(args.num_clients, len(df))
        client_ids = df[CLIENT_ID].iloc[:num_clients].tolist()
        logger.info(f"Evaluating {num_clients} clients from dataset (total available: {len(df)})")

    # Parse input: query or explicit what-if
    delta = {}
    rag_query_text: Optional[str] = None
    query_analysis_label: str = ""
    extracted_client_id = None

    if args.query:
        logger.info(f"Processing natural language query: '{args.query}'")

        # Check OpenAI API Key for query parsing
        if not get_config().effective_openai_api_key:
            logger.error("No OpenAI API key found. Natural language query requires LLM access.")
            sys.exit(2)

        # Extract Client ID from query text
        extracted_client_id, cleaned_query = parse_client_id_and_intent(args.query)

        # Parse intent via LLM
        cfg = get_config()
        query_parser = QueryParser(model=cfg.llm.model_name, temperature=0.0)
        parsed_data: Optional[ParsedQuery] = query_parser.parse(cleaned_query)

        if parsed_data:
            delta = parsed_data.delta
            query_analysis_label = parsed_data.label
            rag_query_text = cleaned_query
            logger.info(f"Query parsed successfully: action_type={parsed_data.action_type}, delta={delta}")
        else:
            logger.warning("Failed to parse query intent. Using empty delta.")
            rag_query_text = cleaned_query
    elif args.what_if:
        logger.info(f"Using explicit what-if: {args.what_if}")
        for pair in args.what_if.split(","):
            if "=" not in pair:
                logger.warning(f"Skipping invalid pair: {pair}")
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                # Try float first
                delta[k] = float(v)
            except ValueError:
                # Then try int
                try:
                    delta[k] = int(v)
                except ValueError:
                    # Keep as string
                    delta[k] = v
    else:
        logger.warning("No query or what-if interventions provided. Using empty delta.")

    if not delta:
        logger.warning("Delta interventions are empty. Agent behavior may be undefined.")

    # If query extracted a specific client ID and no explicit --client-id provided, use it
    if extracted_client_id and not args.client_id and args.query:
        if extracted_client_id in df[CLIENT_ID].values:
            client_ids = [extracted_client_id]
            logger.info(f"Using client ID extracted from query: {extracted_client_id}")
        else:
            logger.warning(f"Client ID from query not found in dataset: {extracted_client_id}")

    logger.info(f"Evaluating {len(client_ids)} client(s) with interventions: {delta}")

    # Initialize evaluator and agent
    logger.info(f"Loading Pollux model on device={args.device}...")
    evaluator = UpliftAgentEvaluator(device=args.device)

    logger.info("Initializing CausalAgent...")
    agent = CausalAgent(graph_method="llm")

    # Run evaluation for all selected clients
    all_evaluations = []
    for idx, client_id in enumerate(client_ids, 1):
        logger.info(f"\n[{idx}/{len(client_ids)}] Evaluating client {client_id}...")
        try:
            baseline_eval, whatif_eval = evaluator.evaluate_agent_run(
                agent=agent,
                df=df,
                client_id=client_id,
                delta=delta,
                use_graph=args.use_graph,
                use_rag=args.use_rag,
                use_psm=args.use_psm,
                rag_query_text=rag_query_text,
            )
            # Verify that results contain valid scores
            baseline_has_scores = any(r.score is not None for r in baseline_eval.results)
            whatif_has_scores = any(r.score is not None for r in whatif_eval.results)

            if baseline_has_scores or whatif_has_scores:
                all_evaluations.extend([baseline_eval, whatif_eval])
                logger.info(f"Completed evaluation for client {client_id}")
            else:
                logger.warning(f"No valid scores obtained for client {client_id}")
        except Exception as e:
            logger.error(f"Error evaluating client {client_id}: {e}", exc_info=True)
            continue

    # Save results
    if all_evaluations:
        logger.info(f"Saving {len(all_evaluations)} scenario evaluations to {args.output}...")
        evaluator.save_results(
            all_evaluations,
            output_path=args.output,
        )
        logger.info(f"Evaluation completed! Results saved to {args.output}")
    else:
        logger.error("No successful evaluations to save")
        sys.exit(1)
