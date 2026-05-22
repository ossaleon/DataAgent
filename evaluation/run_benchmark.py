#!/usr/bin/env python3
"""Standalone benchmark script for evaluating the DataAgent against ground-truth datasets.

Usage:
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 3
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 1 --save-dir ./results
"""

import argparse
import glob as _glob
import json
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Agent.config import AgentConfig
from Agent.schema import ColumnSchema, DatabaseSchema, TableSchema
from Agent.utils import (
    make_csv_evaluator_gt,
    make_csv_evaluator_no_gt,
    make_text_evaluator_gt,
    make_text_evaluator_no_gt,
    make_vis_evaluator_gt,
    make_vis_evaluator_no_gt,
)
from run_agent import run_single  # non-interactive single-run entry point


def _load_schema(data_dir: Optional[str] = None) -> Optional[DatabaseSchema]:
    """Discover and load all *_schema.yaml files from data_dir.

    Mirrors the logic in AgentConfig.from_yaml so that run_benchmark has
    the same multi-table support as run_agent.
    """
    if data_dir is None:
        data_dir = os.path.join(_PROJECT_ROOT, "data")
    data_dir = os.path.abspath(data_dir)
    schema_files = sorted(_glob.glob(os.path.join(data_dir, "*_schema.yaml")))
    if not schema_files:
        return None
    tables = []
    for table_path in schema_files:
        with open(table_path, encoding="utf-8") as tf:
            t = yaml.safe_load(tf)
        columns = [
            ColumnSchema(
                name=c["name"],
                description=c.get("description", c["name"]),
                data_type=c.get("data_type", "VARCHAR"),
                example_values=c.get("example_values"),
                nullable=c.get("nullable", True),
            )
            for c in t.get("columns", [])
        ]
        file_path = t["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(table_path), file_path)
        tables.append(TableSchema(
            name=t["name"],
            description=t.get("description", t["name"]),
            file_path=file_path,
            columns=columns,
        ))
    return DatabaseSchema(tables=tables, compact_threshold=5)


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load and validate a unified GT dataset JSON."""
    with open(path) as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"Empty dataset: {path}")
    first = entries[0]
    if "prompt" not in first:
        raise ValueError("Dataset entries must have a 'prompt' field")
    if "gt_data" not in first and "gt_chart_config" not in first:
        raise ValueError("Dataset entries must have 'gt_data' and/or 'gt_chart_config'")
    return entries


def run_benchmark(
    dataset_path: str,
    *,
    agent_config: Optional[AgentConfig] = None,
    config_path: Optional[str] = None,
    n: int = 1,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
    gt_judge_model: Optional[str] = None,
    gt_judge_provider: Optional[str] = None,
    no_gt_judge_model: Optional[str] = None,
    no_gt_judge_provider: Optional[str] = None,
    save_dir: str = "./evaluation/results",
    data_dir: Optional[str] = None,
    save_execution_artifacts: bool = False,
    enable_codecarbon: bool = False,
    max_prompts: Optional[int] = None,
    config_label: Optional[str] = None,
    vary_step: Optional[str] = None,
    repetition: Optional[str] = None,
) -> pd.DataFrame:
    """Run benchmark against a unified GT dataset.

    Args:
        dataset_path: Path to the benchmark JSON file.
        agent_config: Pre-built AgentConfig to use directly. When provided,
            sampling parameters (n, temp, top_p) are preserved as-is; only
            eval functions are attached from the benchmark GT data.
            Mutually exclusive with config_path / n.
        config_path: Optional path to run_config.yaml for base AgentConfig.
            Ignored when agent_config is provided.
        n: Best-of-N per step. Ignored when agent_config is provided.
        judge_model: Legacy alias for ``gt_judge_model``.
        judge_provider: Legacy alias for ``gt_judge_provider``.
        gt_judge_model: Model for ground-truth text and visualization scoring.
            Defaults to the legacy judge model when supplied, then the agent
            model.
        gt_judge_provider: Provider for the ground-truth judge.
            Defaults to the legacy judge provider when supplied, then the
            agent provider.
        no_gt_judge_model: Model for no-GT analysis and visualization scoring
            used during local selection/diagnostics. Defaults to the tested
            agent model so benchmark judging does not change real-run behavior.
        no_gt_judge_provider: Provider for the no-GT judge. Defaults to the
            tested agent provider.
        save_dir: Directory to save results CSV.

    Returns:
        DataFrame with per-test-case scores.
    """
    entries = load_benchmark_dataset(dataset_path)
    if max_prompts is not None:
        entries = entries[:max_prompts]
    print(f"Loaded {len(entries)} test cases from {dataset_path}")

    # Load schema once (covers all tables: sales, stores, products, ...)
    # Same logic as AgentConfig.from_yaml — without this, SalesDataAgent falls
    # back to single-table legacy mode and JOIN queries on stores/products fail.
    schema = _load_schema(data_dir)
    if schema:
        print(f"Loaded schema: {[t.name for t in schema.tables]}")

    # Determine base config and judge identity.
    # A caller-provided AgentConfig or YAML config path is treated as authoritative:
    # benchmark mode may attach evaluators, but it must not override step params.
    _preserve_config = agent_config is not None or config_path is not None
    if agent_config is not None:
        config = agent_config
        judge_model = judge_model or config.model
        judge_provider = judge_provider or config.provider
    elif config_path:
        config, _run_params, _schema = AgentConfig.from_yaml(config_path)
        judge_model = judge_model or config.model
        judge_provider = judge_provider or config.provider
    else:
        judge_model = judge_model or "gpt-4o-mini"
        judge_provider = judge_provider or "openai"
        config = AgentConfig(
            model=judge_model,
            provider=judge_provider,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

    # GT judging is an offline measurement concern. No-GT judging is part of
    # agent selection behavior, so it follows the tested agent by default.
    # ``judge_*`` remains a compatibility alias for older benchmark commands
    # that configured only one judge pair.
    effective_gt_judge_model = gt_judge_model or judge_model or config.model
    effective_gt_judge_provider = gt_judge_provider or judge_provider or config.provider
    effective_no_gt_judge_model = no_gt_judge_model or config.model
    effective_no_gt_judge_provider = no_gt_judge_provider or config.provider

    results = []
    _SEP  = "=" * 62
    _SEP2 = "-" * 62

    for idx, entry in enumerate(entries):
        prompt = entry["prompt"]
        vis_goal = entry.get("visualization_goal")
        has_vis = entry.get("gt_chart_config") is not None
        has_data = entry.get("gt_data") is not None

        print(f"\n{_SEP}")
        print(f"  PROMPT {idx + 1}/{len(entries)}")
        if repetition is not None:
            print(f"  Repetition :  {repetition}")
        print(f"  Model      :  {config.model}  [{config.provider}]")
        if vary_step is not None:
            print(f"  Vary step  :  {vary_step}")
        if config_label is not None:
            print(f"  Config     :  {config_label}")
        print(f"  GT         :  data={has_data}  vis={has_vis}")
        print(_SEP2)
        print(f"  {prompt}")
        print(_SEP)

        # Configure step-level eval functions for this entry.
        # GT eval functions are used for tracking/logging only (gt_eval_fn).
        # Non-GT eval functions are used for best-of-n selection (eval_fn / batch_eval_fn).
        # When a config object or config file is provided, sampling params are preserved.
        if has_data:
            if not _preserve_config:
                config.lookup_sales_data.n = n
                config.lookup_sales_data.temp_min = 0.1
                config.lookup_sales_data.temp_max = 0.5
            config.lookup_sales_data.gt_eval_fn = make_csv_evaluator_gt(
                ground_truth_csv_text=entry["gt_data"]
            )
            config.lookup_sales_data.batch_eval_fn = make_csv_evaluator_no_gt()
            config.lookup_sales_data.eval_fn = None
            # Force column names to GT columns during standardize_candidate_columns
            # (same as AgentConfig.from_yaml lines 417/423 — missing here was the root cause of csv_iou=0)
            _gt_df = pd.read_csv(pd.io.common.StringIO(entry["gt_data"]))
            config.lookup_sales_data.gt_columns = [c.lower() for c in _gt_df.columns]

        if has_data and entry.get("gt_analysis"):
            if not _preserve_config:
                config.analyzing_data.n = n
                config.analyzing_data.temp_min = 0.1
                config.analyzing_data.temp_max = 0.7
            config.analyzing_data.gt_eval_fn = make_text_evaluator_gt(
                ground_truth_text=entry["gt_analysis"],
                judge_model=effective_gt_judge_model,
                provider=effective_gt_judge_provider,
            )
            config.analyzing_data.eval_fn = make_text_evaluator_no_gt(
                judge_model=effective_no_gt_judge_model,
                provider=effective_no_gt_judge_provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        if has_vis:
            if not _preserve_config:
                config.create_visualization.n = n
                config.create_visualization.temp_min = 0.1
                config.create_visualization.temp_max = 0.5
            config.create_visualization.gt_eval_fn = make_vis_evaluator_gt(
                ground_truth_config=entry["gt_chart_config"],
                ground_truth_code=entry.get("gt_chart_code", ""),
                explicit_requirements=entry.get("explicit_requirements"),
                judge_model=effective_gt_judge_model,
                provider=effective_gt_judge_provider,
            )
            config.create_visualization.eval_fn = make_vis_evaluator_no_gt(
                judge_model=effective_no_gt_judge_model,
                provider=effective_no_gt_judge_provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        # Run agent via run_single (same path as run_agent.py — no logic duplication)
        result = run_single(
            config,
            prompt,
            schema,
            visualization_goal=vis_goal,
            no_vis=not has_vis,
            save_dir=save_dir,
            save_results=False,
            save_execution_artifacts=save_execution_artifacts,
            enable_codecarbon=enable_codecarbon,
        )

        # --- Extract scores from result (same path as run_agent.py) ---
        # gt_eval_fn was configured above and called inside agent.run(); scores
        # are already normalised (column name forcing, LLM judge, etc.).
        gt_scores = result.get("_gt_scores_per_step", {})
        eval_scores = result.get("_step_eval_scores", {})

        # --- Per-prompt timing (bug fix: was the total run time repeated on every row) ---
        step_timings = result.get("_step_timings_sec", {})
        total_time = result.get("_total_run_time_sec")

        # --- Per-step LLM call timings and energy ---
        llm_timings = result.get("_step_llm_timings_sec") or {}
        llm_energy = result.get("_step_llm_energy") or {}

        # --- Energy (populated only when enable_codecarbon=True) ---
        energy = result.get("_energy") or {}

        # Extract GT reasoning from evaluator closures (populated only when score < 1.0)
        csv_iou_val    = gt_scores.get("lookup_sales_data", {}).get("gt_score") if has_data else None
        text_score_val = gt_scores.get("analyzing_data", {}).get("gt_score") if entry.get("gt_analysis") else None
        vis_score_val  = gt_scores.get("create_visualization", {}).get("gt_score") if has_vis else None

        def _get_reasoning(eval_fn, score):
            if score is None or score >= 1.0:
                return None
            return getattr(eval_fn, "_store", {}).get("reasoning")

        csv_reasoning  = _get_reasoning(config.lookup_sales_data.gt_eval_fn, csv_iou_val)
        text_reasoning = _get_reasoning(config.analyzing_data.gt_eval_fn, text_score_val)
        vis_reasoning  = _get_reasoning(config.create_visualization.gt_eval_fn, vis_score_val)

        def _selection_metrics(step_name: str):
            info = eval_scores.get(step_name, {}) or {}
            raw_scores = info.get("scores") or []
            try:
                best_idx = int(info["best_idx"]) if info.get("best_idx") is not None else None
            except (TypeError, ValueError):
                best_idx = None

            numeric_scores = []
            for score in raw_scores:
                try:
                    numeric_scores.append(float(score))
                except (TypeError, ValueError):
                    pass

            margin = None
            if best_idx is not None and 0 <= best_idx < len(numeric_scores) and len(numeric_scores) > 1:
                best_score = numeric_scores[best_idx]
                other_scores = [s for i, s in enumerate(numeric_scores) if i != best_idx]
                if other_scores:
                    margin = best_score - max(other_scores)

            return {
                "candidate_count": len(raw_scores) if raw_scores else None,
                "selected_candidate_index": best_idx,
                "selection_score_margin": margin,
                "candidate_scores": json.dumps(raw_scores) if raw_scores else None,
            }

        lookup_selection = _selection_metrics("lookup_sales_data")
        analysis_selection = _selection_metrics("analyzing_data")
        vis_selection = _selection_metrics("create_visualization")

        cot_diagnostics = result.get("_cot_diagnostics_per_step", {}) or {}

        def _cot_metrics(step_name: str):
            info = cot_diagnostics.get(step_name, {}) or {}
            similarities = info.get("similarities") or []
            return {
                "requested_iterations": info.get("requested_iterations"),
                "attempted_iterations": info.get("attempted_iterations"),
                "executed_iterations": info.get("executed_iterations"),
                "early_stop": info.get("early_stop"),
                "stop_reason": info.get("stop_reason"),
                "final_similarity": info.get("final_similarity"),
                "similarities": json.dumps(similarities) if similarities else None,
            }

        lookup_cot = _cot_metrics("lookup_sales_data")
        analysis_cot = _cot_metrics("analyzing_data")
        vis_cot = _cot_metrics("create_visualization")

        # Override reasoning with timeout messages when a step or its judge timed out
        _step_errors = result.get("_step_errors") or {}
        _step_to_reasoning = {
            "lookup_sales_data":        "csv_reasoning",
            "lookup_sales_data_judge":  "csv_reasoning",
            "analyzing_data":           "text_reasoning",
            "analyzing_data_judge":     "text_reasoning",
            "create_visualization":     "vis_reasoning",
            "create_visualization_judge": "vis_reasoning",
        }
        for _err_key, _err_msg in _step_errors.items():
            if "TIMEOUT" in _err_msg and _err_key in _step_to_reasoning:
                _target = _step_to_reasoning[_err_key]
                if _target == "csv_reasoning":
                    csv_reasoning = _err_msg
                elif _target == "text_reasoning":
                    text_reasoning = _err_msg
                elif _target == "vis_reasoning":
                    vis_reasoning = _err_msg

        has_timeout = any("TIMEOUT" in str(msg) for msg in _step_errors.values())

        row = {
            "test_case_id": idx,
            "prompt": prompt,
            "difficulty": entry.get("difficulty"),
            "has_data_gt": has_data,
            "has_analysis_gt": bool(entry.get("gt_analysis")),
            "has_vis_gt": has_vis,
            "gen_sql": " ".join((result.get("sql_query", "") or "").split()),
            # GT scores — same source as run_metadata.json accuracy.ground_truth_scores
            "csv_iou":    csv_iou_val,
            "text_score": text_score_val,
            "vis_score":  vis_score_val,
            # GT reasoning — populated only when the corresponding score < 1.0
            "csv_iou_reasoning":    csv_reasoning,
            "text_score_reasoning": text_reasoning,
            "vis_score_reasoning":  vis_reasoning,
            "timeout": has_timeout,
            "step_errors": json.dumps(_step_errors) if _step_errors else None,
            # No-GT quality scores (BoN selector) — same source as run_metadata.json accuracy.step_eval_scores
            "csv_eval_score":  eval_scores.get("lookup_sales_data", {}).get("best_score") if has_data else None,
            "text_eval_score": eval_scores.get("analyzing_data", {}).get("best_score") if entry.get("gt_analysis") else None,
            "vis_eval_score":  eval_scores.get("create_visualization", {}).get("best_score") if has_vis else None,
            # Best-of-N / CoT selection diagnostics. These are especially useful
            # for thesis compute-expansion runs where extra calls must justify
            # their extra energy and latency.
            "lookup_candidate_count": lookup_selection["candidate_count"],
            "lookup_selected_candidate_index": lookup_selection["selected_candidate_index"],
            "lookup_selection_score_margin": lookup_selection["selection_score_margin"],
            "lookup_candidate_scores": lookup_selection["candidate_scores"],
            "analysis_candidate_count": analysis_selection["candidate_count"],
            "analysis_selected_candidate_index": analysis_selection["selected_candidate_index"],
            "analysis_selection_score_margin": analysis_selection["selection_score_margin"],
            "analysis_candidate_scores": analysis_selection["candidate_scores"],
            "vis_candidate_count": vis_selection["candidate_count"],
            "vis_selected_candidate_index": vis_selection["selected_candidate_index"],
            "vis_selection_score_margin": vis_selection["selection_score_margin"],
            "vis_candidate_scores": vis_selection["candidate_scores"],
            # CoT depth diagnostics. ``executed_iterations`` counts completed
            # iterations including the initial call, so a value lower than
            # ``requested_iterations`` marks convergence/error early stops.
            "lookup_cot_requested_iterations": lookup_cot["requested_iterations"],
            "lookup_cot_attempted_iterations": lookup_cot["attempted_iterations"],
            "lookup_cot_executed_iterations": lookup_cot["executed_iterations"],
            "lookup_cot_early_stop": lookup_cot["early_stop"],
            "lookup_cot_stop_reason": lookup_cot["stop_reason"],
            "lookup_cot_final_similarity": lookup_cot["final_similarity"],
            "lookup_cot_similarities": lookup_cot["similarities"],
            "analysis_cot_requested_iterations": analysis_cot["requested_iterations"],
            "analysis_cot_attempted_iterations": analysis_cot["attempted_iterations"],
            "analysis_cot_executed_iterations": analysis_cot["executed_iterations"],
            "analysis_cot_early_stop": analysis_cot["early_stop"],
            "analysis_cot_stop_reason": analysis_cot["stop_reason"],
            "analysis_cot_final_similarity": analysis_cot["final_similarity"],
            "analysis_cot_similarities": analysis_cot["similarities"],
            "vis_cot_requested_iterations": vis_cot["requested_iterations"],
            "vis_cot_attempted_iterations": vis_cot["attempted_iterations"],
            "vis_cot_executed_iterations": vis_cot["executed_iterations"],
            "vis_cot_early_stop": vis_cot["early_stop"],
            "vis_cot_stop_reason": vis_cot["stop_reason"],
            "vis_cot_final_similarity": vis_cot["final_similarity"],
            "vis_cot_similarities": vis_cot["similarities"],
            # Per-step total wall-clock timings
            "elapsed_sec":        round(total_time, 2) if total_time is not None else None,
            "lookup_time_sec":    round(step_timings.get("lookup_sales_data", 0), 2),
            "analyzing_time_sec": round(step_timings.get("analyzing_data", 0), 2),
            "vis_time_sec":       round(step_timings.get("create_visualization", 0), 2),
            # Per-step LLM call timings (sum of all LLM invocations incl. BoN, CoT, eval judges)
            "lookup_llm_time_sec":    round(llm_timings.get("lookup_sales_data", 0), 3),
            "analyzing_llm_time_sec": round(llm_timings.get("analyzing_data", 0), 3),
            "vis_llm_time_sec":       round(llm_timings.get("create_visualization", 0), 3),
            # Per-step LLM call energy — all 5 CodeCarbon fields (None when CodeCarbon disabled)
            "lookup_llm_energy_kwh":       (llm_energy.get("lookup_sales_data") or {}).get("energy_consumed_kwh"),
            "lookup_llm_cpu_energy_kwh":   (llm_energy.get("lookup_sales_data") or {}).get("cpu_energy_kwh"),
            "lookup_llm_gpu_energy_kwh":   (llm_energy.get("lookup_sales_data") or {}).get("gpu_energy_kwh"),
            "lookup_llm_ram_energy_kwh":   (llm_energy.get("lookup_sales_data") or {}).get("ram_energy_kwh"),
            "lookup_llm_emissions_co2":    (llm_energy.get("lookup_sales_data") or {}).get("emissions_kg_co2"),
            "analyzing_llm_energy_kwh":       (llm_energy.get("analyzing_data") or {}).get("energy_consumed_kwh"),
            "analyzing_llm_cpu_energy_kwh":   (llm_energy.get("analyzing_data") or {}).get("cpu_energy_kwh"),
            "analyzing_llm_gpu_energy_kwh":   (llm_energy.get("analyzing_data") or {}).get("gpu_energy_kwh"),
            "analyzing_llm_ram_energy_kwh":   (llm_energy.get("analyzing_data") or {}).get("ram_energy_kwh"),
            "analyzing_llm_emissions_co2":    (llm_energy.get("analyzing_data") or {}).get("emissions_kg_co2"),
            "vis_llm_energy_kwh":       (llm_energy.get("create_visualization") or {}).get("energy_consumed_kwh"),
            "vis_llm_cpu_energy_kwh":   (llm_energy.get("create_visualization") or {}).get("cpu_energy_kwh"),
            "vis_llm_gpu_energy_kwh":   (llm_energy.get("create_visualization") or {}).get("gpu_energy_kwh"),
            "vis_llm_ram_energy_kwh":   (llm_energy.get("create_visualization") or {}).get("ram_energy_kwh"),
            "vis_llm_emissions_co2":    (llm_energy.get("create_visualization") or {}).get("emissions_kg_co2"),
            # Run-level energy (None when CodeCarbon disabled or unavailable)
            "energy_consumed_kwh": energy.get("energy_consumed_kwh"),
            "cpu_energy_kwh":      energy.get("cpu_energy_kwh"),
            "gpu_energy_kwh":      energy.get("gpu_energy_kwh"),
            "ram_energy_kwh":      energy.get("ram_energy_kwh"),
            "emissions_kg_co2":    energy.get("emissions_kg_co2"),
        }

        results.append(row)
        print(
            f"\nScores: csv_iou={row['csv_iou']}, text={row['text_score']}, vis={row['vis_score']}"
            f" | elapsed={row['elapsed_sec']}s"
        )

    # Build results DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for col in ["csv_iou", "text_score", "vis_score"]:
        valid = df[col].dropna()
        if not valid.empty:
            print(f"  {col}: mean={valid.mean():.3f}, min={valid.min():.3f}, max={valid.max():.3f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "benchmark_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataAgent benchmark against GT dataset")
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("--n", type=int, default=1, help="Best-of-N per step (default: 1)")
    parser.add_argument("--config", default=None, help="Path to run_config.yaml")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Legacy alias for --gt-judge-model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-provider",
        default="openai",
        help="Legacy alias for --gt-judge-provider (default: openai)",
    )
    parser.add_argument("--gt-judge-model", default=None, help="GT text/visual judge model")
    parser.add_argument("--gt-judge-provider", default=None, help="GT text/visual judge provider")
    parser.add_argument(
        "--no-gt-judge-model",
        default=None,
        help="No-GT selection judge model (default: tested agent model)",
    )
    parser.add_argument(
        "--no-gt-judge-provider",
        default=None,
        help="No-GT selection judge provider (default: tested agent provider)",
    )
    parser.add_argument("--save-dir", default="./evaluation/results", help="Output directory")

    args = parser.parse_args()

    run_benchmark(
        args.dataset,
        config_path=args.config,
        n=args.n,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        gt_judge_model=args.gt_judge_model,
        gt_judge_provider=args.gt_judge_provider,
        no_gt_judge_model=args.no_gt_judge_model,
        no_gt_judge_provider=args.no_gt_judge_provider,
        save_dir=args.save_dir,
    )
