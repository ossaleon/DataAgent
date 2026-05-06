#!/usr/bin/env python3
"""Run deterministic benchmark configs from a YAML manifest.

This is the deterministic counterpart to ``bulk_runner.py``.  It keeps the
same step-ablation idea, but uses named configs instead of random sampling so
the results are easier to report in a thesis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Agent.config import AgentConfig, StepConfig  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from evaluation.search_space import SearchSpace  # noqa: E402


STEPS = ("lookup_sales_data", "analyzing_data", "create_visualization")
SCORE_COLS = ("csv_iou", "text_score", "vis_score")
ENERGY_COLS = ("energy_consumed_kwh", "gpu_energy_kwh", "emissions_kg_co2")


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")
    if not manifest.get("configs"):
        raise ValueError("Manifest must contain a non-empty 'configs' list")
    return manifest


def _step_from_dict(step_name: str, spec: Dict[str, Any]) -> StepConfig:
    data = dict(spec)
    data["step_name"] = step_name
    sc = StepConfig.from_dict(data)
    sc.step_name = step_name
    sc.use_cache = bool(data.get("use_cache", False))
    return sc


def _build_config(
    manifest: Dict[str, Any],
    config_spec: Dict[str, Any],
    *,
    provider: str,
    model: str,
    ollama_url: str,
    openai_api_key: Optional[str],
) -> AgentConfig:
    cfg = AgentConfig(
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        openai_api_key=openai_api_key,
    )

    defaults = manifest.get("defaults", {})
    overrides = config_spec.get("steps", {})
    for step_name in STEPS:
        step_spec = dict(defaults.get(step_name, {}))
        step_spec.update(overrides.get(step_name, {}))
        cfg.set_step_config(step_name, _step_from_dict(step_name, step_spec))

    return cfg


def _config_record(config_id: int, cfg: AgentConfig, spec: Dict[str, Any]) -> Dict[str, Any]:
    record = SearchSpace.config_to_record(config_id, cfg)
    record["config_name"] = spec["name"]
    record["vary_step"] = spec.get("vary_step")
    record["axis"] = spec.get("axis")
    record["description"] = spec.get("description")
    return record


def _mean_or_none(series: pd.Series) -> Optional[float]:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _aggregate_results(save_dir: Path, records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames: List[pd.DataFrame] = []
    configs_df = pd.DataFrame(records)

    for config_dir in sorted(save_dir.glob("config_*")):
        result_path = config_dir / "benchmark_results.csv"
        if not result_path.exists():
            continue
        try:
            config_id = int(config_dir.name.split("_")[-1])
        except ValueError:
            continue
        df = pd.read_csv(result_path)
        df["config_id"] = config_id
        detail_frames.append(df)

    if not detail_frames:
        raise ValueError(f"No benchmark_results.csv files found under {save_dir}")

    detail = pd.concat(detail_frames, ignore_index=True)
    detail = detail.merge(configs_df, on="config_id", how="left")

    summary_rows: List[Dict[str, Any]] = []
    for config_id, group in detail.groupby("config_id", sort=True):
        row: Dict[str, Any] = {"config_id": int(config_id), "n_test_cases": int(len(group))}
        for col in SCORE_COLS:
            if col in group:
                row[f"{col}_mean"] = _mean_or_none(group[col])
                row[f"{col}_std"] = float(group[col].std()) if group[col].dropna().size > 1 else None
        if "elapsed_sec" in group:
            row["elapsed_sec_mean"] = _mean_or_none(group["elapsed_sec"])
            row["elapsed_sec_total"] = float(group["elapsed_sec"].dropna().sum())
        for col in ENERGY_COLS:
            if col in group:
                row[f"{col}_mean"] = _mean_or_none(group[col])
                row[f"{col}_total"] = float(group[col].dropna().sum()) if group[col].notna().any() else None
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).merge(configs_df, on="config_id", how="left")

    leading = ["config_id", "config_name", "vary_step", "axis", "test_case_id", "prompt", "difficulty"]
    detail = detail[[c for c in leading if c in detail.columns] + [c for c in detail.columns if c not in leading]]

    summary_leading = ["config_id", "config_name", "vary_step", "axis", "description", "n_test_cases"]
    summary = summary[[c for c in summary_leading if c in summary.columns] + [c for c in summary.columns if c not in summary_leading]]

    detail.to_csv(save_dir / "detail.csv", index=False)
    summary.to_csv(save_dir / "summary.csv", index=False)
    _write_thesis_summary(summary, save_dir / "thesis_summary.csv")
    _write_plots(summary, save_dir / "plots")
    return detail, summary


def _quality_score(row: pd.Series) -> Optional[float]:
    vals = [row.get(col) for col in ("csv_iou_mean", "text_score_mean", "vis_score_mean")]
    vals = [float(v) for v in vals if pd.notna(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _write_thesis_summary(summary: pd.DataFrame, path: Path) -> None:
    thesis = summary.copy()
    thesis["quality_mean"] = thesis.apply(_quality_score, axis=1)

    baseline_by_step = (
        thesis[thesis["axis"].eq("baseline")]
        .set_index("vary_step")
        .to_dict(orient="index")
    )

    for metric in ("quality_mean", "csv_iou_mean", "text_score_mean", "vis_score_mean", "elapsed_sec_mean", "energy_consumed_kwh_mean"):
        delta_col = f"delta_{metric}"
        values = []
        for _, row in thesis.iterrows():
            baseline = baseline_by_step.get(row.get("vary_step"), {})
            base_val = baseline.get(metric)
            val = row.get(metric)
            values.append(float(val) - float(base_val) if pd.notna(val) and pd.notna(base_val) else None)
        thesis[delta_col] = values

    thesis.to_csv(path, index=False)


def _write_plots(summary: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] Skipping plot generation: {exc}")
        return

    plot_df = summary.copy()
    plot_df["quality_mean"] = plot_df.apply(_quality_score, axis=1)
    model_label = "Manifest DOE"
    if "model" in plot_df and plot_df["model"].notna().any():
        models = sorted({str(m) for m in plot_df["model"].dropna().unique()})
        if len(models) == 1:
            model_label = models[0]
    labels = plot_df["config_name"].tolist()
    x = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ("csv_iou_mean", "text_score_mean", "vis_score_mean"):
        if col in plot_df:
            ax.plot(x, plot_df[col], marker="o", label=col.replace("_mean", ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title(f"{model_label}: Quality by Config")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_by_config.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(x, plot_df.get("elapsed_sec_mean"), color="#4c78a8", alpha=0.75, label="elapsed_sec_mean")
    ax1.set_ylabel("Mean elapsed seconds")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax2 = ax1.twinx()
    if "energy_consumed_kwh_mean" in plot_df:
        ax2.plot(x, plot_df["energy_consumed_kwh_mean"], color="#f58518", marker="o", label="energy_consumed_kwh_mean")
    ax2.set_ylabel("Mean energy kWh")
    ax1.set_title(f"{model_label}: Latency and Energy by Config")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_energy_by_config.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    energy_col = "energy_consumed_kwh_mean"
    if energy_col in plot_df and plot_df[energy_col].notna().any():
        ax.scatter(plot_df[energy_col], plot_df["quality_mean"])
        for _, row in plot_df.iterrows():
            if pd.notna(row.get(energy_col)) and pd.notna(row.get("quality_mean")):
                ax.annotate(row["config_name"], (row[energy_col], row["quality_mean"]), fontsize=7)
    ax.set_xlabel("Mean energy kWh")
    ax.set_ylabel("Mean quality score")
    ax.set_title(f"{model_label}: Quality vs Energy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_vs_energy.png", dpi=160)
    plt.close(fig)


def _iter_configs(manifest: Dict[str, Any], max_configs: Optional[int]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    configs = manifest["configs"]
    if max_configs is not None:
        configs = configs[:max_configs]
    for idx, spec in enumerate(configs):
        if "name" not in spec:
            raise ValueError(f"Config at index {idx} is missing required field 'name'")
        yield idx, spec


def run_manifest_benchmark(
    dataset_path: str,
    manifest_path: str,
    *,
    provider: str,
    model: str,
    judge_provider: Optional[str],
    judge_model: Optional[str],
    ollama_url: str,
    save_dir: str,
    enable_codecarbon: bool,
    max_configs: Optional[int],
    max_prompts: Optional[int],
    resume: bool,
) -> pd.DataFrame:
    manifest = _load_manifest(manifest_path)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    effective_judge_provider = judge_provider or provider
    effective_judge_model = judge_model or model

    records: List[Dict[str, Any]] = []
    for config_id, spec in _iter_configs(manifest, max_configs):
        cfg = _build_config(
            manifest,
            spec,
            provider=provider,
            model=model,
            ollama_url=ollama_url,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        record = _config_record(config_id, cfg, spec)
        record["judge_provider"] = effective_judge_provider
        record["judge_model"] = effective_judge_model
        records.append(record)

    with open(save_path / "configs_sampled.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    for config_id, spec in _iter_configs(manifest, max_configs):
        config_dir = save_path / f"config_{config_id:04d}"
        result_csv = config_dir / "benchmark_results.csv"
        if resume and result_csv.exists():
            print(f"[config {config_id:04d}] Skipping existing result: {result_csv}")
            continue

        cfg = _build_config(
            manifest,
            spec,
            provider=provider,
            model=model,
            ollama_url=ollama_url,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(records[config_id], f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"CONFIG {config_id + 1}/{len(records)}: {spec['name']}")
        print(f"vary_step={spec.get('vary_step')} axis={spec.get('axis')}")
        print(f"{'=' * 70}")

        start = time.perf_counter()
        df = run_benchmark(
            dataset_path,
            agent_config=cfg,
            judge_provider=effective_judge_provider,
            judge_model=effective_judge_model,
            save_dir=str(config_dir),
            save_execution_artifacts=True,
            enable_codecarbon=enable_codecarbon,
            max_prompts=max_prompts,
            config_label=spec["name"],
        )
        df.to_csv(result_csv, index=False)
        print(f"[config {config_id:04d}] Done in {time.perf_counter() - start:.1f}s")

    _, summary = _aggregate_results(save_path, records)
    print(f"\nResults saved to {save_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic manifest benchmark configs")
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("manifest", help="Path to deterministic DOE YAML manifest")
    parser.add_argument("--provider", default="ollama", help="Agent provider")
    parser.add_argument("--model", default="gemma4:31b", help="Agent model")
    parser.add_argument("--judge-provider", default=None, help="Recorded judge provider; run_benchmark uses agent provider")
    parser.add_argument("--judge-model", default=None, help="Recorded judge model; run_benchmark uses agent model")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--save-dir", default="runs/gemma4_thesis_doe_12h")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-codecarbon", action="store_true", help="Disable CodeCarbon; enabled by default")

    args = parser.parse_args()
    run_manifest_benchmark(
        args.dataset,
        args.manifest,
        provider=args.provider,
        model=args.model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        ollama_url=args.ollama_url,
        save_dir=args.save_dir,
        enable_codecarbon=not args.no_codecarbon,
        max_configs=args.max_configs,
        max_prompts=args.max_prompts,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
