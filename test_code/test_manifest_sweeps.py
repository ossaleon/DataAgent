from pathlib import Path

import pandas as pd

from evaluation.run_manifest_benchmark import _build_config, _load_manifest, _write_plots


def test_final_test02_manifest_expands_static_ofat_grid():
    path = Path("evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml")

    manifest = _load_manifest(str(path))
    configs = manifest["configs"]

    assert len(configs) == 153
    assert sum(config["axis"] == "baseline" for config in configs) == 3
    assert {config["axis"] for config in configs} == {
        "baseline",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "repeat_last_n",
    }


def test_final_test02_repeat_values_materialize_into_step_config():
    path = Path("evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml")
    manifest = _load_manifest(str(path))
    config_spec = next(config for config in manifest["configs"] if config["name"] == "analysis_repeat_penalty_1p2")

    config = _build_config(
        manifest,
        config_spec,
        provider="ollama",
        model="gemma4:e4b",
        ollama_url="http://localhost:11434",
        openai_api_key=None,
    )

    assert config.analyzing_data.repeat_penalty == 1.2
    assert config.analyzing_data.repeat_last_n == 64
    assert config.lookup_sales_data.repeat_penalty == 1.1


def test_manifest_plots_allow_failed_summary_without_elapsed_time(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "config_name": "failed_config",
                "elapsed_sec_mean": None,
                "energy_consumed_kwh_mean": None,
            }
        ]
    )

    _write_plots(summary, tmp_path / "plots")

    assert (tmp_path / "plots" / "latency_energy_by_config.png").exists()
