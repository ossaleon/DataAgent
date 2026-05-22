import json

from Agent.config import AgentConfig
from evaluation import run_benchmark as benchmark_module


def _fake_eval_fn(*_args, **_kwargs):
    return 1.0


def test_run_benchmark_splits_gt_and_no_gt_judges(monkeypatch, tmp_path):
    dataset_path = tmp_path / "benchmark.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "prompt": "Show revenue by region as a bar chart",
                    "gt_data": "region,revenue\nNorth,10\n",
                    "gt_analysis": "North revenue is 10.",
                    "gt_chart_config": {"chart_type": "bar", "x_axis": "region", "y_axis": "revenue"},
                    "gt_chart_code": "plt.bar(data['region'], data['revenue'])",
                }
            ]
        ),
        encoding="utf-8",
    )

    factory_calls = {}

    def record_factory(name):
        def factory(*_args, **kwargs):
            factory_calls[name] = kwargs
            return _fake_eval_fn

        return factory

    monkeypatch.setattr(benchmark_module, "_load_schema", lambda _data_dir: None)
    monkeypatch.setattr(benchmark_module, "make_csv_evaluator_gt", lambda **_kwargs: _fake_eval_fn)
    monkeypatch.setattr(benchmark_module, "make_csv_evaluator_no_gt", lambda: _fake_eval_fn)
    monkeypatch.setattr(benchmark_module, "make_text_evaluator_gt", record_factory("text_gt"))
    monkeypatch.setattr(benchmark_module, "make_text_evaluator_no_gt", record_factory("text_no_gt"))
    monkeypatch.setattr(benchmark_module, "make_vis_evaluator_gt", record_factory("vis_gt"))
    monkeypatch.setattr(benchmark_module, "make_vis_evaluator_no_gt", record_factory("vis_no_gt"))

    def fake_run_single(*_args, **_kwargs):
        return {
            "_gt_scores_per_step": {
                "lookup_sales_data": {"gt_score": 1.0},
                "analyzing_data": {"gt_score": 1.0},
                "create_visualization": {"gt_score": 1.0},
            },
            "_step_eval_scores": {},
            "_step_timings_sec": {},
            "_step_llm_timings_sec": {},
            "_step_llm_energy": {},
            "_energy": {},
            "_step_errors": {},
            "_cot_diagnostics_per_step": {
                "lookup_sales_data": {
                    "requested_iterations": 4,
                    "attempted_iterations": 2,
                    "executed_iterations": 2,
                    "early_stop": True,
                    "stop_reason": "converged",
                    "final_similarity": 0.99,
                    "similarities": [0.99],
                },
            },
            "_total_run_time_sec": 0.0,
            "sql_query": "SELECT 1",
        }

    monkeypatch.setattr(benchmark_module, "run_single", fake_run_single)

    config = AgentConfig(provider="ollama", model="gemma4:e4b")
    result_df = benchmark_module.run_benchmark(
        str(dataset_path),
        agent_config=config,
        gt_judge_provider="openai",
        gt_judge_model="gpt-5.4",
        save_dir=str(tmp_path / "results"),
    )

    assert factory_calls["text_gt"]["provider"] == "openai"
    assert factory_calls["text_gt"]["judge_model"] == "gpt-5.4"
    assert factory_calls["vis_gt"]["provider"] == "openai"
    assert factory_calls["vis_gt"]["judge_model"] == "gpt-5.4"

    assert factory_calls["text_no_gt"]["provider"] == "ollama"
    assert factory_calls["text_no_gt"]["judge_model"] == "gemma4:e4b"
    assert factory_calls["vis_no_gt"]["provider"] == "ollama"
    assert factory_calls["vis_no_gt"]["judge_model"] == "gemma4:e4b"

    row = result_df.iloc[0]
    assert row["lookup_cot_requested_iterations"] == 4
    assert row["lookup_cot_executed_iterations"] == 2
    assert row["lookup_cot_early_stop"]
    assert row["lookup_cot_stop_reason"] == "converged"
