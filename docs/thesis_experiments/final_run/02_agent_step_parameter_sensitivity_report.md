# Test 02 Final Report: Agent-Step Parameter Sensitivity

## Data Source

Results analyzed from:

```text
/home/oss/Downloads/02v2_agent_step_parameter_sensitivity
```

Completeness check:

| Model | Repeat | Expected configs | Completed configs | Rows | Status |
| --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | rep01 | 153 | 153 | 2295 | included |
| `gemma4:26b` | rep02 | 153 | 153 | 2295 | included |
| `mistral-small3.2:24b` | rep01 | 153 | 153 | 2295 | included |
| `mistral-small3.2:24b` | rep02 | 153 | 153 | 2295 | included |
| `gemma4:e4b` | rep01 | 153 | 39 | None | excluded: missing summary.csv; missing detail.csv; 39/153 completed config CSVs |
| `gemma4:e4b` | rep02 | 153 | 14 | None | excluded: missing summary.csv; missing detail.csv; 14/153 completed config CSVs |
| `gemma4:e4b` | rep03 | None | 0 | None | excluded: missing/invalid configs_sampled.json; missing summary.csv; missing detail.csv |

The incomplete `gemma4:e4b` folders are excluded from the numeric analysis.

## Method Notes

This test varies one agent step at a time while the other two stay fixed. Each phase tests `temperature`, `top_p`, `top_k`, `repeat_penalty`, and `repeat_last_n`. Expected missing GT score slots are counted as `0`, so failures remain part of accuracy.

## Executive Findings

- `mistral-small3.2:24b` remains the efficient control, with lower time and energy than `gemma4:26b`.
- `gemma4:26b` is more sensitive: it has larger positive movements but also much larger losses.
- Repeat energy variance is small relative to the model gap; quality variance is visible enough that close config rankings should remain tentative.

## Overall Resource Use

| Model | Repeats | Rows | Mean quality | Mean sec / prompt | Mean kWh / prompt | Mean completion |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | 2 | 4,590 | 0.8325 | 189.0 | 0.01189 | 98.7% |
| `mistral-small3.2:24b` | 2 | 4,590 | 0.8783 | 67.2 | 0.00416 | 95.2% |

## Repeat Stability

| Model | Matched configs | Mean quality rep01 | Mean quality rep02 | Abs mean quality diff | Median quality diff | P90 quality diff | Mean kWh rep01 | Mean kWh rep02 | Abs mean kWh diff | Abs mean sec diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | 153 | 0.8332 | 0.8318 | 0.0408 | 0.0319 | 0.0858 | 0.01168 | 0.01210 | 0.00139 | 22.4 |
| `mistral-small3.2:24b` | 153 | 0.8781 | 0.8786 | 0.0156 | 0.0083 | 0.0473 | 0.00413 | 0.00420 | 0.00010 | 2.6 |

A small energy difference means more repeats are not needed for the broad energy conclusion. Accuracy is less stable, especially for `gemma4:26b`; large effects are meaningful, but small differences should not decide final parameters alone.

## Sensitivity Range

| Model | Step | Baseline quality | Best delta | Worst delta | Range | Best config | Worst config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | Analysis | 0.8675 | +0.0065 | -0.1111 | 0.1176 | `analysis_top_k_24` | `analysis_top_p_0p75` |
| `gemma4:26b` | Visualization | 0.8116 | +0.0684 | -0.0337 | 0.1021 | `visualization_repeat_penalty_1` | `visualization_top_p_0p8` |
| `gemma4:26b` | Lookup | 0.8480 | +0.0319 | -0.1229 | 0.1548 | `lookup_repeat_last_n_96` | `lookup_top_k_56` |
| `mistral-small3.2:24b` | Analysis | 0.8861 | +0.0069 | -0.0986 | 0.1056 | `analysis_top_k_56` | `analysis_repeat_penalty_1p2` |
| `mistral-small3.2:24b` | Visualization | 0.8889 | +0.0014 | -0.0401 | 0.0415 | `visualization_repeat_last_n_56` | `visualization_temperature_0p1` |
| `mistral-small3.2:24b` | Lookup | 0.8861 | +0.0042 | -0.0362 | 0.0404 | `lookup_repeat_penalty_1p3` | `lookup_repeat_last_n_80` |

## Direct Agent Metrics

| Model | Step | Metric | Baseline | Best | Best config | Worst | Worst config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | Analysis | `text_score` | 0.9750 | 0.9750 | `analysis_repeat_penalty_1p3` | 0.8375 | `analysis_top_p_0p75` |
| `gemma4:26b` | Visualization | `vis_score` | 0.5677 | 0.6198 | `visualization_repeat_penalty_1` | 0.3880 | `visualization_top_p_0p85` |
| `gemma4:26b` | Lookup | `csv_iou` | 0.9441 | 0.9982 | `lookup_repeat_penalty_1p3` | 0.8982 | `lookup_repeat_last_n_256` |
| `mistral-small3.2:24b` | Analysis | `text_score` | 0.8583 | 0.8667 | `analysis_top_k_56` | 0.6875 | `analysis_repeat_penalty_1p2` |
| `mistral-small3.2:24b` | Visualization | `vis_score` | 0.8542 | 0.8542 | `visualization_temperature_0p175` | 0.8229 | `visualization_top_k_32` |
| `mistral-small3.2:24b` | Lookup | `csv_iou` | 0.9333 | 0.9333 | `lookup_temperature_0p03` | 0.9126 | `lookup_top_k_32` |

## Figures

![quality heatmap](plots/test02v2_quality_delta_heatmap.png)

![metric heatmap](plots/test02v2_step_metric_delta_heatmap.png)

![quality-energy](plots/test02v2_quality_energy_delta_pareto.png)

![energy by step](plots/test02v2_resource_delta_by_step.png)

![sensitivity range](plots/test02v2_sensitivity_range_by_step.png)

![repeat stability](plots/test02v2_repeat_stability.png)

## Best And Worst Static Changes

Top quality improvements:

| Model | Step | Config | Delta quality | Delta metric | Delta kWh | Delta sec |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | Visualization | `visualization_repeat_penalty_1` | +0.0684 | +0.0521 | 0.00128 | +23.5 |
| `gemma4:26b` | Visualization | `visualization_top_k_20` | +0.0663 | +0.0443 | 0.00107 | +21.0 |
| `gemma4:26b` | Visualization | `visualization_top_k_48` | +0.0593 | +0.0234 | 0.00083 | +17.8 |
| `gemma4:26b` | Visualization | `visualization_top_k_128` | +0.0559 | +0.0052 | 0.00070 | +15.6 |
| `gemma4:26b` | Visualization | `visualization_repeat_last_n_256` | +0.0517 | +0.0156 | 0.00143 | +26.6 |
| `gemma4:26b` | Visualization | `visualization_repeat_penalty_1p03` | +0.0517 | +0.0521 | 0.00057 | +12.9 |
| `gemma4:26b` | Visualization | `visualization_temperature_1p3` | +0.0489 | -0.0052 | 0.00086 | +19.2 |
| `gemma4:26b` | Visualization | `visualization_top_k_16` | +0.0489 | -0.0208 | 0.00078 | +16.4 |
| `gemma4:26b` | Visualization | `visualization_temperature_1p1` | +0.0475 | -0.0365 | 0.00118 | +25.5 |
| `gemma4:26b` | Visualization | `visualization_temperature_0p95` | +0.0468 | +0.0182 | 0.00230 | +39.4 |

Largest quality losses:

| Model | Step | Config | Delta quality | Delta metric | Delta kWh | Delta sec |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4:26b` | Lookup | `lookup_top_k_56` | -0.1229 | 0.0000 | 0.00157 | +34.4 |
| `gemma4:26b` | Analysis | `analysis_top_p_0p75` | -0.1111 | -0.1375 | -0.00000 | -1.5 |
| `mistral-small3.2:24b` | Analysis | `analysis_repeat_penalty_1p2` | -0.0986 | -0.1708 | 0.00337 | +90.4 |
| `gemma4:26b` | Analysis | `analysis_top_p_0p88` | -0.0896 | -0.1125 | -0.00028 | -5.6 |
| `gemma4:26b` | Analysis | `analysis_repeat_last_n_96` | -0.0868 | -0.0750 | -0.00069 | -18.7 |
| `gemma4:26b` | Analysis | `analysis_repeat_last_n_16` | -0.0868 | -0.1042 | -0.00057 | -15.8 |
| `gemma4:26b` | Lookup | `lookup_top_k_48` | -0.0834 | -0.0432 | 0.00471 | +72.1 |
| `gemma4:26b` | Analysis | `analysis_temperature_0p7` | -0.0791 | -0.0833 | -0.00014 | -2.3 |

## Prompt Difficulty Analysis

This difficulty view keeps the Test 02 logic intact: each row compares configurations only against the baseline for the same model, same agent step, and same prompt difficulty. The incomplete `gemma4:e4b` folders remain excluded.

![Test 02 difficulty baseline quality](plots/test02v2_difficulty_baseline_quality.png)

![Test 02 best parameter delta by difficulty](plots/test02v2_difficulty_best_delta_heatmap.png)

Largest difficulty-specific gains:

| Model | Step | Difficulty | Baseline quality | Best config | Best delta |
| --- | --- | --- | --- | --- | --- |
| gemma4:26b | Visualization | 1 | 0.622 | visualization_repeat_penalty_1 | +0.246 |
| gemma4:26b | Lookup | 1 | 0.715 | lookup_repeat_last_n_128 | +0.194 |
| gemma4:26b | Visualization | 3 | 0.803 | visualization_temperature_0p95 | +0.101 |
| gemma4:26b | Analysis | 3 | 0.832 | analysis_top_p_1 | +0.069 |
| gemma4:26b | Visualization | 2 | 0.870 | visualization_top_k_16 | +0.057 |
| gemma4:26b | Analysis | 4 | 0.917 | analysis_repeat_last_n_56 | +0.042 |
| gemma4:26b | Lookup | 4 | 0.924 | lookup_temperature_0p95 | +0.042 |
| gemma4:26b | Analysis | 2 | 0.885 | analysis_repeat_last_n_56 | +0.036 |

Largest difficulty-specific losses:

| Model | Step | Difficulty | Baseline quality | Worst config | Worst delta |
| --- | --- | --- | --- | --- | --- |
| gemma4:26b | Analysis | 4 | 0.917 | analysis_repeat_last_n_96 | -0.312 |
| gemma4:26b | Visualization | 4 | 0.938 | visualization_repeat_last_n_48 | -0.250 |
| mistral-small3.2:24b | Analysis | 3 | 0.917 | analysis_repeat_penalty_1p2 | -0.242 |
| gemma4:26b | Analysis | 1 | 0.854 | analysis_temperature_1p2 | -0.229 |
| gemma4:26b | Lookup | 4 | 0.924 | lookup_temperature_0p8 | -0.215 |
| gemma4:26b | Lookup | 3 | 0.857 | lookup_top_k_56 | -0.215 |
| mistral-small3.2:24b | Analysis | 4 | 0.653 | analysis_repeat_last_n_96 | -0.169 |
| mistral-small3.2:24b | Lookup | 4 | 0.646 | lookup_repeat_last_n_80 | -0.146 |

Interpretation: parameter sensitivity is not uniform across task difficulty. `gemma4:26b` gets the largest upside on visualization-heavy and harder slices, but it also has the largest negative swings when a parameter setting destabilizes an agent. `mistral-small3.2:24b` is flatter: this is good for robustness, but it also means Test 02 finds fewer large accuracy gains from simple one-parameter changes.

## Thesis Implications

The two complete models confirm that parameter sensitivity is agent-specific. Mistral is the stronger efficient baseline, while Gemma 26B needs careful step-level tuning and should not be summarized by a single global sampling setting.
