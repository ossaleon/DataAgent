# Test 05 Report: Final Pareto Confirmation

## Data Source

Results analyzed from:

```text
/home/oss/Downloads/05_final_pareto_confirmation/
```

Completed result folders:

```text
gemma4_e4b/rep
gemma4_26b/rep
mistral_small32_24b/rep
```

Each completed model contains:

```text
6 configs x 20 prompts = 120 benchmark rows
```

Coverage limitation:

```text
Only one repeat per model is present.
```

The original Test 05 plan asked for three repeats. This report should therefore be read as the first final-confirmation pass, not yet as the final statistical claim.

## Method Notes

Test 05 evaluates full-agent Pareto candidates on the full 20-prompt benchmark. Unlike Tests 02-04, the configurations are not pure single-step ablations. They are final candidate policies assembled from the earlier experiments.

Two quality metrics are important:

```text
quality_mean         average of metric means: csv_iou, text_score, vis_score
prompt_quality_mean  average per-prompt quality over available scores
```

`prompt_quality_mean` is important because `quality_mean` can hide failures where an early SQL error prevents downstream text or visualization scores from being produced.

Completion metrics:

```text
completion_rate       completed score slots / expected score slots
full_completion_rate  prompts where all expected scores were produced
```

For the full benchmark, each config has:

```text
20 data scores
20 text scores
14 visualization scores
54 expected score slots
```

## Executive Findings

The final Pareto picture is clear:

- `gemma4:26b` with `best_step_static` gives the highest absolute quality: `quality_mean=0.9619`, `prompt_quality=0.9694`, full completion.
- `gemma4:e4b` with `efficient_static` is the best high-quality Pareto compromise: `quality_mean=0.9595`, only `0.0024` below the best config, with much lower time and energy than Gemma 26B.
- `mistral-small3.2:24b` baseline remains the best low-cost fully reliable baseline: `quality_mean=0.9379`, `prompt_quality=0.9265`, full completion, lowest time among full-completion configs.
- Several combined or low-token candidates did not generalize from the 10-case exploratory set to the full benchmark.

Main thesis conclusion:

```text
The best absolute model is the larger thinking MoE, but the best quality/energy compromise is the small thinking Gemma E4B with targeted static tuning. The larger non-thinking Mistral remains the strongest efficient baseline, but it is no longer the top Pareto point once tuned Gemma E4B is included.
```

## Overall Ranking

Sorted by official `quality_mean`:

| Rank | Model | Config | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `gemma4:26b` | `best_step_static` | 0.9619 | 0.9694 | 1.000 | 1.00 | 230.1 | 0.01515 |
| 2 | `gemma4:e4b` | `efficient_static` | 0.9595 | 0.9515 | 1.000 | 1.00 | 154.6 | 0.01049 |
| 3 | `gemma4:26b` | `visualization_bon_top_k_n2` | 0.9586 | 0.9665 | 1.000 | 1.00 | 324.7 | 0.01104 |
| 4 | `gemma4:e4b` | `baseline` | 0.9485 | 0.9079 | 0.944 | 0.90 | 146.8 | 0.00985 |
| 5 | `gemma4:e4b` | `combined_best_candidate` | 0.9466 | 0.9100 | 0.963 | 0.95 | 146.5 | 0.00827 |
| 6 | `gemma4:e4b` | `lookup_cot_n2` | 0.9414 | 0.9405 | 1.000 | 1.00 | 170.0 | 0.01097 |
| 7 | `mistral-small3.2:24b` | `baseline` | 0.9379 | 0.9265 | 1.000 | 1.00 | 114.3 | 0.00773 |

Within `0.02` of the best official quality, the candidates are:

| Model | Config | Quality | Prompt quality | Completion | Sec / prompt | kWh / prompt |
|---|---|---:|---:|---:|---:|---:|
| `gemma4:e4b` | `combined_best_candidate` | 0.9466 | 0.9100 | 0.963 | 146.5 | 0.00827 |
| `gemma4:e4b` | `baseline` | 0.9485 | 0.9079 | 0.944 | 146.8 | 0.00985 |
| `gemma4:e4b` | `efficient_static` | 0.9595 | 0.9515 | 1.000 | 154.6 | 0.01049 |
| `gemma4:26b` | `visualization_bon_top_k_n2` | 0.9586 | 0.9665 | 1.000 | 324.7 | 0.01104 |
| `gemma4:26b` | `best_step_static` | 0.9619 | 0.9694 | 1.000 | 230.1 | 0.01515 |

If full completion is required, `gemma4:e4b efficient_static` is the lowest-energy config within `0.02` of the best official quality.

## Pareto Frontier

For official quality vs energy, the non-dominated points are:

| Model | Config | Quality | Prompt quality | Completion | kWh / prompt |
|---|---|---:|---:|---:|---:|
| `mistral-small3.2:24b` | `analysis_top_k_high_low_tokens` | 0.9092 | 0.8729 | 0.963 | 0.00722 |
| `mistral-small3.2:24b` | `max_tokens_adjusted` | 0.9123 | 0.8741 | 0.963 | 0.00726 |
| `mistral-small3.2:24b` | `efficient_static` | 0.9355 | 0.8963 | 0.963 | 0.00734 |
| `mistral-small3.2:24b` | `baseline` | 0.9379 | 0.9265 | 1.000 | 0.00773 |
| `gemma4:e4b` | `combined_best_candidate` | 0.9466 | 0.9100 | 0.963 | 0.00827 |
| `gemma4:e4b` | `baseline` | 0.9485 | 0.9079 | 0.944 | 0.00985 |
| `gemma4:e4b` | `efficient_static` | 0.9595 | 0.9515 | 1.000 | 0.01049 |
| `gemma4:26b` | `best_step_static` | 0.9619 | 0.9694 | 1.000 | 0.01515 |

For prompt quality vs energy, the important full-completion frontier is:

| Model | Config | Prompt quality | Quality | Sec / prompt | kWh / prompt |
|---|---|---:|---:|---:|---:|
| `mistral-small3.2:24b` | `baseline` | 0.9265 | 0.9379 | 114.3 | 0.00773 |
| `gemma4:e4b` | `efficient_static` | 0.9515 | 0.9595 | 154.6 | 0.01049 |
| `gemma4:26b` | `visualization_bon_top_k_n2` | 0.9665 | 0.9586 | 324.7 | 0.01104 |
| `gemma4:26b` | `best_step_static` | 0.9694 | 0.9619 | 230.1 | 0.01515 |

The most thesis-useful Pareto comparison is:

```text
Mistral baseline:
  best low-cost fully reliable baseline

Gemma E4B efficient_static:
  best high-quality energy-aware candidate

Gemma 26B best_step_static:
  best absolute accuracy candidate
```

## Model-Level Findings

### Gemma E4B

| Config | Quality | Prompt quality | Completion | Sec | kWh | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 0.9485 | 0.9079 | 0.944 | 146.8 | 0.00985 | High official quality, but completion gaps remain. |
| `efficient_static` | 0.9595 | 0.9515 | 1.000 | 154.6 | 0.01049 | Best Gemma E4B config and strongest high-quality Pareto point. |
| `best_step_static` | 0.8658 | 0.7936 | 0.926 | 136.9 | 0.00920 | Does not generalize; severe SQL failures. |
| `max_tokens_adjusted` | 0.8868 | 0.7879 | 0.889 | 128.9 | 0.00866 | Token-only adjustment hurts reliability. |
| `lookup_cot_n2` | 0.9414 | 0.9405 | 1.000 | 170.0 | 0.01097 | Repairs completion but not the best quality/energy point. |
| `combined_best_candidate` | 0.9466 | 0.9100 | 0.963 | 146.5 | 0.00827 | Efficient, but not fully reliable. |

Best Gemma E4B recommendation:

```text
efficient_static
```

It improves over baseline:

```text
quality_mean:        +0.0110
prompt_quality_mean: +0.0436
completion_rate:     +0.0556
elapsed_sec:         +7.8s
energy:              +0.00064 kWh
```

The important qualitative result is that `efficient_static` fixes the hard-prompt reliability issue. On difficulty-4 prompts:

```text
baseline difficulty-4 prompt quality:         0.719
efficient_static difficulty-4 prompt quality: 0.958
lookup_cot_n2 difficulty-4 prompt quality:    0.933
```

This confirms the thesis idea that the small thinking model can become competitive when tuned at the right agent steps. But it also rejects the naive combined candidate: combining several good-looking exploratory settings did not automatically improve reliability.

### Gemma 26B MoE

| Config | Quality | Prompt quality | Completion | Sec | kWh | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 0.8900 | 0.8853 | 0.963 | 218.9 | 0.01457 | Strong text/SQL, weak visualization. |
| `efficient_static` | 0.9330 | 0.9473 | 1.000 | 222.1 | 0.01465 | Static visualization repair helps strongly. |
| `best_step_static` | 0.9619 | 0.9694 | 1.000 | 230.1 | 0.01515 | Best absolute config. |
| `max_tokens_adjusted` | 0.8715 | 0.8413 | 0.963 | 433.5 | 0.02003 | Bad and very slow; reject. |
| `visualization_bon_top_k_n2` | 0.9586 | 0.9665 | 1.000 | 324.7 | 0.01104 | High quality, but slow. |
| `combined_best_candidate` | 0.9257 | 0.9030 | 0.963 | 222.3 | 0.00757 | Efficient-looking but not reliable. |

Best Gemma 26B recommendation:

```text
best_step_static
```

It improves over baseline:

```text
quality_mean:        +0.0718
prompt_quality_mean: +0.0842
completion_rate:     +0.0370
elapsed_sec:         +11.2s
energy:              +0.00058 kWh
```

This is a very strong result. It confirms the earlier diagnosis that Gemma 26B's main bottleneck was visualization. A static visualization fix with `top_k_high` and `max_tokens=10000` is enough to make it the best absolute model.

`visualization_bon_top_k_n2` is also strong, but it is dominated for final recommendation:

```text
best_step_static:
  quality_mean 0.9619
  prompt_quality 0.9694
  elapsed 230.1s

visualization_bon_top_k_n2:
  quality_mean 0.9586
  prompt_quality 0.9665
  elapsed 324.7s
```

The compute-expansion visualization candidate is accurate, but the static visualization fix is faster and slightly better.

### Mistral Small 3.2 24B

| Config | Quality | Prompt quality | Completion | Sec | kWh | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 0.9379 | 0.9265 | 1.000 | 114.3 | 0.00773 | Best Mistral config. |
| `efficient_static` | 0.9355 | 0.8963 | 0.963 | 108.5 | 0.00734 | Slightly lower energy, but completion loss. |
| `best_step_static` | 0.9307 | 0.9239 | 1.000 | 114.3 | 0.00776 | Full completion, but lower quality than baseline. |
| `max_tokens_adjusted` | 0.9123 | 0.8741 | 0.963 | 108.3 | 0.00726 | Lower caps hurt hard prompts. |
| `analysis_top_k_high_low_tokens` | 0.9092 | 0.8729 | 0.963 | 107.7 | 0.00722 | Lowest energy, but too much quality loss. |
| `combined_best_candidate` | 0.9163 | 0.9083 | 1.000 | 113.8 | 0.00767 | Full completion, but lower quality than baseline. |

Best Mistral recommendation:

```text
baseline
```

The final run confirms that Mistral is already near its local optimum. Attempts to reduce token caps or combine cheap static changes usually save a small amount of time/energy but reduce quality or completion.

The main hard-prompt problem is case 4:

```text
Compare average monthly revenue between store regions for 2022 and 2023
```

Low-token Mistral variants fail this case, while baseline completes all expected scores.

## Difficulty-Level Findings

The most revealing difficulty split is difficulty 4.

| Model | Best relevant config | Difficulty-4 prompt quality | Baseline difficulty-4 prompt quality | Interpretation |
|---|---|---:|---:|---|
| `gemma4:e4b` | `efficient_static` | 0.958 | 0.719 | Tuning repairs hard-prompt fragility. |
| `gemma4:26b` | `best_step_static` | 0.985 | 0.954 | Already strong; visualization repair still helps. |
| `mistral-small3.2:24b` | `baseline` | 0.953 | 0.953 | Baseline is best; low-token variants hurt. |

This is a strong thesis result:

```text
The small thinking model was not intrinsically weak. Its baseline failed on hard SQL/visualization interactions, and targeted static tuning largely repaired that weakness.
```

## What Test 05 Confirms

Confirmed from earlier tests:

- Visualization is the main repair target for `gemma4:26b`.
- Mistral is stable and efficient at baseline.
- Completion-adjusted metrics are necessary; official quality alone can overstate configs with missing downstream scores.
- `n=2` compute expansion can help, but static settings are often better.

Partially confirmed:

- `lookup_cot_n2` improves Gemma E4B reliability, but it is not the best final Pareto choice.
- Gemma E4B can compete strongly with larger models when tuned, but not every exploratory improvement generalizes.

Rejected:

- Gemma E4B `best_step_static` does not generalize to the full benchmark.
- Gemma E4B `max_tokens_adjusted` alone is not safe.
- Gemma 26B `max_tokens_adjusted` with reduced lookup/analysis caps is harmful and very slow.
- Mistral low-token candidates are not worth the completion loss.
- Gemma 26B visualization Best-of-N is not necessary when the static visualization fix is available.

## Final Recommendations

For thesis tables, report three primary final configs:

| Role | Model | Config | Why |
|---|---|---|---|
| Best low-cost baseline | `mistral-small3.2:24b` | `baseline` | Fastest fully complete strong baseline. |
| Best quality/energy compromise | `gemma4:e4b` | `efficient_static` | Near-best quality with much lower cost than Gemma 26B. |
| Best absolute accuracy | `gemma4:26b` | `best_step_static` | Highest quality and prompt quality. |

For final Pareto discussion:

```text
Mistral baseline:
  quality=0.9379
  prompt_quality=0.9265
  energy=0.00773 kWh
  elapsed=114.3s

Gemma E4B efficient_static:
  quality=0.9595
  prompt_quality=0.9515
  energy=0.01049 kWh
  elapsed=154.6s

Gemma 26B best_step_static:
  quality=0.9619
  prompt_quality=0.9694
  energy=0.01515 kWh
  elapsed=230.1s
```

Interpretation:

```text
Gemma 26B buys the best prompt-level correctness at the highest cost.
Gemma E4B recovers almost all official quality at much lower cost.
Mistral remains the best fast and efficient baseline, but is no longer the best high-quality Pareto point.
```

## Required Next Step

Run at least two more repeats before writing the final statistical conclusion.

The one-repeat signal is coherent and thesis-useful, but several configs are close enough that repeat variance matters:

```text
gemma4:e4b efficient_static vs gemma4:26b best_step_static
gemma4:26b best_step_static vs visualization_bon_top_k_n2
mistral baseline vs efficient_static
```

If time is limited, prioritize repeats for the three final recommended configs:

```text
mistral-small3.2:24b baseline
gemma4:e4b efficient_static
gemma4:26b best_step_static
```
