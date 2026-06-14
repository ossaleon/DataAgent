# Test 05 Final-Run Report: Pareto Confirmation

## Data Source

Results analyzed from:

```text
/home/oss/Downloads/05_final_pareto_confirmation/
/home/oss/Downloads/01_model_baseline_comparison/
```

Formal full-benchmark Pareto pool:

```text
Test 01: model baselines, rep01-rep03, full 20-prompt benchmark
Test 05: combined final candidates, rep01-rep02, full 20-prompt benchmark
GT judge: openai / gpt-5.4
No-GT judge during the agent run: local Ollama model under test
```

The report intentionally excludes the plain `rep` folders in the Test 05 download directory because those rows were judged with the local Ollama model, not GPT-5.4. It also keeps the formal Pareto frontier on the full 20-prompt benchmark only: Test 01 baselines plus the new Test 05 combined candidates. Tests 02-04 remain the design provenance for the selected candidates, but their measurements were run on the 10-prompt screening subset and are therefore not mixed into the main full-benchmark frontier.

Coverage after filtering to GPT-5.4 judged full-benchmark runs:

| Source | Model | Repeat | Configs | Rows | Prompts |
| --- | --- | --- | --- | --- | --- |
| Test 01 baseline | Gemma 26B MoE | rep01 | 1 | 20 | 20 |
| Test 01 baseline | Gemma 26B MoE | rep02 | 1 | 20 | 20 |
| Test 01 baseline | Gemma 26B MoE | rep03 | 1 | 20 | 20 |
| Test 01 baseline | Gemma E4B | rep01 | 1 | 20 | 20 |
| Test 01 baseline | Gemma E4B | rep02 | 1 | 20 | 20 |
| Test 01 baseline | Gemma E4B | rep03 | 1 | 20 | 20 |
| Test 01 baseline | Mistral Small 3.2 24B | rep01 | 1 | 20 | 20 |
| Test 01 baseline | Mistral Small 3.2 24B | rep02 | 1 | 20 | 20 |
| Test 01 baseline | Mistral Small 3.2 24B | rep03 | 1 | 20 | 20 |
| Test 05 candidate | Gemma 26B MoE | rep01 | 10 | 200 | 20 |
| Test 05 candidate | Gemma 26B MoE | rep02 | 10 | 200 | 20 |
| Test 05 candidate | Gemma E4B | rep01 | 8 | 160 | 20 |
| Test 05 candidate | Gemma E4B | rep02 | 8 | 160 | 20 |
| Test 05 candidate | Mistral Small 3.2 24B | rep01 | 8 | 160 | 20 |
| Test 05 candidate | Mistral Small 3.2 24B | rep02 | 8 | 160 | 20 |

Prompt difficulty distribution in the full benchmark:

| Difficulty | Prompts |
| --- | --- |
| 1 | 4 |
| 2 | 5 |
| 3 | 7 |
| 4 | 4 |

## Method Notes

Quality is computed strictly: if a prompt expects a data, text, or visualization score and that score is missing, the missing slot is counted as `0`. This keeps failed runs inside the accuracy estimate instead of dropping them from the mean.

Metrics used in the report:

```text
quality_mean_strict   mean of strict csv_iou, text_score, and vis_score component means
prompt_quality_mean   mean per-prompt score over the expected GT slots for that prompt
completion_rate       completed score slots / expected score slots
full_completion_rate  fraction of prompts where every expected score slot exists
```

The main Pareto objective minimizes mean energy per prompt and maximizes strict prompt quality. A configuration is non-dominated if no other configuration has both lower-or-equal energy and higher-or-equal prompt quality, with at least one strict improvement.

## Executive Findings

- The highest strict prompt quality is `lookup_cot_n2_temp_0p8_low_cost_tokens` on `Gemma E4B` with prompt quality `0.9371` at `0.00602` kWh per prompt.
- The lowest-energy point in the full-benchmark pool is `combined_static_best_efficient_tokens` on `Mistral Small 3.2 24B` at `0.00364` kWh per prompt and prompt quality `0.8434`.
- The highest-quality point on the energy/quality frontier is `lookup_cot_n2_temp_0p8_low_cost_tokens` on `Gemma E4B`, so it is the best absolute confirmed candidate among the non-dominated executions.
- `Gemma 26B MoE` improves most with `combined_static_direct`: prompt quality delta `+0.0578` and energy delta `-0.00668` kWh per prompt versus its Test 01 baseline.
- `Gemma E4B` improves most with `lookup_cot_n2_temp_0p8_low_cost_tokens`: prompt quality delta `+0.1022` and energy delta `-0.00281` kWh per prompt versus its Test 01 baseline.
- `Mistral Small 3.2 24B` improves most with `combined_cot_static`: prompt quality delta `+0.0769` and energy delta `-0.00152` kWh per prompt versus its Test 01 baseline.

Compared with the earlier local-judge Test 05 report, this GPT-5.4 judged run is more conservative. Several candidates that looked very strong under the local judge lose quality once GPT-5.4 evaluates text and visualization outputs more strictly. The Pareto conclusion therefore shifts from simple model-size ranking to a more nuanced tradeoff between robust completion, prompt difficulty, and energy.

## Overall Ranking

Sorted by strict prompt quality:

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9238 | 0.9371 | 1.000 | 1.000 | 102.2 | 0.00602 |
| Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9206 | 0.9287 | 1.000 | 1.000 | 108.9 | 0.00645 |
| Mistral Small 3.2 24B | `combined_cot_static` | Test 05 candidate | 0.9105 | 0.9206 | 0.981 | 0.975 | 71.3 | 0.00449 |
| Gemma E4B | `lookup_cot_n4_temp_0p8` | Test 05 candidate | 0.9121 | 0.9189 | 1.000 | 1.000 | 107.0 | 0.00633 |
| Gemma E4B | `lookup_cot_n4_low_cost_tokens` | Test 05 candidate | 0.9095 | 0.9176 | 1.000 | 1.000 | 108.0 | 0.00636 |
| Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | Test 05 candidate | 0.9063 | 0.9168 | 0.981 | 0.975 | 70.7 | 0.00445 |
| Gemma E4B | `lookup_cot_n2_temp_0p8` | Test 05 candidate | 0.9047 | 0.9109 | 1.000 | 1.000 | 103.2 | 0.00609 |
| Mistral Small 3.2 24B | `lookup_cot_n3_efficient_tokens` | Test 05 candidate | 0.8963 | 0.9015 | 0.981 | 0.975 | 71.6 | 0.00452 |
| Mistral Small 3.2 24B | `combined_cot_static_tokens` | Test 05 candidate | 0.8794 | 0.8902 | 0.963 | 0.950 | 70.0 | 0.00441 |
| Gemma E4B | `lookup_cot_n2_low_cost_tokens` | Test 05 candidate | 0.8710 | 0.8737 | 0.981 | 0.975 | 102.1 | 0.00603 |
| Gemma 26B MoE | `combined_static_direct` | Test 05 candidate | 0.8271 | 0.8720 | 1.000 | 1.000 | 153.3 | 0.00998 |
| Gemma E4B | `lookup_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.8650 | 0.8709 | 0.981 | 0.975 | 89.7 | 0.00530 |
| Gemma 26B MoE | `combined_visualization_topk20_tokens` | Test 05 candidate | 0.8448 | 0.8647 | 1.000 | 1.000 | 153.8 | 0.01003 |
| Gemma 26B MoE | `max_tokens_direct_safe` | Test 05 candidate | 0.8310 | 0.8610 | 1.000 | 1.000 | 157.9 | 0.01030 |
| Gemma 26B MoE | `combined_all_best_static` | Test 05 candidate | 0.8302 | 0.8605 | 1.000 | 1.000 | 156.0 | 0.01020 |
| Gemma 26B MoE | `lookup_cot_n3` | Test 05 candidate | 0.8148 | 0.8605 | 0.981 | 0.975 | 176.9 | 0.01164 |
| Gemma 26B MoE | `lookup_cot_n2` | Test 05 candidate | 0.8095 | 0.8569 | 0.981 | 0.975 | 176.5 | 0.01161 |
| Mistral Small 3.2 24B | `max_tokens_quality` | Test 05 candidate | 0.8318 | 0.8531 | 0.926 | 0.900 | 57.9 | 0.00365 |

Top positive Test 05 deltas against each model baseline:

| Model | Config | Prompt quality | Delta quality | Delta kWh | Completion | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | 0.9371 | 0.1022 | -0.00281 | 1.000 | 0.00602 |
| Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | 0.9287 | 0.0938 | -0.00238 | 1.000 | 0.00645 |
| Gemma E4B | `lookup_cot_n4_temp_0p8` | 0.9189 | 0.0841 | -0.00250 | 1.000 | 0.00633 |
| Gemma E4B | `lookup_cot_n4_low_cost_tokens` | 0.9176 | 0.0827 | -0.00247 | 1.000 | 0.00636 |
| Mistral Small 3.2 24B | `combined_cot_static` | 0.9206 | 0.0769 | -0.00152 | 0.981 | 0.00449 |
| Gemma E4B | `lookup_cot_n2_temp_0p8` | 0.9109 | 0.0760 | -0.00274 | 1.000 | 0.00609 |
| Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | 0.9168 | 0.0731 | -0.00156 | 0.981 | 0.00445 |
| Gemma 26B MoE | `combined_static_direct` | 0.8720 | 0.0578 | -0.00668 | 1.000 | 0.00998 |
| Mistral Small 3.2 24B | `lookup_cot_n3_efficient_tokens` | 0.9015 | 0.0578 | -0.00149 | 0.981 | 0.00452 |
| Gemma 26B MoE | `combined_visualization_topk20_tokens` | 0.8647 | 0.0505 | -0.00663 | 1.000 | 0.01003 |

Largest negative Test 05 deltas against each model baseline:

| Model | Config | Prompt quality | Delta quality | Delta kWh | Completion | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- |
| Mistral Small 3.2 24B | `combined_static_best` | 0.8341 | -0.0096 | -0.00236 | 0.926 | 0.00365 |
| Gemma 26B MoE | `lookup_cot_n2_combined_all_best` | 0.8069 | -0.0073 | 0.00231 | 0.981 | 0.01898 |
| Mistral Small 3.2 24B | `max_tokens_efficient` | 0.8420 | -0.0017 | -0.00232 | 0.926 | 0.00369 |
| Gemma 26B MoE | `combined_visualization_rp1_tokens` | 0.8125 | -0.0016 | -0.00687 | 0.991 | 0.00979 |
| Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | 0.8434 | -0.0003 | -0.00237 | 0.926 | 0.00364 |
| Gemma E4B | `max_tokens_low_cost` | 0.8361 | 0.0013 | -0.00371 | 0.944 | 0.00512 |

## Pareto Frontier

Strict prompt-quality frontier:

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | Test 05 candidate | 0.8273 | 0.8434 | 0.926 | 0.900 | 57.9 | 0.00364 |
| Mistral Small 3.2 24B | `max_tokens_quality` | Test 05 candidate | 0.8318 | 0.8531 | 0.926 | 0.900 | 57.9 | 0.00365 |
| Mistral Small 3.2 24B | `combined_cot_static_tokens` | Test 05 candidate | 0.8794 | 0.8902 | 0.963 | 0.950 | 70.0 | 0.00441 |
| Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | Test 05 candidate | 0.9063 | 0.9168 | 0.981 | 0.975 | 70.7 | 0.00445 |
| Mistral Small 3.2 24B | `combined_cot_static` | Test 05 candidate | 0.9105 | 0.9206 | 0.981 | 0.975 | 71.3 | 0.00449 |
| Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9238 | 0.9371 | 1.000 | 1.000 | 102.2 | 0.00602 |

Component-quality frontier, using `quality_mean_strict` instead of prompt-level quality:

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | Test 05 candidate | 0.8273 | 0.8434 | 0.926 | 0.900 | 57.9 | 0.00364 |
| Mistral Small 3.2 24B | `max_tokens_quality` | Test 05 candidate | 0.8318 | 0.8531 | 0.926 | 0.900 | 57.9 | 0.00365 |
| Mistral Small 3.2 24B | `combined_cot_static_tokens` | Test 05 candidate | 0.8794 | 0.8902 | 0.963 | 0.950 | 70.0 | 0.00441 |
| Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | Test 05 candidate | 0.9063 | 0.9168 | 0.981 | 0.975 | 70.7 | 0.00445 |
| Mistral Small 3.2 24B | `combined_cot_static` | Test 05 candidate | 0.9105 | 0.9206 | 0.981 | 0.975 | 71.3 | 0.00449 |
| Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9238 | 0.9371 | 1.000 | 1.000 | 102.2 | 0.00602 |

Interpretation: the prompt-level frontier is the better thesis figure because it treats each benchmark question as one end-to-end task. The component frontier is still useful as a sanity check: it shows whether a candidate is only strong because many easy component scores survive while harder prompts fail.

## Prompt Difficulty Pareto

The same non-dominance rule was applied separately inside each prompt difficulty. This asks whether a configuration is still efficient when the benchmark is split into easier and harder tasks.

Non-dominated configurations by difficulty:

| Difficulty | Model | Config | Prompt quality | Completion | kWh / prompt |
| --- | --- | --- | --- | --- | --- |
| 1 | Mistral Small 3.2 24B | `combined_static_best` | 0.9688 | 1.000 | 0.00332 |
| 1 | Mistral Small 3.2 24B | `max_tokens_quality` | 0.9740 | 1.000 | 0.00332 |
| 1 | Mistral Small 3.2 24B | `combined_cot_static_tokens` | 0.9792 | 1.000 | 0.00376 |
| 2 | Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | 0.9422 | 1.000 | 0.00430 |
| 2 | Mistral Small 3.2 24B | `max_tokens_quality` | 0.9750 | 1.000 | 0.00434 |
| 2 | Mistral Small 3.2 24B | `lookup_cot_n3_efficient_tokens` | 0.9792 | 1.000 | 0.00481 |
| 3 | Mistral Small 3.2 24B | `max_tokens_quality` | 0.9167 | 1.000 | 0.00398 |
| 3 | Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | 0.9256 | 1.000 | 0.00400 |
| 3 | Mistral Small 3.2 24B | `combined_cot_static` | 0.9286 | 1.000 | 0.00443 |
| 3 | Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | 0.9390 | 1.000 | 0.00577 |
| 3 | Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | 0.9464 | 1.000 | 0.00609 |
| 4 | Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | 0.4688 | 0.636 | 0.00252 |
| 4 | Mistral Small 3.2 24B | `combined_static_best` | 0.4740 | 0.636 | 0.00254 |
| 4 | Mistral Small 3.2 24B | `combined_cot_static_tokens` | 0.6797 | 0.818 | 0.00443 |
| 4 | Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | 0.8307 | 0.909 | 0.00475 |
| 4 | Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | 0.9193 | 1.000 | 0.00704 |

Best observed configuration per model and difficulty:

| Difficulty | Model | Config | Prompt quality | Completion | kWh / prompt |
| --- | --- | --- | --- | --- | --- |
| 1 | Gemma 26B MoE | `combined_visualization_topk20_tokens` | 0.9427 | 1.000 | 0.00856 |
| 1 | Gemma E4B | `lookup_cot_n4_temp_0p8` | 0.9792 | 1.000 | 0.00512 |
| 1 | Mistral Small 3.2 24B | `combined_cot_static_tokens` | 0.9792 | 1.000 | 0.00376 |
| 2 | Gemma 26B MoE | `lookup_cot_n2` | 0.9167 | 1.000 | 0.01246 |
| 2 | Gemma E4B | `lookup_cot_n4_low_cost_tokens` | 0.9650 | 1.000 | 0.00673 |
| 2 | Mistral Small 3.2 24B | `lookup_cot_n3_efficient_tokens` | 0.9792 | 1.000 | 0.00481 |
| 3 | Gemma 26B MoE | `combined_visualization_topk20_tokens` | 0.8871 | 1.000 | 0.00992 |
| 3 | Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | 0.9464 | 1.000 | 0.00609 |
| 3 | Mistral Small 3.2 24B | `combined_cot_static` | 0.9286 | 1.000 | 0.00443 |
| 4 | Gemma 26B MoE | `lookup_cot_n3` | 0.9167 | 1.000 | 0.01296 |
| 4 | Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | 0.9193 | 1.000 | 0.00704 |
| 4 | Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | 0.8307 | 0.909 | 0.00475 |

Difficulty-4 is the most important thesis split because it contains the hardest multi-step analytical prompts. The best difficulty-4 rows are:

| Difficulty | Model | Config | Prompt quality | Completion | kWh / prompt |
| --- | --- | --- | --- | --- | --- |
| 4 | Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | 0.9193 | 1.000 | 0.00704 |
| 4 | Gemma 26B MoE | `lookup_cot_n3` | 0.9167 | 1.000 | 0.01296 |
| 4 | Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | 0.9132 | 1.000 | 0.00777 |
| 4 | Gemma 26B MoE | `baseline` | 0.9097 | 1.000 | 0.01769 |
| 4 | Gemma 26B MoE | `lookup_cot_n2_visualization_rp1` | 0.9062 | 1.000 | 0.01300 |
| 4 | Gemma E4B | `lookup_cot_n2_temp_0p8` | 0.9062 | 1.000 | 0.00747 |

The difficulty split is also where some aggregate wins become less convincing: a candidate can be Pareto-efficient overall because it saves energy on easy prompts, but still lose the hard prompts that matter most for demonstrating agentic robustness.

## Figures

![Test 05 final-run Pareto frontier](plots/test05_final_run_pareto_frontier.png)

This is the main final-decision plot. Points are full-benchmark model/config executions. The dashed line is the strict prompt-quality Pareto frontier, and outlined points are non-dominated.

![Test 05 baseline deltas](plots/test05_final_run_baseline_deltas.png)

This plot isolates whether each new Test 05 combined candidate improved over its own model baseline, and whether the improvement required additional energy.

![Test 05 Pareto by difficulty](plots/test05_final_run_pareto_by_difficulty.png)

This is the main prompt-difficulty plot. It repeats the Pareto analysis separately for difficulty 1, 2, 3, and 4, making it visible whether a candidate is only efficient on easy prompts or remains useful on hard prompts.

![Test 05 frontier difficulty heatmap](plots/test05_final_run_frontier_difficulty_heatmap.png)

The heatmap shows the overall Pareto-front configurations only, split by difficulty. This is useful for thesis slides because it compresses the frontier into an easy "where does each candidate fail?" view.

## Model-Level Findings

### Gemma E4B

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma E4B | `lookup_cot_n2_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9238 | 0.9371 | 1.000 | 1.000 | 102.2 | 0.00602 |
| Gemma E4B | `lookup_cot_n4_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.9206 | 0.9287 | 1.000 | 1.000 | 108.9 | 0.00645 |
| Gemma E4B | `lookup_cot_n4_temp_0p8` | Test 05 candidate | 0.9121 | 0.9189 | 1.000 | 1.000 | 107.0 | 0.00633 |
| Gemma E4B | `lookup_cot_n4_low_cost_tokens` | Test 05 candidate | 0.9095 | 0.9176 | 1.000 | 1.000 | 108.0 | 0.00636 |
| Gemma E4B | `lookup_cot_n2_temp_0p8` | Test 05 candidate | 0.9047 | 0.9109 | 1.000 | 1.000 | 103.2 | 0.00609 |
| Gemma E4B | `lookup_cot_n2_low_cost_tokens` | Test 05 candidate | 0.8710 | 0.8737 | 0.981 | 0.975 | 102.1 | 0.00603 |
| Gemma E4B | `lookup_temp_0p8_low_cost_tokens` | Test 05 candidate | 0.8650 | 0.8709 | 0.981 | 0.975 | 89.7 | 0.00530 |
| Gemma E4B | `max_tokens_low_cost` | Test 05 candidate | 0.8187 | 0.8361 | 0.944 | 0.925 | 86.8 | 0.00512 |
| Gemma E4B | `baseline` | Test 01 baseline | 0.8213 | 0.8348 | 0.951 | 0.933 | 75.4 | 0.00883 |

Best recommendation for `Gemma E4B` is `lookup_cot_n2_temp_0p8_low_cost_tokens`. It changes prompt quality by `+0.1022`, completion by `+0.049`, and energy by `-0.00281` kWh per prompt relative to the Test 01 baseline.
Best by difficulty: D1: `lookup_cot_n4_temp_0p8` (0.979); D2: `lookup_cot_n4_low_cost_tokens` (0.965); D3: `lookup_cot_n4_temp_0p8_low_cost_tokens` (0.946); D4: `lookup_cot_n2_temp_0p8_low_cost_tokens` (0.919).

### Gemma 26B MoE

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 26B MoE | `combined_static_direct` | Test 05 candidate | 0.8271 | 0.8720 | 1.000 | 1.000 | 153.3 | 0.00998 |
| Gemma 26B MoE | `combined_visualization_topk20_tokens` | Test 05 candidate | 0.8448 | 0.8647 | 1.000 | 1.000 | 153.8 | 0.01003 |
| Gemma 26B MoE | `max_tokens_direct_safe` | Test 05 candidate | 0.8310 | 0.8610 | 1.000 | 1.000 | 157.9 | 0.01030 |
| Gemma 26B MoE | `combined_all_best_static` | Test 05 candidate | 0.8302 | 0.8605 | 1.000 | 1.000 | 156.0 | 0.01020 |
| Gemma 26B MoE | `lookup_cot_n3` | Test 05 candidate | 0.8148 | 0.8605 | 0.981 | 0.975 | 176.9 | 0.01164 |
| Gemma 26B MoE | `lookup_cot_n2` | Test 05 candidate | 0.8095 | 0.8569 | 0.981 | 0.975 | 176.5 | 0.01161 |
| Gemma 26B MoE | `lookup_cot_n2_visualization_rp1` | Test 05 candidate | 0.8060 | 0.8496 | 0.981 | 0.975 | 178.2 | 0.01173 |
| Gemma 26B MoE | `max_tokens_quality` | Test 05 candidate | 0.7996 | 0.8162 | 1.000 | 1.000 | 157.7 | 0.01027 |
| Gemma 26B MoE | `baseline` | Test 01 baseline | 0.7725 | 0.8142 | 0.975 | 0.967 | 136.0 | 0.01667 |
| Gemma 26B MoE | `combined_visualization_rp1_tokens` | Test 05 candidate | 0.7797 | 0.8125 | 0.991 | 0.975 | 150.8 | 0.00979 |
| Gemma 26B MoE | `lookup_cot_n2_combined_all_best` | Test 05 candidate | 0.7844 | 0.8069 | 0.981 | 0.975 | 282.6 | 0.01898 |

Best recommendation for `Gemma 26B MoE` is `combined_static_direct`. It changes prompt quality by `+0.0578`, completion by `+0.025`, and energy by `-0.00668` kWh per prompt relative to the Test 01 baseline.
Best by difficulty: D1: `combined_visualization_topk20_tokens` (0.943); D2: `lookup_cot_n2` (0.917); D3: `combined_visualization_topk20_tokens` (0.887); D4: `lookup_cot_n3` (0.917).

### Mistral Small 3.2 24B

| Model | Config | Source | Quality | Prompt quality | Completion | Full completion | Sec / prompt | kWh / prompt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral Small 3.2 24B | `combined_cot_static` | Test 05 candidate | 0.9105 | 0.9206 | 0.981 | 0.975 | 71.3 | 0.00449 |
| Mistral Small 3.2 24B | `lookup_cot_n2_efficient_tokens` | Test 05 candidate | 0.9063 | 0.9168 | 0.981 | 0.975 | 70.7 | 0.00445 |
| Mistral Small 3.2 24B | `lookup_cot_n3_efficient_tokens` | Test 05 candidate | 0.8963 | 0.9015 | 0.981 | 0.975 | 71.6 | 0.00452 |
| Mistral Small 3.2 24B | `combined_cot_static_tokens` | Test 05 candidate | 0.8794 | 0.8902 | 0.963 | 0.950 | 70.0 | 0.00441 |
| Mistral Small 3.2 24B | `max_tokens_quality` | Test 05 candidate | 0.8318 | 0.8531 | 0.926 | 0.900 | 57.9 | 0.00365 |
| Mistral Small 3.2 24B | `baseline` | Test 01 baseline | 0.8271 | 0.8437 | 0.926 | 0.900 | 48.9 | 0.00601 |
| Mistral Small 3.2 24B | `combined_static_best_efficient_tokens` | Test 05 candidate | 0.8273 | 0.8434 | 0.926 | 0.900 | 57.9 | 0.00364 |
| Mistral Small 3.2 24B | `max_tokens_efficient` | Test 05 candidate | 0.8250 | 0.8420 | 0.926 | 0.900 | 58.8 | 0.00369 |
| Mistral Small 3.2 24B | `combined_static_best` | Test 05 candidate | 0.8175 | 0.8341 | 0.926 | 0.900 | 57.8 | 0.00365 |

Best recommendation for `Mistral Small 3.2 24B` is `combined_cot_static`. It changes prompt quality by `+0.0769`, completion by `+0.056`, and energy by `-0.00152` kWh per prompt relative to the Test 01 baseline.
Best by difficulty: D1: `combined_cot_static_tokens` (0.979); D2: `lookup_cot_n3_efficient_tokens` (0.979); D3: `combined_cot_static` (0.929); D4: `lookup_cot_n2_efficient_tokens` (0.831).

## Reused-Candidate Note

The Test 05 design reused insights from Tests 02-04 to avoid rerunning every single-agent candidate. Those tests are not part of the formal Pareto table above because they used the 10-prompt screening subset, while this report is intentionally full-benchmark. The single-agent tests should be cited as candidate-selection evidence; the Pareto conclusions should be cited from the full-benchmark Test 01 and Test 05 data.

This distinction matters especially for prompt difficulty: the full benchmark has a different prompt mix than the 10-case screening set, so difficulty-specific Pareto claims are only valid on the full-benchmark pool used here.
