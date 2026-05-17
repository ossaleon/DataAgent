# Test 04 Report: Gemma E4B Compute Expansion

## Data Source

Results analyzed from:

```text
/home/oss/Downloads/04_compute_expansion_n_and_cot/
```

Completed result folders:

```text
gemma4_e4b/rep01
gemma4_e4b/rep02
```

Each repeat contains:

```text
10 configs x 10 prompts = 100 benchmark rows
```

This test is limited to:

```text
gemma4:e4b
```

The purpose is to see whether the small thinking model can recover accuracy through extra inference calls: Best-of-N or CoT refinement.

## Measurement Caveat

The absolute energy values in Test 04 are much lower than Test 03 for similar elapsed times. For example, each Test 04 repeat has about `17,100` seconds of runtime but only about `0.21 kWh` total energy.

Therefore, this report uses:

```text
elapsed_sec       primary compute-cost signal
energy deltas     useful within Test 04, but not compared directly against Test 03
```

The direction of energy deltas is still informative inside this test because configs were run under the same measurement setup.

## Method Notes

This test keeps the bulk-runner logic:

```text
vary one agent step at a time
keep the other two steps at baseline
```

It tests:

```text
lookup_sales_data:
  baseline
  best_of_n with temperature, n=2
  best_of_n with temperature, n=3
  cot_n=2

analyzing_data:
  baseline
  best_of_n with temperature, n=2

create_visualization:
  baseline
  best_of_n with top_k, n=2
  best_of_n with top_k, n=3
  cot_n=2
```

Interpretation uses the mean across `rep01` and `rep02`.

## Executive Findings

Compute expansion helps Gemma E4B, but only in specific forms.

- Lookup is the best place for extra compute. `lookup_cot_n2` gives the best average quality gain: `+0.1142`, with only `+5.9s` per prompt. `lookup_bon_temperature_n2` is almost as strong and very consistent across repeats.
- Analysis Best-of-N helps, but it is expensive: `+0.0771` quality for `+70.4s` per prompt. It should be compared against cheaper static settings from Test 02 before becoming a final recommendation.
- Visualization Best-of-N with `top_k`, `n=2` helps: `+0.0501` quality and perfect average `vis_score`, but costs `+53.4s`.
- `n=3` is not justified. It is worse than `n=2` for lookup and visualization while adding more calls.
- Visualization CoT is not useful in this run. It reduces average quality by `-0.0276`.

Practical thesis signal:

```text
Extra compute should be step-specific and capped at n=2.
More calls are not automatically better.
```

## Overall Resource Use

| Repeat | Rows | Total sec | Total kWh | Total GPU kWh | Mean sec / prompt | Mean kWh / prompt |
|---|---:|---:|---:|---:|---:|---:|
| `rep01` | 100 | 17,098.6 | 0.2108 | 0.1643 | 171.0 | 0.0021 |
| `rep02` | 100 | 17,219.0 | 0.2127 | 0.1659 | 172.2 | 0.0021 |

The two repeats have very similar runtime and measured energy, which makes within-test deltas stable enough for design decisions.

## Aggregate Results

| Config | Varied step | Quality | Quality delta | Step metric | Step delta | Sec / prompt | Sec delta | kWh / prompt | kWh delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lookup_baseline` | lookup | 0.8400 | 0.0000 | `csv_iou=0.8327` | 0.0000 | 164.3 | 0.0 | 0.0020 | 0.0000 |
| `lookup_bon_temperature_n2` | lookup | 0.9418 | +0.1018 | `csv_iou=0.9905` | +0.1579 | 178.4 | +14.1 | 0.0022 | +0.0002 |
| `lookup_bon_temperature_n3` | lookup | 0.8959 | +0.0559 | `csv_iou=0.8934` | +0.0608 | 167.5 | +3.1 | 0.0020 | -0.0000 |
| `lookup_cot_n2` | lookup | 0.9542 | +0.1142 | `csv_iou=0.9687` | +0.1361 | 170.2 | +5.9 | 0.0021 | +0.0001 |
| `analysis_baseline` | analysis | 0.8572 | 0.0000 | `text_score=0.7929` | 0.0000 | 121.8 | 0.0 | 0.0015 | 0.0000 |
| `analysis_bon_temperature_n2` | analysis | 0.9343 | +0.0771 | `text_score=0.9028` | +0.1099 | 192.2 | +70.4 | 0.0024 | +0.0009 |
| `visualization_baseline` | visualization | 0.9083 | 0.0000 | `vis_score=0.9795` | 0.0000 | 142.1 | 0.0 | 0.0018 | 0.0000 |
| `visualization_bon_top_k_n2` | visualization | 0.9584 | +0.0501 | `vis_score=1.0000` | +0.0205 | 195.5 | +53.4 | 0.0025 | +0.0007 |
| `visualization_bon_top_k_n3` | visualization | 0.9022 | -0.0060 | `vis_score=1.0000` | +0.0205 | 227.2 | +85.1 | 0.0029 | +0.0011 |
| `visualization_cot_n2` | visualization | 0.8807 | -0.0276 | `vis_score=0.9766` | -0.0029 | 156.7 | +14.6 | 0.0019 | +0.0002 |

The best overall config is:

```text
lookup_cot_n2
```

The best Best-of-N config is:

```text
lookup_bon_temperature_n2
```

The strongest negative result is:

```text
visualization_cot_n2
```

## Figures

The figures below show Test 04 as a compute-expansion experiment: when extra calls improve accuracy, how much latency they add, and whether the selector actually uses later candidates.

![Test 04 compute expansion Pareto](plots/test04_compute_expansion_pareto.png)

This is the main Test 04 decision plot. Points above zero improve full-pipeline quality, while points farther right add more energy per prompt. The strongest result is that lookup CoT and lookup Best-of-N both improve quality with relatively modest extra energy, while visualization `n=3` and visualization CoT are not attractive.

![Test 04 agent metric gain cost](plots/test04_agent_metric_gain_cost.png)

This plot isolates the direct metric of the varied agent: `csv_iou` for lookup, `text_score` for analysis, and `vis_score` for visualization. It shows that lookup compute expansion repairs the intended SQL metric, while visualization improvements are smaller because the baseline visual score was already high.

![Test 04 accuracy vs extra calls](plots/test04_accuracy_vs_extra_calls.png)

This plot shows that more calls are not monotonically better. `n=2` is the useful Best-of-N setting; `n=3` is dominated for lookup and visualization in this run.

![Test 04 Best-of-N selection behavior](plots/test04_bon_selection_behavior.png)

This plot checks whether Best-of-N is really benefiting from later candidates. Later candidates are rarely selected, especially for lookup and visualization `n=3`, which supports the caveat that some Best-of-N gains may come from the first candidate's parameter setting rather than from the selector.

## Lookup Agent

Lookup is the clearest winner for compute expansion.

| Config | Quality delta | csv_iou delta | Sec delta | Interpretation |
|---|---:|---:|---:|---|
| `lookup_bon_temperature_n2` | +0.1018 | +0.1579 | +14.1 | Strong and consistent improvement. |
| `lookup_bon_temperature_n3` | +0.0559 | +0.0608 | +3.1 | Positive, but dominated by `n=2`. |
| `lookup_cot_n2` | +0.1142 | +0.1361 | +5.9 | Best quality-cost tradeoff in this test. |

`lookup_cot_n2` is especially important because SQL has an executable artifact and concrete failure modes. A feedback loop can repair joins, filters, grouping, and result shape in a way that is directly useful to the benchmark.

However, `lookup_bon_temperature_n2` should also be kept as a candidate for final confirmation because it is more repeat-stable:

```text
rep01 quality delta: +0.1028
rep02 quality delta: +0.1007
```

`lookup_cot_n2` is better on average, but more variable:

```text
rep01 quality delta: +0.1352
rep02 quality delta: +0.0931
```

## Analysis Agent

Analysis Best-of-N improves text quality, but the compute cost is much larger than lookup.

```text
quality_mean: +0.0771
text_score:   +0.1099
elapsed_sec:  +70.4s per prompt
energy:       +0.0009 kWh per prompt
```

This is useful evidence that Gemma E4B's analysis step benefits from candidate diversity. But it is not automatically the best thesis recommendation, because Test 02 found cheaper static analysis changes that should be compared against it.

Recommended interpretation:

```text
Analysis Best-of-N is accuracy-positive but probably not Pareto-optimal unless static settings fail in the final confirmation test.
```

## Visualization Agent

Visualization has a mixed result.

`visualization_bon_top_k_n2` is useful:

```text
quality_mean: +0.0501
vis_score:    0.9795 -> 1.0000
elapsed_sec:  +53.4s per prompt
```

`visualization_bon_top_k_n3` is not useful:

```text
quality_mean: -0.0060
elapsed_sec:  +85.1s per prompt
```

`visualization_cot_n2` is negative:

```text
quality_mean: -0.0276
vis_score:    -0.0029
elapsed_sec:  +14.6s per prompt
```

This suggests that visualization does benefit from controlled diversity, but not from adding more than two candidates and not from the current CoT loop.

## Candidate Selection Behavior

The candidate-selection logs show an important detail: Best-of-N often selected the first candidate.

| Config | Candidates | Non-first selections | Mean selection margin | Interpretation |
|---|---:|---:|---:|---|
| `lookup_bon_temperature_n2` | 2 | 0 / 20 | 0.0000 | Selection usually tied; first candidate dominated. |
| `lookup_bon_temperature_n3` | 3 | 2 / 20 | 0.0250 | Extra candidates were rarely selected. |
| `analysis_bon_temperature_n2` | 2 | 1 / 18 scored | 0.1296 | Some separation, but mostly first candidate. |
| `visualization_bon_top_k_n2` | 2 | 3 / 13 visual cases | 0.0462 | The second candidate is sometimes useful. |
| `visualization_bon_top_k_n3` | 3 | 0 / 12 visual cases | 0.0000 | Third candidate adds cost without selection benefit. |

This is thesis-relevant. Best-of-N is not only testing "more samples"; it is also testing the ordered candidate schedule. If the first candidate is usually selected, the gain may partly come from the first candidate's parameter value, not from the selector.

For final confirmation, compare these against cheaper static settings:

```text
lookup_bon_temperature_n2      vs static low/medium lookup temperature
analysis_bon_temperature_n2    vs static analysis temperature/top_k from Test 02
visualization_bon_top_k_n2     vs static visualization top_k_low from Test 02
```

## CoT Findings

CoT behaves differently by agent.

| Step | CoT result | Interpretation |
|---|---|---|
| lookup | Strong positive: `quality +0.1142`, `csv_iou +0.1361`. | Keep for final confirmation. SQL can exploit feedback. |
| visualization | Negative: `quality -0.0276`, `vis_score -0.0029`. | Reject in current form. The visualization baseline is already high, and the extra pass can perturb otherwise good outputs. |

This supports the user's original intuition only partially:

```text
SQL generation benefits from feedback.
Visualization does not benefit from this specific feedback loop in this run.
```

## Thesis Interpretation

This test supports four thesis-level claims.

First, extra compute is not universally valuable. It must be targeted at the agent step that can use it.

Second, `n=2` is the practical upper bound for this pipeline. `n=3` is dominated in both lookup and visualization.

Third, CoT should be artifact-aware. SQL has a natural correction loop; visualization, at least with the current implementation, does not show the same benefit.

Fourth, Best-of-N can be confounded with parameter scheduling. Since the first candidate is selected most of the time, a static parameter setting may recover much of the gain at lower cost.

## Recommended Follow-Up

Carry these configs into the final Pareto confirmation:

| Candidate | Why keep it |
|---|---|
| `lookup_cot_n2` | Best average quality-cost tradeoff in Test 04. |
| `lookup_bon_temperature_n2` | Very stable quality gain across both repeats. |
| `analysis_bon_temperature_n2` | Large analysis gain, but must compete with cheaper static settings. |
| `visualization_bon_top_k_n2` | Useful controlled diversity; reaches perfect average `vis_score`. |

Reject these for final confirmation unless there is a new reason to retest:

| Candidate | Why reject it |
|---|---|
| `lookup_bon_temperature_n3` | Positive but dominated by `n=2` and CoT. |
| `visualization_bon_top_k_n3` | More expensive than `n=2` and lower full quality. |
| `visualization_cot_n2` | Negative average quality and no visual-score gain. |

The final thesis discussion can frame Test 04 as evidence that small thinking models can recover accuracy with selective extra computation, but the recovery is not free and not monotonic.
