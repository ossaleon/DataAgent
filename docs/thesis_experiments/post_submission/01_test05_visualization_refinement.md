# Post-Submission Run 01: Test 05 Visualization Refinement

## Objective

This post-submission run refines the final Test 05 Pareto analysis without
rewriting the submitted experiment history.

It has two goals:

1. Check whether the weak `gemma4:26b` visualization axis in the radar plot is a
   configuration problem, especially an insufficient visualization token budget.
2. Rerun the E4B and Mistral Test 05 candidates that were originally reused from
   previous tests, so those rows are measured on the same machine and full
   dataset as the newer Test 05 configurations.

The run uses the full benchmark:

```text
evaluation/benchmark_dataset.json
```

Each manifest should be run twice:

```text
rep01
rep02
```

Results are written under a new folder so they do not overwrite the submitted
final-run data:

```text
runs/post_submission/test05_refinement/
```

## Why Gemma 26B Needs A Visualization Check

The final Test 05 radar plot showed a very asymmetric `gemma4:26b` profile:
strong SQL/data and text scores, but a weak visualization score for the selected
representative `combined_static_direct`.

The raw Test 05 full-dataset results support this concern:

| Config | Prompt quality | CSV | Text | Vis | Completion | kWh / prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined_static_direct` | 0.8720 | 0.9675 | 0.9625 | 0.5513 | 100.0% | 0.00998 |
| `combined_visualization_topk20_tokens` | 0.8647 | 0.9987 | 0.8594 | 0.6763 | 100.0% | 0.01003 |
| `max_tokens_quality` | 0.8162 | 0.9581 | 0.8156 | 0.6250 | 100.0% | 0.0103 |
| `max_tokens_direct_safe` | 0.8610 | 0.9425 | 0.9344 | 0.6161 | 100.0% | 0.0103 |

The problem is not that the visualization step is skipped. In the two Test 05
repetitions, `combined_static_direct` completed all expected visualization
slots, but 4 out of 28 expected visualizations received a zero score and 5 out
of 28 were below 0.5. The corresponding judge reasons include no chart code,
partial renderability, and wrong grouped-bar mappings. By contrast,
`combined_visualization_topk20_tokens` had no zero visualization slots and only
1 out of 28 below 0.5.

The earlier max-token ladder also points in the same direction. On the Test 03
subset, `visualization_max_tokens_10000` reached prompt quality 0.9123 and
visualization score 0.6696, while the 7000-token visualization baseline reached
0.8901 and 0.6161. The curve did not show a clear plateau. This makes a focused
full-dataset visualization-cap rerun reasonable.

## New Gemma 26B Visualization Repair Configurations

The new `gemma4:26b` manifest contains 10 whole-agent configurations. They keep
lookup and analysis generous enough that visualization remains the main
variable, then test larger visualization caps and the strongest visualization
sampling signals from Tests 02 and 05.

Manifest:

```text
evaluation/post_submission/thesis_post_submission_gemma4_26b_visualization_repair.yaml
```

| Config | Purpose |
| --- | --- |
| `vis_tokens_10000_reference` | Full-dataset rerun of the strongest Test 03 visualization-token direction. |
| `vis_tokens_12000` | Check whether the 10000-token benefit continues. |
| `vis_tokens_14000` | Higher cap for possible unfinished plotting code. |
| `vis_tokens_16000` | Stress-test whether visualization still benefits from more output budget. |
| `vis_topk20_tokens12000` | Combine the best Test 05 visualization sampling signal with a larger cap. |
| `vis_topk20_tokens14000` | Same, with a higher cap. |
| `vis_repeat_penalty_1_tokens12000` | Recheck the strong Test 02 `repeat_penalty=1.0` signal with more tokens. |
| `vis_repeat_penalty_1p03_tokens12000` | Recheck the best direct visualization-score signal from Test 02. |
| `vis_temp0p95_topk20_tokens12000` | Combine the difficulty-3 temperature signal with `top_k=20`. |
| `vis_static_topk20_rp1p03_tokens14000` | Most aggressive visualization-repair candidate: `top_k=20`, `repeat_penalty=1.03`, and 14000 tokens, with strong lookup/analysis settings. |

## Reused Test 05 Candidates To Rerun

The submitted Test 05 analysis intentionally reused some candidate rows from
Tests 01-04. For this post-submission refinement, these rows should be rerun on
the full dataset and the same machine as the new Test 05 rows.

### Gemma E4B

Manifest:

```text
evaluation/post_submission/thesis_post_submission_gemma4_e4b_reused_test05.yaml
```

| Source | Config |
| --- | --- |
| Test 01 | `baseline` |
| Test 03 | `analysis_max_tokens_5000` |
| Test 03 | `visualization_max_tokens_4000` |
| Test 04 | `lookup_cot_n2` |
| Test 04 | `lookup_cot_n4` |
| Test 04b | `lookup_temp_0p3` |
| Test 04b | `lookup_temp_0p8` |

### Mistral Small 3.2 24B

Manifest:

```text
evaluation/post_submission/thesis_post_submission_mistral_small32_24b_reused_test05.yaml
```

| Source | Config |
| --- | --- |
| Test 01 | `baseline` |
| Test 02 | `analysis_top_k_56` |
| Test 02 | `lookup_repeat_penalty_1p3` |
| Test 02 | `visualization_repeat_last_n_56` |
| Test 04 | `lookup_cot_n2` |
| Test 04 | `lookup_cot_n3` |
| Test 04 | `lookup_cot_n4` |

## Run Command

Build the Docker image once:

```bash
docker build -t data-agent .
```

Then run the whole post-submission batch sequentially:

```bash
bash evaluation/run_post_submission_test05_refinement_docker.sh
```

Optional dry run:

```bash
bash evaluation/run_post_submission_test05_refinement_docker.sh --dry-run
```

Useful environment overrides:

```bash
DATA_AGENT_IMAGE=data-agent
GPU_DEVICE=0
OLLAMA_HOST=http://localhost:11434
RUNS_HOST_DIR="$(pwd)/runs"
GT_JUDGE_PROVIDER=openai
GT_JUDGE_MODEL=gpt-5.4
OPENAI_API_KEY=...
```

The script runs all model/repetition pairs sequentially on one GPU device to keep
energy attribution atomic.

## Expected New Rows

| Model | Configs | Repeats | Prompts | Expected prompt rows |
| --- | ---: | ---: | ---: | ---: |
| `gemma4:26b` | 10 | 2 | 20 | 400 |
| `gemma4:e4b` | 7 | 2 | 20 | 280 |
| `mistral-small3.2:24b` | 7 | 2 | 20 | 280 |

Total expected prompt rows:

```text
960
```

## Merge Protocol

When updating the post-submission Test 05 analysis:

1. Keep the submitted final-run data unchanged as historical evidence.
2. Add these rows with source label:

```text
post_submission_test05_refinement
```

3. For E4B and Mistral, replace the previous-test reused candidate rows in the
   Test 05 candidate pool with the corresponding post-submission rerun rows.
4. Keep the original new Test 05 combined rows unless an exact rerun exists.
5. Add the 10 new `gemma4:26b` visualization-repair rows as additional final
   candidates.
6. Report the analysis as a refinement of the submitted thesis, not as part of
   the original submitted run history.

