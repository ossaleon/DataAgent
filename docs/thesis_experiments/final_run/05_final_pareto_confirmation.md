# Test 05: Incremental Final Pareto Confirmation

## Objective

Build the final Pareto comparison without rerunning configurations that are already available from Tests 01-04.

The objective is no longer parameter exploration. Test 05 should run only new whole-agent configurations that are likely to be competitive after the previous step-isolated tests. The final analysis then merges:

```text
previously measured strong candidates
new Test 05 combined candidates
```

This keeps the run small enough to execute quickly while still giving the final thesis plots enough non-dominated configurations.

## Dataset And Repetitions

New Test 05 runs use the full benchmark:

```text
evaluation/benchmark_dataset.json
```

Full dataset size:

```text
20 prompts
```

Run only two repetitions:

```text
rep01
rep02
```

Reason: Tests 02-04 already showed that the broad energy/latency conclusions are stable enough with two complete repetitions. With limited time, the budget is better spent on more useful Pareto candidates than on a third repeat.

## Models

```text
gemma4:e4b
gemma4:26b
mistral-small3.2:24b
```

## Design Rule

Do not rerun configurations already measured in previous tests. Instead:

```text
Use previous tests for single-axis candidates.
Use Test 05 only for combined high-performance candidates.
```

This means Test 05 manifests intentionally do not include baseline rows or simple single-agent variants that already exist in Tests 01-04.

## Candidate Pool

The main final Pareto pool should have about 15 candidate configurations per model after merging selected previous-test candidates with the new Test 05 runs.

| Model | Priority existing candidates | New Test 05 configs | Main merged pool |
| --- | ---: | ---: | ---: |
| `gemma4:e4b` | 7 | 8 | 15 |
| `gemma4:26b` | 5 | 10 | 15 |
| `mistral-small3.2:24b` | 7 | 8 | 15 |

Additional previous-test configs can still be kept in appendix tables, but they should not crowd the main Pareto plot unless they become non-dominated after normalization.

## Existing Configurations To Reuse

### Gemma E4B

Reuse:

```text
Test 01:
  baseline

Test 03:
  analysis_max_tokens_5000
  visualization_max_tokens_4000

Test 04:
  lookup_cot_n2
  lookup_cot_n4

Test 04b:
  lookup_temp_0p3
  lookup_temp_0p8
```

Do not reuse E4B Test 02 because the final-run data is incomplete.

### Gemma 26B MoE

Reuse:

```text
Test 01:
  baseline

Test 02:
  visualization_repeat_penalty_1
  visualization_top_k_20
  lookup_repeat_penalty_1p3

Test 03:
  visualization_max_tokens_10000
```

Other previous Gemma 26B candidates such as `visualization_repeat_penalty_1p03`, `visualization_top_k_48`, `visualization_top_k_128`, `lookup_repeat_last_n_96`, `lookup_max_tokens_8000`, and `analysis_max_tokens_4000` can be retained for appendix analysis, but the five above are the priority points for the main Pareto plot.

### Mistral Small 3.2 24B

Reuse:

```text
Test 01:
  baseline

Test 02:
  analysis_top_k_56
  lookup_repeat_penalty_1p3
  visualization_repeat_last_n_56

Test 04:
  lookup_cot_n2
  lookup_cot_n3
  lookup_cot_n4
```

The single-agent max-token rows from Test 03 are useful supporting evidence, but the main Mistral Pareto plot should use the new combined token-cap configs from Test 05.

## New Test 05 Configurations

### Gemma E4B

Manifest:

```text
evaluation/thesis_final_run/thesis_test05_gemma4_e4b_final_pareto.yaml
```

Configs to run:

| Config | Objective |
| --- | --- |
| `max_tokens_low_cost` | Combine the two best low-cost token-cap directions. |
| `lookup_temp_0p8_low_cost_tokens` | Static lookup repair plus low-cost downstream caps. |
| `lookup_cot_n2_temp_0p8` | Test interaction between useful lookup CoT and static lookup temperature. |
| `lookup_cot_n4_temp_0p8` | Higher-depth lookup repair with the best static temperature. |
| `lookup_cot_n2_low_cost_tokens` | Cost-aware CoT candidate. |
| `lookup_cot_n4_low_cost_tokens` | Higher-depth CoT with cheaper downstream stages. |
| `lookup_cot_n2_temp_0p8_low_cost_tokens` | Main E4B Pareto candidate. |
| `lookup_cot_n4_temp_0p8_low_cost_tokens` | Highest-accuracy E4B candidate. |

### Gemma 26B MoE

Manifest:

```text
evaluation/thesis_final_run/thesis_test05_gemma4_26b_final_pareto.yaml
```

Configs to run:

| Config | Objective |
| --- | --- |
| `max_tokens_quality` | Combine the best per-agent token caps from Test 03. |
| `max_tokens_direct_safe` | Safer token-cap variant that keeps analysis at the baseline cap. |
| `combined_visualization_rp1_tokens` | Strongest visualization repair plus token caps. |
| `combined_visualization_topk20_tokens` | Alternative visualization repair plus token caps. |
| `combined_static_direct` | Combine the strongest static direct-agent settings. |
| `combined_all_best_static` | Full static/token high-accuracy candidate. |
| `lookup_cot_n2` | New Gemma 26B CoT check inspired by E4B lookup gains. |
| `lookup_cot_n3` | Second Gemma 26B CoT depth check. |
| `lookup_cot_n2_visualization_rp1` | Lookup CoT plus strongest visualization repair. |
| `lookup_cot_n2_combined_all_best` | Highest-accuracy Gemma 26B candidate. |

### Mistral Small 3.2 24B

Manifest:

```text
evaluation/thesis_final_run/thesis_test05_mistral_small32_24b_final_pareto.yaml
```

Configs to run:

| Config | Objective |
| --- | --- |
| `max_tokens_efficient` | Combine efficient token caps that were tested separately in Test 03. |
| `max_tokens_quality` | Token candidate that keeps lookup at the safer baseline cap. |
| `lookup_cot_n2_efficient_tokens` | Lower-cost lookup CoT with efficient token caps. |
| `lookup_cot_n3_efficient_tokens` | Best lookup CoT depth with efficient token caps. |
| `combined_static_best` | Combine small positive static signals from Test 02. |
| `combined_static_best_efficient_tokens` | Static best candidate plus efficient token caps. |
| `combined_cot_static` | Best lookup CoT depth plus static settings. |
| `combined_cot_static_tokens` | Final Mistral high-performance candidate. |

## Expected New Rows

Only the new Test 05 runs are counted here:

| Model | New configs | Repeats | Full-dataset prompts | Expected new rows |
| --- | ---: | ---: | ---: | ---: |
| `gemma4:e4b` | 8 | 2 | 20 | 320 |
| `gemma4:26b` | 10 | 2 | 20 | 400 |
| `mistral-small3.2:24b` | 8 | 2 | 20 | 320 |

Total expected new prompt rows:

```text
1040
```

## Merge Protocol

The final Test 05 report should be built from a merged table with one row per:

```text
model
config_name
source_test
source_dataset
repeat
```

Recommended source labels:

```text
test01_full_baseline
test02_static_sensitivity
test03_max_tokens
test04_lookup_cot
test04b_lookup_temperature
test05_incremental_pareto
```

For every source, keep these columns:

```text
model
config_name
source_test
source_dataset
repeat
n_prompts
quality_mean_strict
csv_iou_mean
text_score_mean
vis_score_mean
completion_rate
full_completion_rate
elapsed_sec_mean
energy_consumed_kwh_mean
gpu_energy_kwh_mean
emissions_kg_co2_mean
```

Important accuracy rule:

```text
Expected missing GT score slots count as 0.
```

Do not drop failed prompt rows from the mean. A failed SQL query, missing text answer, or missing visualization is part of accuracy.

## Pareto Reporting Rules

Use two Pareto views:

```text
1. Full-dataset confirmation Pareto
2. Screening-supported Pareto
```

Full-dataset confirmation Pareto:

```text
Use Test 01 baselines plus the new Test 05 runs only.
```

Reason: these are evaluated on the same full 20-prompt dataset.

Screening-supported Pareto:

```text
Use Tests 02-04b plus Test 05, but mark source_dataset clearly.
```

Reason: Tests 02-04b were run on smaller fixed subsets. They are valid evidence for selecting candidates and understanding behavior, but they should not be visually indistinguishable from full-dataset confirmation points.

Main thesis plot recommendation:

```text
Show full-dataset points as filled markers.
Show reused screening points as hollow or lighter markers.
Label only non-dominated full-dataset points.
```

This avoids pretending that a 10-prompt screening run and a 20-prompt final run have exactly the same evidential weight.

## Selection Criteria

Best quality:

```text
highest strict quality_mean
tie-break by lower energy
```

Best efficient candidate:

```text
lowest energy among configs within 0.02 quality of the best model-specific full-dataset quality
```

Compute expansion acceptance:

```text
Keep lookup CoT only if it improves full-dataset quality by at least 0.03
or fixes a clear SQL failure class at acceptable energy cost.
```

Final thesis recommendation:

```text
Prefer n=1 static/token settings unless lookup CoT gives a clear full-dataset accuracy gain.
```

## Remote Docker Commands

Run these from the repository root on the remote machine.

Build the image once after pulling the final manifest changes:

```bash
docker build -t data-agent .
```

Start Ollama once outside Docker:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the models:

```bash
ollama pull gemma4:e4b
ollama pull gemma4:26b
ollama pull mistral-small3.2:24b
```

Export the shared GT judge key:

```bash
export OPENAI_API_KEY="..."
```

Run the two final measured repeats sequentially on one GPU:

```bash
for REP in 01 02; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test05_gemma4_e4b_final_pareto.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/05_final_pareto_confirmation/gemma4_e4b/rep${REP} \
    --repetition ${REP}/02 \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test05_gemma4_26b_final_pareto.yaml \
    --provider ollama \
    --model gemma4:26b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/05_final_pareto_confirmation/gemma4_26b/rep${REP} \
    --repetition ${REP}/02 \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test05_mistral_small32_24b_final_pareto.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/05_final_pareto_confirmation/mistral_small32_24b/rep${REP} \
    --repetition ${REP}/02 \
    --resume
done
```

## Smoke Test

Before launching the full run:

```bash
docker run --rm --gpus '"device=0"' --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset.json \
  evaluation/thesis_final_run/thesis_test05_gemma4_e4b_final_pareto.yaml \
  --provider ollama \
  --model gemma4:e4b \
  --gt-judge-provider openai \
  --gt-judge-model gpt-5.4 \
  --save-dir runs/smoke/05_final_pareto_confirmation/gemma4_e4b \
  --max-configs 2 \
  --max-prompts 1 \
  --no-codecarbon \
  --resume
```
