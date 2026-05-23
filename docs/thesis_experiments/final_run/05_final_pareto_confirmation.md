# Test 05: Final Pareto Confirmation (Final Rerun)

## Objective

Confirm final model/config choices on a larger benchmark after the step-isolated experiments identify candidate settings.

## Hypothesis

The final thesis should show not only the highest-quality configuration, but also the most energy-efficient configuration that preserves nearly the same quality.

## Evidence From Tests 01-04

The final confirmation should be changed based on the completed exploratory tests.

Important findings:

```text
Test 01:
  mistral-small3.2:24b is the strongest baseline Pareto point.
  gemma4:26b has the best SQL reliability but weaker visualization and higher cost.
  gemma4:e4b is promising but has hard-prompt SQL fragility.

Test 02:
  parameter sensitivity is agent-specific.
  mistral-small3.2:24b is mostly insensitive to tuning.
  gemma4:e4b benefits most from analysis and visualization tuning.
  gemma4:26b benefits most from visualization tuning.

Test 03:
  max_tokens should be set per agent, not globally.
  gemma4:26b visualization strongly benefits from max_tokens=10000.
  mistral-small3.2:24b does not need generous token caps in this range.

Test 04:
  gemma4:e4b lookup_cot_n2 and lookup_bon_temperature_n2 are worth confirming.
  visualization best-of-n with n=2 is useful; n=3 is not.
  visualization CoT is not useful in the current implementation.
```

Therefore Test 05 should not simply re-run the generic best config from each previous axis. It should confirm a small set of interpretable Pareto candidates:

```text
baseline
cost-aware static tuning
accuracy-oriented static tuning
targeted compute expansion only where it helped
combined best candidate
```

The currently available evidence supports final confirmation for:

```text
gemma4:e4b
gemma4:26b
mistral-small3.2:24b
```

## Test Logic

This test must use the full benchmark dataset and three repeats. The manifests should be generated from the evidence gathered in Tests 01-04.

For each completed model, run six configurations:

```text
baseline
efficient_static
best_step_static
max_tokens_adjusted
compute_expansion_candidate
combined_best_candidate
```

This keeps the final experiment small enough to run on the full dataset while still testing the main thesis tradeoffs:

```text
model type vs accuracy/energy
agent-specific parameter sensitivity
max_tokens as a low-cost usability lever
extra compute vs accuracy gain
```

## Candidate Configuration Design

### Gemma E4B

Purpose:

```text
small thinking model; test whether targeted extra compute repairs SQL fragility
```

Configs:

| Config | Intended parameters | Objective |
|---|---|---|
| `baseline` | current Gemma E4B baseline | Reference point. |
| `efficient_static` | lookup max_tokens `3000` or `5000`, analysis max_tokens `10000`, visualization max_tokens `10000`, visualization `top_k_low` | Low-cost static improvement from Tests 02-03. |
| `best_step_static` | analysis static tuning from Test 02 plus visualization `top_k_low` | Confirm static tuning against compute expansion. |
| `max_tokens_adjusted` | lookup not above `5000`, analysis `10000`, visualization `10000` | Confirm that per-agent token caps improve usability. |
| `compute_expansion_candidate` | `lookup_cot_n2` | Confirm the best Test 04 quality-cost signal. |
| `combined_best_candidate` | `lookup_cot_n2` plus strongest static analysis/visualization settings, with per-agent max_tokens | Test the high-quality final candidate. |

Do not include:

```text
lookup_bon_temperature_n3
visualization_bon_top_k_n3
visualization_cot_n2
```

Reason: Test 04 showed that `n=3` is dominated and visualization CoT is negative.

### Gemma 26B MoE

Purpose:

```text
larger thinking MoE; test whether visualization repair makes its high SQL accuracy Pareto-competitive
```

Configs:

| Config | Intended parameters | Objective |
|---|---|---|
| `baseline` | current Gemma 26B baseline | Reference point. |
| `efficient_static` | lookup max_tokens `3000` or `5000`, analysis max_tokens `5000`, visualization max_tokens `10000` | Keep SQL/text strong while removing unnecessary token budget. |
| `best_step_static` | visualization `top_k_high` with visualization max_tokens `10000` | Confirm the cost-aware visualization fix from Test 02 plus Test 03. |
| `max_tokens_adjusted` | visualization max_tokens `10000`, other agents at safe lower caps | Confirm the large visualization token effect from Test 03. |
| `compute_expansion_candidate` | visualization Best-of-N `top_k`, `n=2` | Confirm the strongest visualization repair from Test 02. |
| `combined_best_candidate` | visualization max_tokens `10000` plus best static visualization setting; optional lookup/analysis static improvements only if they do not hurt step metrics | Test high-quality MoE candidate. |

Do not include:

```text
analysis_temperature_low
analysis_cot_n2
visualization_temperature_low
```

Reason: these were harmful or large runtime/energy outliers.

### Mistral Small 3.2 24B

Purpose:

```text
larger non-thinking baseline; confirm that the best Pareto model remains strong with lightweight tuning
```

Configs:

| Config | Intended parameters | Objective |
|---|---|---|
| `baseline` | current Mistral Small baseline | Reference point and likely Pareto anchor. |
| `efficient_static` | lookup max_tokens `3000`, analysis max_tokens `5000` or `7000`, visualization max_tokens `5000` or `7000` | Confirm lower token caps preserve quality. |
| `best_step_static` | analysis `top_k_high` or lookup `top_k_high`, selected from Test 02 | Confirm the only small static gains observed. |
| `max_tokens_adjusted` | lower safe token caps from Test 03 | Test whether efficiency improves without accuracy loss. |
| `analysis_top_k_high_low_tokens` | analysis `top_k_high` with low safe token caps | Extra efficient static variant replacing compute expansion. |
| `combined_best_candidate` | efficient token caps plus the best cheap static parameter | Confirm the final Mistral Pareto candidate. |

Do not include Best-of-N or CoT unless there is a new result showing a large gain. Tests 02-03 suggest Mistral is already near its local optimum.

## Selection Rules

Best step config:

```text
highest step-relevant score
tie-break by lower energy
```

Best overall quality config:

```text
highest quality_mean
tie-break by lower energy
```

Best energy-efficient config:

```text
lowest energy among configs within 0.02 quality_mean of the best available quality
```

Additional final selection rule:

```text
Prefer n=1 static settings unless Best-of-N or CoT improves quality by at least 0.03 on the full benchmark.
```

Reason: extra calls increase time and energy by construction. Test 04 showed that only selected compute-expansion settings are worth confirming.

## Metrics

Primary:

```text
quality_mean
prompt_quality
csv_iou
text_score
vis_score
energy_consumed_kwh
gpu_energy_kwh
elapsed_sec
```

Report:

```text
mean
median
standard deviation
timeout rate
completion rate
accuracy_per_kwh
prompt_quality_per_kwh
```

Metric note:

```text
prompt_quality should be reported alongside quality_mean.
```

Reason: Test 01 showed that `quality_mean` can hide full-pipeline failures when SQL errors prevent downstream text or visualization scores from being produced.

## Thesis Objective

This test produces the final tables and Pareto plots for the thesis discussion.

## Completeness Check

Expected rows:

```text
3 models x 3 repeats x 6 configs x full dataset size
```

## Code Readiness

Ready to execute after rebuilding the Docker image from this branch.

The final manifests have been generated and contain exactly six configs each.
The manifest runner also writes the final-report metrics needed by this test:

```text
prompt_quality_mean
completion_rate
full_completion_rate
quality_per_kwh
prompt_quality_per_kwh
```

Expected final manifests:

```text
evaluation/thesis_final_run/thesis_test05_gemma4_e4b_final_pareto.yaml
evaluation/thesis_final_run/thesis_test05_gemma4_26b_final_pareto.yaml
evaluation/thesis_final_run/thesis_test05_mistral_small32_24b_final_pareto.yaml
```

Each final manifest should contain exactly six configs:

```text
baseline
efficient_static
best_step_static
max_tokens_adjusted
compute_expansion_candidate or efficient_static_variant
combined_best_candidate
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

Run the three final measured repeats sequentially on one GPU:

```bash
for REP in 01 02 03; do
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
    --resume
done
```
