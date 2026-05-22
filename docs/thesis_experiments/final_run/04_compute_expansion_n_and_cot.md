# Test 04: CoT Depth Effectiveness (Final Rerun)

## Objective

Measure whether iterative CoT refinement improves the two agent steps that can exploit a concrete feedback loop:

```text
lookup_sales_data
create_visualization
```

The test keeps the bulk-runner ablation logic. One step changes at a time, the other two stay at the tested model baseline.

## Models

```text
gemma4:e4b
mistral-small3.2:24b
```

Reason: Gemma E4B is the small thinking-model candidate where CoT already showed a useful signal. Mistral Small is the larger non-thinking control, so this test checks whether external refinement closes a quality gap or only adds compute.

## Why CoT Only

The first batch mixed Best-of-N and CoT as compute-expansion alternatives. The final rerun removes Best-of-N from this test so the measured effect is the CoT loop itself:

```text
requested refinement depth
actual refinement depth reached
quality gain
time and energy increase
early convergence rate
```

Analysis is excluded. Its text feedback loop is less concrete than SQL execution and visualization repair, and the previous run did not make it the strongest CoT target.

## Baseline Defaults

All configs use:

```text
n=1
bon_param=temperature
max_tokens lookup_sales_data=5000
max_tokens analyzing_data=7000
max_tokens create_visualization=7000
use_cache=false
```

Sampling baselines:

```text
gemma4:e4b:
  temperature=1.0
  top_p=0.95
  top_k=64

mistral-small3.2:24b:
  temperature=0.15
  top_p=0.95
  top_k=64
```

## Test Logic

Run a depth ladder from one initial generation to four total CoT iterations.

Lookup phase:

```text
lookup_sales_data      cot_n varied: 1, 2, 3, 4
analyzing_data         fixed baseline
create_visualization   fixed baseline
```

Visualization phase:

```text
lookup_sales_data      fixed baseline
analyzing_data         fixed baseline
create_visualization   cot_n varied: 1, 2, 3, 4
```

`cot_n=1` is the phase baseline. `cot_n=4` does not force four completed generations: the runtime may stop earlier when the generated artifact converges.

## Configuration Set

Each model manifest has exactly 8 configs:

```text
lookup_cot_n1
lookup_cot_n2
lookup_cot_n3
lookup_cot_n4
visualization_cot_n1
visualization_cot_n2
visualization_cot_n3
visualization_cot_n4
```

The varied CoT depth is the only configuration change inside each phase.

## CoT Stop Diagnostics

The benchmark result rows now expose CoT execution depth per step. For lookup use:

```text
lookup_cot_requested_iterations
lookup_cot_attempted_iterations
lookup_cot_executed_iterations
lookup_cot_early_stop
lookup_cot_stop_reason
lookup_cot_final_similarity
lookup_cot_similarities
```

For visualization use the same columns with the `vis_cot_` prefix.

Interpretation:

```text
requested_iterations   configured cot_n
attempted_iterations   last iteration the loop attempted, including failed refinement attempts
executed_iterations    completed iterations, including the initial generation
early_stop             true when the loop ended before the requested depth
stop_reason            converged, refinement_error, or requested_depth_reached
```

This lets the final analysis separate the cost of requesting high CoT depth from the cost actually paid when convergence stops the loop early.

## Metrics

Primary:

```text
lookup phase:         csv_iou
visualization phase:  vis_score
elapsed_sec
energy_consumed_kwh
lookup_llm_time_sec
lookup_llm_energy_kwh
vis_llm_time_sec
vis_llm_energy_kwh
```

Derived:

```text
delta_quality_vs_cot_n1
delta_energy_vs_cot_n1
quality_gain_per_extra_executed_iteration
early_stop_rate_by_requested_depth
executed_depth_distribution
```

## Thesis Objective

This test answers whether CoT refinement is a useful compute knob for agent steps with verifiable intermediate artifacts, and whether thinking versus non-thinking models use that knob differently.

## Completeness Check

Dataset:

```text
evaluation/benchmark_dataset_gemma4_thesis_10.json
```

Expected rows:

```text
2 models x 3 repeats x 8 configs x 10 prompts = 480 benchmark rows
```

## Code Readiness

Ready to run with the final manifest runner.

Manifests:

```text
gemma4:e4b             evaluation/thesis_final_run/thesis_test04_gemma4_e4b_compute_expansion.yaml
mistral-small3.2:24b   evaluation/thesis_final_run/thesis_test04_mistral_small32_compute_expansion.yaml
```

The benchmark CSVs include CoT early-stop diagnostics in each `config_*/benchmark_results.csv` and in aggregated `detail.csv`.

## Remote Docker Commands

Run from the repository root on the remote machine. Build the image once after pulling the branch:

```bash
docker build -t data-agent .
```

Start Ollama once outside Docker:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the tested models:

```bash
ollama pull gemma4:e4b
ollama pull mistral-small3.2:24b
```

Run three measured repeats sequentially on one GPU:

```bash
for REP in 01 02 03; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_final_run/thesis_test04_gemma4_e4b_compute_expansion.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/04_compute_expansion_n_and_cot/gemma4_e4b/rep${REP} \
    --repetition ${REP}/03 \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_final_run/thesis_test04_mistral_small32_compute_expansion.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/04_compute_expansion_n_and_cot/mistral_small32_24b/rep${REP} \
    --repetition ${REP}/03 \
    --resume
done
```

Run the two models sequentially if you want the energy comparison to stay interpretable.
