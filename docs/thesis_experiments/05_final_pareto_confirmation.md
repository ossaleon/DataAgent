# Test 05: Final Pareto Confirmation

## Objective

Confirm final model/config choices on a larger benchmark after the step-isolated experiments identify candidate settings.

## Hypothesis

The final thesis should show not only the highest-quality configuration, but also the most energy-efficient configuration that preserves nearly the same quality.

## Test Logic

This test must wait until Tests 01-04 are complete. Its manifests should be generated only after the previous experiments identify:

```text
best_lookup_config
best_analysis_config
best_visualization_config
best_overall_quality_config
best_energy_efficient_config
```

For each model, run these configurations:

```text
baseline
best_lookup_config
best_analysis_config
best_visualization_config
best_overall_quality_config
best_energy_efficient_config
```

Use the full benchmark dataset and three repeats.

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

## Metrics

Primary:

```text
quality_mean
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
accuracy_per_kwh
```

## Thesis Objective

This test produces the final tables and Pareto plots for the thesis discussion.

## Completeness Check

Expected rows:

```text
models x 3 repeats x 6 configs x full dataset size
```

## Code Readiness

Not ready to execute yet by design. The runner is ready, but the final manifests must be written after Tests 01-04 have produced their `thesis_summary.csv` files.

Expected final manifests:

```text
evaluation/thesis_test05_gemma4_e4b_final_pareto.yaml
evaluation/thesis_test05_gemma4_26b_final_pareto.yaml
evaluation/thesis_test05_nemotron3_nano_4b_final_pareto.yaml
evaluation/thesis_test05_mistral_small32_24b_final_pareto.yaml
```

Each final manifest should contain exactly six configs:

```text
baseline
best_lookup_config
best_analysis_config
best_visualization_config
best_overall_quality_config
best_energy_efficient_config
```

## Remote Docker Commands

Run these only after the final manifests above have been created from Tests 01-04.

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
ollama pull nemotron-3-nano:4b
ollama pull mistral-small3.2:24b
```

Run the three final measured repeats sequentially on one GPU:

```bash
for REP in 01 02 03; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_test05_gemma4_e4b_final_pareto.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --judge-provider ollama \
    --judge-model gemma4:e4b \
    --save-dir runs/thesis_tests/05_final_pareto_confirmation/gemma4_e4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_test05_gemma4_26b_final_pareto.yaml \
    --provider ollama \
    --model gemma4:26b \
    --judge-provider ollama \
    --judge-model gemma4:26b \
    --save-dir runs/thesis_tests/05_final_pareto_confirmation/gemma4_26b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_test05_nemotron3_nano_4b_final_pareto.yaml \
    --provider ollama \
    --model nemotron-3-nano:4b \
    --judge-provider ollama \
    --judge-model nemotron-3-nano:4b \
    --save-dir runs/thesis_tests/05_final_pareto_confirmation/nemotron3_nano_4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_test05_mistral_small32_24b_final_pareto.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --judge-provider ollama \
    --judge-model mistral-small3.2:24b \
    --save-dir runs/thesis_tests/05_final_pareto_confirmation/mistral_small32_24b/rep${REP} \
    --resume
done
```
