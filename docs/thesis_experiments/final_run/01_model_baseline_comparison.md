# Test 01: Baseline Model Comparison (Final Rerun)

## Objective

Compare a small thinking model, a larger thinking MoE model, and a larger non-thinking model under baseline agent settings before any parameter sensitivity analysis.

## Hypothesis

The small thinking model should be competitive on reasoning-heavy prompts, but may be less robust than larger models. The larger non-thinking model should provide a size-based accuracy/energy reference. The larger Gemma MoE model should show whether a bigger thinking model can improve accuracy while keeping energy lower than a similarly large dense model would be expected to require.

## Test Logic

No individual agent step is varied.

```text
lookup_sales_data      baseline
analyzing_data         baseline
create_visualization   baseline
```

Run the full benchmark dataset with three repeats per model.

## Models

```text
gemma4:e4b
gemma4:26b
mistral-small3.2:24b
```

Model roles:

```text
gemma4:e4b             small thinking model
gemma4:26b             larger thinking MoE model
mistral-small3.2:24b   larger non-thinking model
```

## Configuration Details

All three agent steps use the same baseline structure:

```text
n=1
cot_n=1
bon_param=temperature
num_beams=1
no_repeat_ngram_size=null
use_cache=false
```

Per-step token budgets:

```text
lookup_sales_data:      max_tokens=5000
analyzing_data:         max_tokens=7000
create_visualization:   max_tokens=7000
```

Per-model sampling defaults:

```text
gemma4:e4b:
  temperature=1.0
  top_p=0.95
  top_k=64

gemma4:26b:
  temperature=1.0
  top_p=0.95
  top_k=64

mistral-small3.2:24b:
  temperature=0.15
  top_p=0.95
  top_k=64
```

No agent-specific parameter is varied in this test. The rerun uses one shared GT judge while no-GT evaluation remains on the tested local model:

```text
--model <tested-model>
--gt-judge-provider openai
--gt-judge-model gpt-5.4
```

Execution manifests:

```text
gemma4:e4b             evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml
gemma4:26b             evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml
mistral-small3.2:24b   evaluation/thesis_final_run/thesis_test01_mistral_small32_baseline.yaml
```

## Metrics

Primary:

```text
csv_iou
text_score
vis_score
quality_mean
elapsed_sec
energy_consumed_kwh
gpu_energy_kwh
```

Secondary:

```text
accuracy_per_kwh
median_elapsed_sec
median_energy_consumed_kwh
timeout_rate
```

## Thesis Objective

This test provides the model-level baseline for discussing whether small thinking models can compete with larger non-thinking models, and whether a larger MoE thinking model offers a better accuracy/energy tradeoff.

## Completeness Check

Expected rows:

```text
3 models x 3 repeats x 1 config x full dataset size
```

## Code Readiness

Ready to run with the current manifest runner.

Manifests:

```text
gemma4:e4b             evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml
gemma4:26b             evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml
mistral-small3.2:24b   evaluation/thesis_final_run/thesis_test01_mistral_small32_baseline.yaml
```

## Remote Docker Commands

Run from the repository root on the remote machine. Build the image once after pulling the branch:

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

Run the three measured repeats sequentially on one GPU:

```bash
for REP in 01 02 03; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/01_model_baseline_comparison/gemma4_e4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml \
    --provider ollama \
    --model gemma4:26b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/01_model_baseline_comparison/gemma4_26b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset.json \
    evaluation/thesis_final_run/thesis_test01_mistral_small32_baseline.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/01_model_baseline_comparison/mistral_small32_24b/rep${REP} \
    --resume
done
```
