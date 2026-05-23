# Test 02: Agent-Step Parameter Sensitivity (Final Rerun)

## Objective

Run one DOE that follows the bulk-runner logic: execute three ablation phases, varying one agent step at a time while keeping the other two fixed at the model baseline.

This replaces separate lookup, analysis, and visualization sensitivity tests.

## Hypothesis

Each agent step has different sensitive parameters:

```text
lookup_sales_data      sensitive to SQL-generation sampling and repetition control
analyzing_data         sensitive to reasoning/coverage sampling and repetition control
create_visualization   sensitive to structured-output sampling and code repetition control
```

The experiment should reveal which parameter changes matter for each step, and whether the energy cost is justified by the accuracy gain.

## Test Logic

The test has three phases per model.

Phase 1:

```text
lookup_sales_data      varied
analyzing_data         fixed baseline
create_visualization   fixed baseline
```

Phase 2:

```text
lookup_sales_data      fixed baseline
analyzing_data         varied
create_visualization   fixed baseline
```

Phase 3:

```text
lookup_sales_data      fixed baseline
analyzing_data         fixed baseline
create_visualization   varied
```

Use the thesis subset with three measured repeats. Run measured experiments sequentially on one GPU.

## Models

```text
gemma4:e4b
gemma4:26b
mistral-small3.2:24b
```

## Baseline Configurations

All non-varied steps use model baseline settings:

```text
n=1
cot_n=1
bon_param=temperature
num_beams=1
no_repeat_ngram_size=null
use_cache=false
```

Token budgets:

```text
lookup_sales_data:      max_tokens=5000
analyzing_data:         max_tokens=7000
create_visualization:   max_tokens=7000
```

Sampling baselines:

```text
Gemma 4 models:
  temperature=1.0
  top_p=0.95
  top_k=64
  repeat_penalty=1.1
  repeat_last_n=64

Mistral Small:
  temperature=0.15
  top_p=0.95
  top_k=64
  repeat_penalty=1.1
  repeat_last_n=64
```

## Configuration Set

Each phase starts with one phase-specific baseline, then varies one static LLM parameter on that phase's agent step. Best-of-N and CoT are excluded from this final Test 02 screen. Token caps are also fixed because Test 03 isolates the `max_tokens` question.

The five parameters are:

```text
temperature
top_p
top_k
repeat_penalty
repeat_last_n
```

Each parameter contributes 10 off-baseline configs. The phase baseline is stored once instead of repeating an identical default config for every parameter:

```text
1 baseline + 5 parameters x 10 levels = 51 configs per varied agent
```

### Parameter Ranges

Temperature is model-specific because the Gemma and Mistral baselines differ substantially.

```text
Gemma 4 baseline temperature=1.0
  tested: 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.30

Mistral Small baseline temperature=0.15
  tested: 0.03, 0.05, 0.075, 0.10, 0.125, 0.175, 0.20, 0.25, 0.35, 0.45
```

The other four ranges are shared by the compared models:

```text
top_p baseline=0.95
  tested: 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.98, 1.00

top_k baseline=64
  tested: 16, 20, 24, 32, 40, 48, 56, 80, 96, 128

repeat_penalty baseline=1.1
  tested: 0.90, 0.95, 1.00, 1.03, 1.06, 1.08, 1.12, 1.15, 1.20, 1.30

repeat_last_n baseline=64
  tested: 16, 24, 32, 40, 48, 56, 80, 96, 128, 256
```

The ranges are denser near the baselines and still extend far enough to expose meaningful accuracy or energy movement. They keep output caps and the other sampling controls fixed while one parameter moves.

### Step Metrics

The varied-agent metrics remain step-specific:

```text
lookup_sales_data:
  csv_iou
  SQL failure rate
  empty result rate
  lookup_llm_time_sec
  lookup_llm_energy_kwh

analyzing_data:
  text_score
  judge parse failure rate
  analyzing_llm_time_sec
  analyzing_llm_energy_kwh

create_visualization:
  vis_score
  parse failure rate
  render/equivalence failure rate
  vis_llm_time_sec
  vis_llm_energy_kwh
```

## Cross-Phase Metrics

For every config, also report:

```text
quality_mean
elapsed_sec
energy_consumed_kwh
gpu_energy_kwh
delta_quality_vs_phase_baseline
delta_energy_vs_phase_baseline
accuracy_gain_per_kwh
accuracy_gain_per_second
```

## Thesis Objective

This test is the main source of evidence for parameter sensitivity. It keeps the agent-step behavior interpretable while still measuring downstream effects on the full pipeline.

## Completeness Check

Configs per model:

```text
lookup phase:         51 configs
analysis phase:       51 configs
visualization phase:  51 configs
total:               153 configs
```

Expected rows:

```text
3 models x 3 repeats x 153 configs x 15 prompts = 20655 benchmark rows
```

## Code Readiness

Ready to run with the current manifest runner.

Manifests:

```text
gemma4:e4b             evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml
gemma4:26b             evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml
mistral-small3.2:24b   evaluation/thesis_final_run/thesis_test02_mistral_small32_sensitivity.yaml
```

The manifests materialize exactly 153 configs per model and preserve the bulk-runner logic: one step varies, one parameter moves, and the other two steps stay fixed.

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

Run three measured repeats sequentially on one GPU:

```bash
for REP in 01 02 03; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_15.json \
    evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/02v2_agent_step_parameter_sensitivity/gemma4_e4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_15.json \
    evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml \
    --provider ollama \
    --model gemma4:26b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/02v2_agent_step_parameter_sensitivity/gemma4_26b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_15.json \
    evaluation/thesis_final_run/thesis_test02_mistral_small32_sensitivity.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/02v2_agent_step_parameter_sensitivity/mistral_small32_24b/rep${REP} \
    --resume
done
```
