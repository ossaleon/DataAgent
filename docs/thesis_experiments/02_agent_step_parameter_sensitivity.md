# Test 02: Agent-Step Parameter Sensitivity

## Objective

Run one DOE that follows the bulk-runner logic: execute three ablation phases, varying one agent step at a time while keeping the other two fixed at the model baseline.

This replaces separate lookup, analysis, and visualization sensitivity tests.

## Hypothesis

Each agent step has different sensitive parameters:

```text
lookup_sales_data      sensitive to SQL-generation sampling and feedback
analyzing_data         sensitive to reasoning/coverage settings
create_visualization   sensitive to structured-output sampling and feedback
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

Use the thesis 10-case subset with at least two repeats. Run measured experiments sequentially on one GPU.

## Models

```text
gemma4:e4b
gemma4:26b
nemotron-3-nano:4b
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

Nemotron:
  temperature=1.0
  top_p=1.0
  top_k=64

Mistral Small:
  temperature=0.15
  top_p=0.95
  top_k=64
```

## Configuration Set

Each phase starts with a phase-specific baseline, then varies only that phase's agent step.

### Phase 1: Lookup Agent Configs

Objective: measure SQL/data extraction behavior.

```text
lookup_baseline
lookup_temperature_low
lookup_temperature_high
lookup_top_p_low
lookup_top_k_low
lookup_top_k_high
lookup_bon_temperature_n2
lookup_bon_temperature_n3
lookup_cot_n2
```

Lookup temperature values:

```text
Gemma 4 models:
  low=0.6
  baseline=1.0
  high=1.3

Nemotron:
  low=0.7
  baseline=1.0
  high=1.15

Mistral Small:
  low=0.05
  baseline=0.15
  high=0.45
```

Lookup sampling values:

```text
top_p_low=0.80
top_k_low=20
top_k_high=128
```

Lookup Best-of-N:

```text
bon_param=temperature
n=2 range: low -> baseline
n=3 range: low -> high
```

Reason: SQL generation may need prompt-dependent exploration. Temperature is the most useful diversity axis for joins, pivots, date handling, and CASE expressions.

Lookup CoT:

```text
cot_n=2
```

Reason: SQL errors are concrete and can often be corrected by a feedback/refinement loop.

Primary lookup metrics:

```text
csv_iou
SQL failure rate
empty result rate
lookup_llm_time_sec
lookup_llm_energy_kwh
```

### Phase 2: Analysis Agent Configs

Objective: measure textual interpretation behavior.

```text
analysis_baseline
analysis_temperature_low
analysis_temperature_high
analysis_top_p_low
analysis_top_k_low
analysis_top_k_high
analysis_bon_temperature_n2
analysis_cot_n2
```

Analysis temperature values use the same model-specific low/baseline/high values as lookup.

Analysis sampling values:

```text
top_p_low=0.80
top_k_low=20
top_k_high=128
```

Analysis Best-of-N:

```text
bon_param=temperature
n=2 range: low -> baseline
```

Reason: analysis candidates differ mostly in reasoning path, coverage, and phrasing.

Analysis CoT:

```text
cot_n=2
```

Reason: included as a secondary signal; thinking models may already internalize this behavior.

Primary analysis metrics:

```text
text_score
judge parse failure rate
analyzing_llm_time_sec
analyzing_llm_energy_kwh
```

### Phase 3: Visualization Agent Configs

Objective: measure chart/config/code robustness.

```text
visualization_baseline
visualization_temperature_low
visualization_temperature_high
visualization_top_p_low
visualization_top_k_low
visualization_top_k_high
visualization_bon_top_k_n2
visualization_cot_n2
```

Visualization temperature values use the same model-specific low/baseline/high values as lookup.

Visualization sampling values:

```text
top_p_low=0.80
top_k_low=20
top_k_high=128
```

Visualization Best-of-N:

```text
bon_param=top_k
n=2 range: 20 -> 128
```

Reason: visualization output is structured. `top_k` provides controlled diversity while reducing the parseability risk of temperature-based diversity.

Visualization CoT:

```text
cot_n=2
```

Reason: chart errors are concrete: wrong axis, wrong chart type, missing group, invalid config, or code mismatch.

Primary visualization metrics:

```text
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
lookup phase:        9 configs
analysis phase:      8 configs
visualization phase: 8 configs
total:              25 configs
```

Expected rows:

```text
models x repeats x 25 configs x 10 prompts
```

## Code Readiness

Ready to run with the current manifest runner.

Manifests:

```text
gemma4:e4b             evaluation/thesis_test02_gemma4_sensitivity.yaml
gemma4:26b             evaluation/thesis_test02_gemma4_sensitivity.yaml
nemotron-3-nano:4b     evaluation/thesis_test02_nemotron3_nano_sensitivity.yaml
mistral-small3.2:24b   evaluation/thesis_test02_mistral_small32_sensitivity.yaml
```

The manifests contain exactly 25 configs per model and preserve the bulk-runner logic: one step varies, the other two steps stay fixed.

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
ollama pull nemotron-3-nano:4b
ollama pull mistral-small3.2:24b
```

Run two measured repeats sequentially on one GPU:

```bash
for REP in 01 02; do
  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_test02_gemma4_sensitivity.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --judge-provider ollama \
    --judge-model gemma4:e4b \
    --save-dir runs/thesis_tests/02_agent_step_parameter_sensitivity/gemma4_e4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_test02_gemma4_sensitivity.yaml \
    --provider ollama \
    --model gemma4:26b \
    --judge-provider ollama \
    --judge-model gemma4:26b \
    --save-dir runs/thesis_tests/02_agent_step_parameter_sensitivity/gemma4_26b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_test02_nemotron3_nano_sensitivity.yaml \
    --provider ollama \
    --model nemotron-3-nano:4b \
    --judge-provider ollama \
    --judge-model nemotron-3-nano:4b \
    --save-dir runs/thesis_tests/02_agent_step_parameter_sensitivity/nemotron3_nano_4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_test02_mistral_small32_sensitivity.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --judge-provider ollama \
    --judge-model mistral-small3.2:24b \
    --save-dir runs/thesis_tests/02_agent_step_parameter_sensitivity/mistral_small32_24b/rep${REP} \
    --resume
done
```
