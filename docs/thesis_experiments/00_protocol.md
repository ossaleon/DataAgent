# Thesis Experiment Protocol

## Objective

Define the common rules for the next thesis experiment set. The core design keeps the bulk-runner logic: change one agent step at a time and keep the other steps fixed at the model baseline.

## Agent Steps

```text
lookup_sales_data      SQL generation and data extraction
analyzing_data         textual analysis of the extracted data
create_visualization   visualization config and code generation
```

Primary experiments must be step-isolated. Whole-agent experiments are secondary checks only.

## Model Set

Thinking / reasoning models:

```text
gemma4:e4b
gemma4:26b
nemotron-3-nano:4b
```

Non-thinking / instruction-oriented controls:

```text
mistral-small3.2:24b
```

## Baseline Defaults

Use model-card defaults where available and controlled defaults otherwise.

```text
Gemma 4 models:
  temperature=1.0
  top_p=0.95
  top_k=64

Nemotron models:
  temperature=1.0
  top_p=1.0
  top_k=64

Mistral Small:
  temperature=0.15
  top_p=0.95
  top_k=64
```

Baseline token budgets:

```text
lookup_sales_data: 5000
analyzing_data: 7000
create_visualization: 7000
```

## Metrics

Accuracy:

```text
csv_iou
text_score
vis_score
quality_mean
SQL failure rate
parse/render failure rate
timeout rate
```

Cost:

```text
elapsed_sec
energy_consumed_kwh
gpu_energy_kwh
lookup_llm_energy_kwh
analyzing_llm_energy_kwh
vis_llm_energy_kwh
```

Derived thesis metrics:

```text
delta_quality_vs_step_baseline
delta_energy_vs_step_baseline
accuracy_gain_per_kwh
accuracy_gain_per_second
```

## Measurement Rules

- Run measured experiments sequentially on one GPU.
- Use self-judging per model for `text_score` and `vis_score`.
- Treat `csv_iou` as the strongest cross-model metric because it is deterministic.
- Report both mean and median latency and energy.
- Apply a hard per-prompt timeout in final implementations before scaling reasoning models.
- Store results outside Docker through the `runs/` volume.

## Result Directory Layout

Every measured run should use the same folder structure:

```text
runs/thesis_tests/<test_id_and_name>/<model_slug>/repXX
```

Example:

```text
runs/thesis_tests/02_agent_step_parameter_sensitivity/gemma4_e4b/rep01
```

This keeps one folder per test, one subfolder per model, and one subfolder per repeat. The Docker volume mount should always map the host `runs/` directory to `/app/runs`:

```bash
-v "$(pwd)/runs:/app/runs"
```

## Completeness Checks

Every experiment document must specify:

```text
models
dataset
repeats
configs per model
prompts per config
varied step
fixed steps
primary metrics
expected thesis claim
```
