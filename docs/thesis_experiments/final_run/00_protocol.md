# Thesis Experiment Protocol: Final Rerun

This folder describes the second and final thesis experiment batch. The first-run documents stay in the parent folder as the history of the experiments already executed.

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
- Repeat every final measured test three times.
- Use `--gt-judge-provider` and `--gt-judge-model` for the offline GT judge that produces `text_score` and `vis_score`.
- Keep no-GT analysis and visualization judging on the tested local model unless a test explicitly studies selection behavior. This is the real-run path used without ground truth.
- Treat `csv_iou` as the strongest cross-model metric because it is deterministic.
- Report both mean and median latency and energy.
- Apply a hard per-prompt timeout in final implementations before scaling reasoning models.
- Store results outside Docker through the `runs/` volume.

## Judge Sources

The benchmark runner separates the judge used for final ground-truth scoring from the no-GT judge used by analysis and visualization selection/diagnostics.

Use these flags for a shared OpenAI GT judge:

```text
--gt-judge-provider openai
--gt-judge-model gpt-5.4
```

When no no-GT judge flags are passed, the no-GT evaluator follows the tested agent:

```text
--provider ollama
--model <tested-local-model>
```

This keeps `text_score` and `vis_score` comparable across models without changing the local agent behavior that would exist in a real run. The optional `--no-gt-judge-provider` and `--no-gt-judge-model` flags exist only for tests that intentionally need a different no-GT judge. The older `--judge-provider` and `--judge-model` flags remain accepted as GT-judge aliases.

For Docker runs with an OpenAI GT judge, pass the API key into the container:

```bash
-e OPENAI_API_KEY="$OPENAI_API_KEY"
```

## Result Directory Layout

Every measured run should use the same folder structure:

```text
runs/thesis_tests_final/<test_id_and_name>/<model_slug>/repXX
```

Example:

```text
runs/thesis_tests_final/02v2_agent_step_parameter_sensitivity/gemma4_e4b/rep01
```

This keeps one folder per test, one subfolder per model, and one subfolder per repeat. The Docker volume mount should always map the host `runs/` directory to `/app/runs`:

```bash
-v "$(pwd)/runs:/app/runs"
```

## Sequential Tests 01-04 Runner

Tests 01-04 can be launched as one long sequential Docker batch. The runner starts exactly one benchmark container at a time, preserves the result directory layout from the per-test documents, and passes `--resume` to every run so it can be restarted after an interruption. If a container fails, the batch stops; fix the cause and rerun the same command to continue from the completed result folders.

Test 05 is not included because its final parameter set must be chosen after the results from Tests 01-04.

Before launching the batch, build the Docker image once:

```bash
docker build -t data-agent .
```

Start Ollama once on the host in a long-lived terminal or service:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the tested models and export the OpenAI key for the shared GT judge:

```bash
ollama pull gemma4:e4b
ollama pull gemma4:26b
ollama pull mistral-small3.2:24b
export OPENAI_API_KEY="..."
```

Run the sequential batch from the repository root:

```bash
bash evaluation/run_thesis_final_tests_01_04_docker.sh
```

Useful preflight without running containers:

```bash
bash evaluation/run_thesis_final_tests_01_04_docker.sh --dry-run
```

The default batch uses Docker image `data-agent`, GPU device `0`, host results directory `$(pwd)/runs`, `OLLAMA_HOST=http://localhost:11434`, and GT judge `openai/gpt-5.4`. These can be overridden with environment variables documented by:

```bash
bash evaluation/run_thesis_final_tests_01_04_docker.sh --help
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
