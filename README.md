## Sales Data Agent

An LLM-powered agent that queries a local parquet dataset with DuckDB, analyzes the results, and optionally generates visualization code. It uses a LangGraph workflow and supports **multiple LLM providers**:

- **OpenAI (default)**: e.g. `gpt-4o-mini`
- **Ollama (local)**: e.g. `llama3.2:3b`
- **Anthropic (Claude models)**: e.g. `anthropic:claude-3-5-sonnet-latest`

### What it does
- **Lookup**: converts natural language into SQL via the LLM and runs it on DuckDB over the parquet files in `data/`.
- **Analyze**: asks the LLM to summarize/interpret the results.
- **Visualize**: requests a chart configuration and generates matplotlib code to plot it.

Each step supports **best-of-N self-consistency**: the agent runs each step N times with a sampling schedule (temperature, top-p, or top-k) and selects the best candidate via an evaluator (consensus CSV IoU, LLM judge, or visualization judge).

---

## Requirements
- Python 3.10+
- An API key for OpenAI **or** Ollama running locally (`https://ollama.com`) with a model pulled
- Parquet file(s) present in `data/`
- Docker with the NVIDIA container runtime if you want to reproduce the GPU benchmark/test workflow

Install Python deps (from the project root):
```bash
pip install -r requirements.txt
```

For Docker/GPU benchmark runs:

```bash
docker --version
nvidia-smi
```

If `nvidia-smi` fails on the host, Docker GPU runs will fail too. Fix the driver/runtime first, then rebuild or rerun the container.

---

## Project layout
```
DataAgent-1/
  Agent/
    data_agent.py          # SalesDataAgent class and LangGraph wiring
    config.py              # AgentConfig (per-step hyperparameters)
    steps.py               # Individual step implementations
    cache.py               # Run caching
    schema.py              # Multi-table schema support
    parameter_provider.py  # Interactive terminal parameter overrides
    tracing.py             # Phoenix/OpenTelemetry tracing helpers
    utils.py
  config/
    run_config.yaml        # Main single-run configuration file
    search_space.yaml      # (used by bulk runner)
    *.yaml                 # Additional config templates
  data/                    # Parquet files + per-table schema YAML files
  evaluation/
    bulk_runner.py         # Bulk ablation study runner
    run_manifest_benchmark.py
                           # Deterministic DOE/manifest benchmark runner
    run_benchmark.py       # Single benchmark run over a dataset
    benchmark_dataset.json # Benchmark prompts + ground truth
    benchmark_dataset_gemma4_thesis_*.json
                           # Fixed thesis benchmark subsets
    search_space.yaml      # Hyperparameter search space for bulk runs
    thesis_final_run/*.yaml
                           # Final thesis experiment manifests
    run_thesis_final_tests_*_docker.sh
                           # Sequential Docker launchers for thesis runs
    aggregate_results.py   # Post-run aggregation utilities
  runs/                    # Output directory for run artifacts
  run_agent.py             # Entry point for single runs
  requirements.txt
```

---

## Docker workflow used for thesis tests

The thesis experiments were run through Docker so the same code, dependencies, datasets, and configuration files are used on the remote GPU machine. The important rule is to mount the host `runs/` directory into `/app/runs`; otherwise results disappear when the container exits.

### 1. Build the image

From the repository root:

```bash
docker build -t data-agent .
```

Rebuild after pulling code changes, because the Docker image contains a copy of `Agent/`, `evaluation/`, `data/`, `config/`, and `run_agent.py` from build time.

### 2. Start Ollama on the host

In a long-lived terminal or service on the host:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the local models you plan to test:

```bash
ollama pull gemma4:e4b
ollama pull gemma4:26b
ollama pull mistral-small3.2:24b
```

The Docker commands use `--network=host` and pass the client endpoint as:

```bash
-e OLLAMA_HOST=http://localhost:11434
```

### 3. Persist results outside the container

All benchmark commands should mount a host output directory:

```bash
-v "$(pwd)/runs:/app/runs"
```

Use `--save-dir runs/...` inside the command. Because `/app/runs` is mounted, that path is written to the host repository's `runs/` folder.

### 4. Run one deterministic manifest

`evaluation/run_manifest_benchmark.py` is the main deterministic DOE runner. It reads a YAML manifest, materializes the named configurations, runs every config on a benchmark dataset, and writes CSV summaries, per-config outputs, and plots.

Example smoke run:

```bash
docker run --rm --gpus '"device=0"' --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset_gemma4_thesis_10.json \
  evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
  --provider ollama \
  --model gemma4:e4b \
  --gt-judge-provider openai \
  --gt-judge-model gpt-5.4 \
  --save-dir runs/smoke/test03/gemma4_e4b/rep01 \
  --max-configs 2 \
  --max-prompts 1 \
  --no-codecarbon \
  --repetition 01/03 \
  --resume
```

Full measured run pattern:

```bash
docker run --rm --gpus '"device=0"' --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  <dataset.json> \
  <manifest.yaml> \
  --provider ollama \
  --model <tested-model> \
  --gt-judge-provider openai \
  --gt-judge-model gpt-5.4 \
  --save-dir runs/thesis_tests_final/<test>/<model_slug>/rep01 \
  --repetition 01/03 \
  --resume
```

`--resume` skips configs that already have `benchmark_results.csv`, so an interrupted run can be restarted with the same command.

### 5. Run the final thesis batch sequentially

To avoid noisy energy attribution, run measured experiments sequentially on one GPU instead of launching multiple containers at the same time.

Final Tests 01-04:

```bash
export OPENAI_API_KEY="..."
bash evaluation/run_thesis_final_tests_01_04_docker.sh
```

Tests 03-04 only:

```bash
bash evaluation/run_thesis_final_tests_03_04_docker.sh
```

Useful preflight:

```bash
bash evaluation/run_thesis_final_tests_01_04_docker.sh --dry-run
bash evaluation/run_thesis_final_tests_01_04_docker.sh --help
```

The scripts default to:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_AGENT_IMAGE` | `data-agent` | Docker image name |
| `GPU_DEVICE` | `0` | GPU passed to `--gpus device=...` |
| `OLLAMA_HOST` | `http://localhost:11434` | Host Ollama endpoint used by the container |
| `RUNS_HOST_DIR` | `<repo>/runs` | Host folder mounted to `/app/runs` |
| `GT_JUDGE_PROVIDER` | `openai` | Provider for ground-truth text/visual scoring |
| `GT_JUDGE_MODEL` | `gpt-5.4` | Model for ground-truth text/visual scoring |

Override them inline when needed:

```bash
GPU_DEVICE=1 RUNS_HOST_DIR=/mnt/experiments/runs \
  bash evaluation/run_thesis_final_tests_01_04_docker.sh
```

### Output layout

The expected measured-run layout is:

```text
runs/thesis_tests_final/<test_id_and_name>/<model_slug>/repXX
```

Each completed manifest run writes:

```text
detail.csv
summary.csv
thesis_summary.csv
configs_sampled.json
plots/
config_0000/benchmark_results.csv
config_0001/benchmark_results.csv
...
```

The most important metrics are:

```text
csv_iou
text_score
vis_score
elapsed_sec
energy_consumed_kwh
gpu_energy_kwh
emissions_kg_co2
```

For thesis reports, count expected but missing GT score slots as failed accuracy, not as missing data.

---

## Running the agent

There are three common ways to execute the project:

- **Single run** via `run_agent.py`.
- **Deterministic manifest benchmark** via `evaluation/run_manifest_benchmark.py`.
- **Exploratory random bulk run** via `evaluation/bulk_runner.py`.

### Single run — `run_agent.py`

All parameters are configured in a YAML file (default: `config/run_config.yaml`). Edit the config, then run:

```bash
python run_agent.py                        # uses config/run_config.yaml
python run_agent.py config/my_config.yaml  # custom config path
```

On first launch the script shows the loaded config and optionally prompts for interactive overrides (set `interactive_config: true` in the YAML to enable).

#### Key run parameters (`run:` section in YAML)

| Parameter | Description |
|---|---|
| `prompt` | Natural language query |
| `visualization_goal` | Chart description (empty to skip) |
| `agent_mode` | `lookup_only` \| `analysis` \| `full` |
| `run_id` | Stable ID for caching / reproducibility (`null` = auto) |
| `save_dir` | Root output directory |
| `save_execution_artifacts` | Write `run_metadata.json` + `result.json` per run |
| `enable_codecarbon` | Enable CodeCarbon energy tracking |
| `reuse_from` | Run ID to reuse cached intermediate results from |
| `step_overrides` | Temporary per-step hyperparameter overrides |

#### Key agent parameters (`agent:` section in YAML)

| Parameter | Description |
|---|---|
| `model` | LLM model name (e.g. `gpt-4o-mini`, `llama3.2:3b`) |
| `provider` | `openai` or `ollama` |
| `ollama_url` | Ollama server URL (ignored for openai) |
| `openai_api_key` | `null` = read from `OPENAI_API_KEY` env var |

#### Per-step configuration (`steps:` section in YAML)

Each step (`decide_tool`, `lookup_sales_data`, `analyzing_data`, `create_visualization`) supports:

| Parameter | Description |
|---|---|
| `n` | Number of best-of-N candidates |
| `temp_min` / `temp_max` | Temperature range across candidates |
| `top_p_min` / `top_p_max` | Top-p range (alternative BoN axis) |
| `top_k_min` / `top_k_max` | Top-k range (Ollama only; alternative BoN axis) |
| `cot_n` | Chain-of-thought refinement iterations |
| `max_tokens` | Max tokens per generation |
| `use_cache` | Enable result caching |
| `eval` | Evaluator: `default` (consensus/LLM judge) or `none` |
| `enabled` | Enable/disable the step entirely |

#### Ground truth (optional)

Provide a ground-truth block in the YAML to log evaluation scores alongside each step (scores are never used to steer selection — only for tracking):

```yaml
ground_truth:
  csv_path: "path/to/gt_data.csv"
  analysis_text: "expected analysis text"
  vis_config: null
  vis_code: null
```

---

### Deterministic manifest benchmark — `evaluation/run_manifest_benchmark.py`

Use this runner for thesis-style DOE experiments. It is the deterministic counterpart to `bulk_runner.py`: configs are named explicitly in YAML instead of sampled randomly. This makes runs reproducible and easier to describe in the thesis.

Basic local command:

```bash
python evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset_gemma4_thesis_10.json \
  evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
  --provider ollama \
  --model gemma4:e4b \
  --gt-judge-provider openai \
  --gt-judge-model gpt-5.4 \
  --save-dir runs/local_manifest_test \
  --resume
```

Useful flags:

| Flag | Description |
|---|---|
| `dataset` | Benchmark dataset JSON with prompts and ground truth |
| `manifest` | Deterministic DOE YAML file |
| `--provider`, `--model` | Tested agent provider/model |
| `--gt-judge-provider`, `--gt-judge-model` | Judge used for final GT text and visual scores |
| `--no-gt-judge-provider`, `--no-gt-judge-model` | Optional judge for no-GT selection; defaults to the tested agent |
| `--ollama-url` | Ollama endpoint; defaults to `OLLAMA_HOST` or `http://localhost:11434` |
| `--save-dir` | Output folder |
| `--max-configs`, `--max-prompts` | Smoke-test limits |
| `--no-codecarbon` | Disable energy tracking for quick smoke runs |
| `--repetition` | Label shown in logs, for example `01/03` |
| `--resume` | Skip already completed configs |

Manifest files usually contain:

```yaml
defaults:
  lookup_sales_data:
    temp_min: 1.0
    temp_max: 1.0
    top_p_min: 0.95
    top_p_max: 0.95
    top_k_min: 64
    top_k_max: 64
    max_tokens: 5000

configs:
  - name: lookup_max_tokens_3000
    vary_step: lookup_sales_data
    axis: max_tokens_3000
    steps:
      lookup_sales_data:
        max_tokens: 3000
```

The runner also supports compact `one_factor_sweeps` in manifests; these are expanded into named configs before execution.

#### Judge separation

The benchmark runner separates two judge paths:

- GT judge: used only for ground-truth `text_score` and `vis_score`.
- No-GT judge: used by real-run candidate selection/diagnostics when no GT is available.

For final thesis runs, use the same OpenAI GT judge across all tested local models:

```bash
--gt-judge-provider openai --gt-judge-model gpt-5.4
```

When `--no-gt-judge-*` is omitted, no-GT judging follows the tested model:

```bash
--provider ollama --model <tested-model>
```

This keeps accuracy scoring comparable while preserving the local agent behavior used in real non-GT runs. The older `--judge-provider` and `--judge-model` flags are still accepted as GT-judge aliases.

---

### Bulk run — `evaluation/bulk_runner.py`

The bulk runner performs a **3-phase ablation study** over the hyperparameter search space defined in `evaluation/search_space.yaml`. In each phase only one step's hyperparameters are varied across N randomly sampled configurations; the other steps are kept at their defaults. Results are aggregated automatically at the end of each phase.

```bash
# Validation run (1 config per phase, no think time)
python evaluation/bulk_runner.py \
    evaluation/benchmark_dataset.json \
    evaluation/search_space.yaml \
    --n-configs 1 --think-time 0 \
    --save-dir runs/bulk_results/validation

# Full 50+50+50 run
python evaluation/bulk_runner.py \
    evaluation/benchmark_dataset.json \
    evaluation/search_space.yaml \
    --n-configs 50 --think-time 5.0 \
    --save-dir runs/bulk_results/full_run

# Resume or run only one specific phase
python evaluation/bulk_runner.py ... --vary-step lookup_sales_data --resume
```

Results (per-config JSON + aggregated CSV/XLSX) are saved under `--save-dir`.

---

## LLM provider configuration

Set the provider and model in the `agent:` block of the YAML config. API keys can be passed directly or via environment variables:

### Per-step model overrides
You can keep a global default model/provider under `agent`, then override individual workflow steps under `steps`.

```yaml
agent:
  provider: "openai"
  model: "gpt-4o-mini"

steps:
  decide_tool:
    provider: "openai"
    model: "gpt-4.1-mini"

  lookup_sales_data:
    provider: "ollama"
    model: "llama3.2:3b"
    ollama_url: "http://localhost:11434"
```

If a step-level `provider`, `model`, or `ollama_url` is `null` or omitted, the agent-level value is used.

Environment variables (PowerShell):
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:ANTHROPIC_API_KEY="YOUR_KEY"
$env:OLLAMA_HOST="http://localhost:11434"
```

Environment variables (Bash):
```bash
export OPENAI_API_KEY="YOUR_KEY"
export ANTHROPIC_API_KEY="YOUR_KEY"
export OLLAMA_HOST="http://localhost:11434"
```

### Using Ollama locally

1. Install Ollama from `https://ollama.com/download`.
2. Pull a model: `ollama pull llama3.2:3b`
3. Start the server: `ollama serve`
4. Set `provider: "ollama"` and `model: "llama3.2:3b"` in the config YAML.

---

## Tracing with Phoenix (optional)

Enable OpenInference/Phoenix tracing to visualize agent runs. Configure in the `tracing:` block of the YAML:

```yaml
tracing:
  enabled: true
  phoenix_endpoint: "http://localhost:6006/v1/traces"
  phoenix_api_key: null   # required for Phoenix Cloud
  project_name: "evaluating-agent"
```

Install tracing dependencies:
```bash
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-api
```

Start Phoenix locally:
```bash
phoenix serve
```

Open the UI at `http://localhost:6006`. Top-level spans: `AgentRun`, `tool_choice`, `sql_query_exec`, `data_analysis`, `gen_visualization`.

---

## Energy and emissions (CodeCarbon)

Set `enable_codecarbon: true` in the `run:` block of the YAML config. Energy usage and CO₂ emissions are measured per-LLM-call and saved in `run_metadata.json` alongside each run's artifacts.

```bash
# View the Carbonboard dashboard (optional)
carbonboard --filepath "codecarbon/emissions.csv" --port 8050
```

---

## High-level flow

1. **Decide tool** (LLM): choose lookup → analyze → visualize → end.
2. **Lookup** (DuckDB): parquet → temp table → LLM SQL → query → text table in state.
3. **Analyze** (LLM): summarize / answer with reference to the result data.
4. **Visualize** (LLM): emit compact config → generate matplotlib code to plot.

Each step runs best-of-N candidates with a sampling schedule and selects the best via an evaluator. The agent exposes a single `run(prompt, ...)` entry point and returns the final state with an ordered `answer` list.
