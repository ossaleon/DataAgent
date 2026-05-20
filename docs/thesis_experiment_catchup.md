# Thesis Experiment Catch-Up

## Summary

This note summarizes the latest benchmark/evaluation changes for the thesis experiments. The main updates are:

- The ground-truth CSV score was redesigned from brittle exact IoU into a graded table-similarity score.
- A deterministic DOE runner was added to replace random bulk sampling while preserving step-by-step ablation.
- New DOE manifests were added for Gemma 4, Ministral 3, and Nemotron 3 Nano.

The current working branch for these changes is:

```bash
gemma4_thesis_doe
```

---

## 1. Improved CSV IoU / Table Similarity

### What Changed

The previous `csv_iou` score was too strict. If the generated table differed from the ground truth by column aliases, column order, boolean labels, or long-vs-wide layout, the score often collapsed to `0.0` even when the answer was semantically correct.

Example problem:

```text
GT:    is_organic = Organic / Non-Organic
Model: is_organic = true / false
```

The old scorer treated this as wrong. The new scorer gives high credit when the rows and values represent the same answer.

### New Behavior

The new GT-only scorer is:

```python
compare_dataframes_similarity_gt(...)
```

It is used by:

```python
make_csv_evaluator_gt(...)
```

Important behavior:

- Failed SQL, missing dataframe, or empty results still score `0.0`.
- Exact tables still score `1.0`.
- Row order and column order no longer matter.
- Column aliases can still receive high credit if values align.
- Swapped numeric columns are aligned by value similarity.
- Boolean/category labels are normalized, for example `true/false`, `1/0`, `Organic/Non-Organic`, and `Promo/Non-Promo`.
- Long-vs-wide binary grouped tables are handled, for example `month, promo_flag, revenue` vs `month, promo_revenue, non_promo_revenue`.

### Files

```text
Agent/utils.py
evaluation/test_iou_comprehensive.py
```

Validation:

```bash
./venv/bin/python evaluation/test_iou_comprehensive.py
```

Expected result:

```text
72/72 passed
```

---

## 2. Deterministic DOE Runner

### Why We Added It

The old `bulk_runner.py` had the right high-level idea:

```text
vary one agent step while keeping the others fixed
```

But it sampled parameters randomly. That makes thesis results harder to explain because each run is partly determined by chance.

The new runner keeps the useful bulk-run logic but removes random sampling.

### New Runner

```text
evaluation/run_manifest_benchmark.py
```

It reads a YAML manifest with named configs, materializes `AgentConfig` objects, runs the benchmark, and writes thesis-ready outputs.

### Fixed 10-Sample Dataset

The thesis DOE uses a smaller fixed benchmark subset:

```text
evaluation/benchmark_dataset_gemma4_thesis_10.json
```

It uses original benchmark case IDs:

```text
2, 5, 0, 1, 11, 12, 14, 16, 17, 19
```

This preserves a balanced difficulty mix and includes known edge cases such as promo splits, organic labels, swapped numeric columns, and YoY growth.

### Output Files

Each run writes:

```text
detail.csv
summary.csv
thesis_summary.csv
configs_sampled.json
plots/quality_by_config.png
plots/latency_energy_by_config.png
plots/quality_vs_energy.png
config_0000/benchmark_results.csv
...
```

Primary metrics:

```text
csv_iou
text_score
vis_score
elapsed_sec
energy_consumed_kwh
gpu_energy_kwh
emissions_kg_co2
```

Note: `csv_iou` is kept as the column name for continuity, but it now means graded GT table similarity.

---

## 3. DOE Manifests

Three model-specific manifests exist:

```text
evaluation/gemma4_thesis_doe.yaml
evaluation/ministral3_thesis_doe.yaml
evaluation/nemotron3_nano_thesis_doe.yaml
```

All three use the same deterministic step-ablation structure:

```text
4 lookup configs
4 analysis configs
4 visualization configs
```

Each config varies only one step. The other two steps stay fixed at that model's baseline parameters.

### Gemma 4

Manifest:

```text
evaluation/gemma4_thesis_doe.yaml
```

Baseline:

```text
temperature = 1.0
top_p = 0.95
top_k = 64
```

This follows Gemma 4's recommended sampling setup.

### Ministral 3 14B

Manifest:

```text
evaluation/ministral3_thesis_doe.yaml
```

Baseline:

```text
temperature = 0.15
top_p = 0.95
top_k = 64
```

The low temperature follows the Ollama model parameters for `ministral-3:14b`.

### Nemotron 3 Nano 4B

Manifest:

```text
evaluation/nemotron3_nano_thesis_doe.yaml
```

Baseline:

```text
temperature = 0.70
top_p = 0.95
top_k = 64
```

Ollama describes this as a reasoning model but does not expose an explicit sampling recommendation, so we use a conservative reasoning-model baseline.

---

## 4. Remote Setup

On the remote machine:

```bash
git fetch origin
git switch gemma4_thesis_doe
git pull
docker build -t data-agent .
```

Start Ollama once:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the models:

```bash
ollama pull gemma4:e4b
ollama pull ministral-3:14b
ollama pull nemotron-3-nano:4b
```

Ollama serves all models from the same API port. The Docker commands just select a different `--model`.

---

## 5. Run Commands

### Gemma 4 E4B

```bash
docker run --rm --gpus all --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset_gemma4_thesis_10.json \
  evaluation/gemma4_thesis_doe.yaml \
  --provider ollama \
  --model gemma4:e4b \
  --judge-provider ollama \
  --judge-model gemma4:e4b \
  --save-dir runs/thesis_doe_gemma4_e4b
```

### Ministral 3 14B

```bash
docker run --rm --gpus all --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset_gemma4_thesis_10.json \
  evaluation/ministral3_thesis_doe.yaml \
  --provider ollama \
  --model ministral-3:14b \
  --judge-provider ollama \
  --judge-model ministral-3:14b \
  --save-dir runs/thesis_doe_ministral3_14b
```

### Nemotron 3 Nano 4B

```bash
docker run --rm --gpus all --network=host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v "$(pwd)/runs:/app/runs" \
  data-agent \
  evaluation/run_manifest_benchmark.py \
  evaluation/benchmark_dataset_gemma4_thesis_10.json \
  evaluation/nemotron3_nano_thesis_doe.yaml \
  --provider ollama \
  --model nemotron-3-nano:4b \
  --judge-provider ollama \
  --judge-model nemotron-3-nano:4b \
  --save-dir runs/thesis_doe_nemotron3_nano_4b
```

For clean thesis carbon and latency comparisons, run the models sequentially rather than all at once. Parallel runs on the same L40S will make GPU scheduling, latency, and energy attribution noisy.
