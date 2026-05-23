# Test 03: Max Tokens Agent Ladder (Final Rerun)

## Objective

Test whether the impact of `max_tokens` differs between agent steps. Each ladder varies one agent's token cap while the other two agents stay at generous, non-bottleneck token caps.

## Hypothesis

Lower token caps can cause different failures depending on the agent: incomplete SQL, incomplete textual analysis, or invalid visualization code/config. The sensitivity should be strongest for the steps that use more reasoning or produce longer structured outputs. Higher caps should reduce these failures, while energy cost should remain modest unless the model actually uses the extra budget.

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

## Baseline Defaults

All non-varied steps stay at the tested model's baseline:

```text
n=1
cot_n=1
bon_param=temperature
num_beams=1
no_repeat_ngram_size=null
use_cache=false
```

Sampling baselines:

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

## Test Logic

Run three separate step-isolated ladders. In each ladder, one agent's `max_tokens` is varied and the other two agents are fixed at generous values so they do not become the bottleneck.

Lookup ladder:

```text
lookup_sales_data      max_tokens varied
analyzing_data         fixed generous max_tokens=10000
create_visualization   fixed generous max_tokens=10000
```

Analysis ladder:

```text
lookup_sales_data      fixed generous max_tokens=8000
analyzing_data         max_tokens varied
create_visualization   fixed generous max_tokens=10000
```

Visualization ladder:

```text
lookup_sales_data      fixed generous max_tokens=8000
analyzing_data         fixed generous max_tokens=10000
create_visualization   max_tokens varied
```

All non-token sampling parameters remain at the tested model's baseline.

## Token Levels

Lookup:

```text
2500
3000
5000 baseline
8000
```

Analysis:

```text
4000
5000
7000 baseline
10000
```

Visualization:

```text
4000
5000
7000 baseline
10000
```

Generous fixed values:

```text
lookup_sales_data:      8000
analyzing_data:         10000
create_visualization:   10000
```

The varied agent may also take the generous value as the top rung of its ladder.

## Metrics

Primary:

```text
step-specific score for the varied agent
quality_mean
SQL failure rate
parse/render failure rate
timeout rate
elapsed_sec
energy_consumed_kwh
step_llm_energy_kwh
```

Derived:

```text
delta_quality_per_extra_1000_tokens
delta_energy_per_extra_1000_tokens
failure_rate_reduction
agent_sensitivity_rank
```

## Thesis Objective

This test supports or refutes the practical recommendation that agent systems should use generous token caps, and identifies whether lookup, analysis, or visualization is most sensitive to token-budget restrictions.

## Completeness Check

Expected rows:

```text
3 models x 3 repeats x 12 token configs x 10 prompts = 1080 benchmark rows
```

## Code Readiness

Ready to run with the current manifest runner.

Manifests:

```text
gemma4:e4b             evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml
gemma4:26b             evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml
mistral-small3.2:24b   evaluation/thesis_final_run/thesis_test03_mistral_small32_max_tokens.yaml
```

The manifests contain exactly 12 configs per model: four lookup token levels, four analysis token levels, and four visualization token levels.

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
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
    --provider ollama \
    --model gemma4:e4b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/03_max_tokens_agent_ladder/gemma4_e4b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
    --provider ollama \
    --model gemma4:26b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/03_max_tokens_agent_ladder/gemma4_26b/rep${REP} \
    --resume

  docker run --rm --gpus '"device=0"' --network=host \
    -e OLLAMA_HOST=http://localhost:11434 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/runs:/app/runs" \
    data-agent \
    evaluation/run_manifest_benchmark.py \
    evaluation/benchmark_dataset_gemma4_thesis_10.json \
    evaluation/thesis_final_run/thesis_test03_mistral_small32_max_tokens.yaml \
    --provider ollama \
    --model mistral-small3.2:24b \
    --gt-judge-provider openai \
    --gt-judge-model gpt-5.4 \
    --save-dir runs/thesis_tests_final/03_max_tokens_agent_ladder/mistral_small32_24b/rep${REP} \
    --resume
done
```
