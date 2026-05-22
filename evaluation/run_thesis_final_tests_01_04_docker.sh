#!/usr/bin/env bash
set -euo pipefail

# Run the final thesis experiment batch sequentially through Docker.
# Test 05 is intentionally excluded because it depends on Tests 01-04.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DATA_AGENT_IMAGE:-data-agent}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OLLAMA_ENDPOINT="${OLLAMA_HOST:-http://localhost:11434}"
RUNS_HOST_DIR="${RUNS_HOST_DIR:-${ROOT_DIR}/runs}"
GT_JUDGE_PROVIDER="${GT_JUDGE_PROVIDER:-openai}"
GT_JUDGE_MODEL="${GT_JUDGE_MODEL:-gpt-5.4}"
REPEATS=(01 02 03)
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/run_thesis_final_tests_01_04_docker.sh [--dry-run]

Runs final thesis Tests 01-04 sequentially with Docker and --resume.
Test 05 is intentionally not included.

Environment overrides:
  DATA_AGENT_IMAGE       Docker image name, default: data-agent
  GPU_DEVICE             Docker GPU device passed to --gpus, default: 0
  OLLAMA_HOST            Host Ollama endpoint, default: http://localhost:11434
  RUNS_HOST_DIR          Host results directory, default: <repo>/runs
  GT_JUDGE_PROVIDER      Ground-truth judge provider, default: openai
  GT_JUDGE_MODEL         Ground-truth judge model, default: gpt-5.4
  OPENAI_API_KEY         Required when GT_JUDGE_PROVIDER=openai
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_file() {
  if [[ ! -f "${ROOT_DIR}/$1" ]]; then
    printf 'Missing required repo file: %s\n' "$1" >&2
    exit 1
  fi
}

require_file Dockerfile
require_file evaluation/run_manifest_benchmark.py
require_file evaluation/benchmark_dataset.json
require_file evaluation/benchmark_dataset_gemma4_thesis_15.json
require_file evaluation/benchmark_dataset_gemma4_thesis_10.json
require_file evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml
require_file evaluation/thesis_final_run/thesis_test01_mistral_small32_baseline.yaml
require_file evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml
require_file evaluation/thesis_final_run/thesis_test02_mistral_small32_sensitivity.yaml
require_file evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml
require_file evaluation/thesis_final_run/thesis_test03_nemotron3_nano_max_tokens.yaml
require_file evaluation/thesis_final_run/thesis_test03_mistral_small32_max_tokens.yaml
require_file evaluation/thesis_final_run/thesis_test04_gemma4_e4b_compute_expansion.yaml
require_file evaluation/thesis_final_run/thesis_test04_mistral_small32_compute_expansion.yaml

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v docker >/dev/null 2>&1 || {
    printf 'docker is required to run the final thesis batch.\n' >&2
    exit 1
  }
  docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || {
    printf 'Docker image %s was not found. Build it first with: docker build -t %s .\n' "$IMAGE_NAME" "$IMAGE_NAME" >&2
    exit 1
  }
  if [[ "$GT_JUDGE_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    printf 'OPENAI_API_KEY is required for the OpenAI GT judge.\n' >&2
    exit 1
  fi
  mkdir -p "$RUNS_HOST_DIR"
fi

run_manifest() {
  local test_dir="$1"
  local model_slug="$2"
  local dataset="$3"
  local manifest="$4"
  local model="$5"
  local rep="$6"
  local save_dir="runs/thesis_tests_final/${test_dir}/${model_slug}/rep${rep}"

  printf '\n%s\n' "================================================================================"
  printf 'Final thesis run: test=%s model=%s repeat=%s/03\n' "$test_dir" "$model" "$rep"
  printf 'dataset=%s\nmanifest=%s\nsave_dir=%s\n' "$dataset" "$manifest" "$save_dir"
  printf '%s\n' "================================================================================"

  local -a cmd=(
    docker run --rm
    --gpus "device=${GPU_DEVICE}"
    --network=host
    -e "OLLAMA_HOST=${OLLAMA_ENDPOINT}"
    -e OPENAI_API_KEY
    -v "${RUNS_HOST_DIR}:/app/runs"
    "$IMAGE_NAME"
    evaluation/run_manifest_benchmark.py
    "$dataset"
    "$manifest"
    --provider ollama
    --model "$model"
    --gt-judge-provider "$GT_JUDGE_PROVIDER"
    --gt-judge-model "$GT_JUDGE_MODEL"
    --save-dir "$save_dir"
    --repetition "${rep}/03"
    --resume
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return
  fi

  "${cmd[@]}"
}

run_test01() {
  local rep
  for rep in "${REPEATS[@]}"; do
    run_manifest \
      01_model_baseline_comparison \
      gemma4_e4b \
      evaluation/benchmark_dataset.json \
      evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml \
      gemma4:e4b \
      "$rep"
    run_manifest \
      01_model_baseline_comparison \
      gemma4_26b \
      evaluation/benchmark_dataset.json \
      evaluation/thesis_final_run/thesis_test01_gemma4_baseline.yaml \
      gemma4:26b \
      "$rep"
    run_manifest \
      01_model_baseline_comparison \
      mistral_small32_24b \
      evaluation/benchmark_dataset.json \
      evaluation/thesis_final_run/thesis_test01_mistral_small32_baseline.yaml \
      mistral-small3.2:24b \
      "$rep"
  done
}

run_test02() {
  local rep
  for rep in "${REPEATS[@]}"; do
    run_manifest \
      02v2_agent_step_parameter_sensitivity \
      gemma4_e4b \
      evaluation/benchmark_dataset_gemma4_thesis_15.json \
      evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml \
      gemma4:e4b \
      "$rep"
    run_manifest \
      02v2_agent_step_parameter_sensitivity \
      gemma4_26b \
      evaluation/benchmark_dataset_gemma4_thesis_15.json \
      evaluation/thesis_final_run/thesis_test02_gemma4_sensitivity.yaml \
      gemma4:26b \
      "$rep"
    run_manifest \
      02v2_agent_step_parameter_sensitivity \
      mistral_small32_24b \
      evaluation/benchmark_dataset_gemma4_thesis_15.json \
      evaluation/thesis_final_run/thesis_test02_mistral_small32_sensitivity.yaml \
      mistral-small3.2:24b \
      "$rep"
  done
}

run_test03() {
  local rep
  for rep in "${REPEATS[@]}"; do
    run_manifest \
      03_max_tokens_agent_ladder \
      gemma4_e4b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
      gemma4:e4b \
      "$rep"
    run_manifest \
      03_max_tokens_agent_ladder \
      gemma4_26b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test03_gemma4_max_tokens.yaml \
      gemma4:26b \
      "$rep"
    run_manifest \
      03_max_tokens_agent_ladder \
      nemotron3_nano_4b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test03_nemotron3_nano_max_tokens.yaml \
      nemotron-3-nano:4b \
      "$rep"
    run_manifest \
      03_max_tokens_agent_ladder \
      mistral_small32_24b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test03_mistral_small32_max_tokens.yaml \
      mistral-small3.2:24b \
      "$rep"
  done
}

run_test04() {
  local rep
  for rep in "${REPEATS[@]}"; do
    run_manifest \
      04_compute_expansion_n_and_cot \
      gemma4_e4b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test04_gemma4_e4b_compute_expansion.yaml \
      gemma4:e4b \
      "$rep"
    run_manifest \
      04_compute_expansion_n_and_cot \
      mistral_small32_24b \
      evaluation/benchmark_dataset_gemma4_thesis_10.json \
      evaluation/thesis_final_run/thesis_test04_mistral_small32_compute_expansion.yaml \
      mistral-small3.2:24b \
      "$rep"
  done
}

cd "$ROOT_DIR"
printf 'Running final thesis Tests 01-04 sequentially. Test 05 is excluded.\n'
printf 'Docker image: %s | GPU device: %s | runs mount: %s\n' "$IMAGE_NAME" "$GPU_DEVICE" "$RUNS_HOST_DIR"
run_test01
run_test02
run_test03
run_test04
printf '\nFinal thesis Tests 01-04 finished.\n'
