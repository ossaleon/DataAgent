#!/usr/bin/env bash
set -euo pipefail

# Run the post-submission Test 05 refinement batch sequentially through Docker.
# The batch adds new Gemma 26B visualization-repair candidates and reruns the
# E4B/Mistral candidates that Test 05 originally reused from earlier tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DATA_AGENT_IMAGE:-data-agent}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OLLAMA_ENDPOINT="${OLLAMA_HOST:-http://localhost:11434}"
RUNS_HOST_DIR="${RUNS_HOST_DIR:-${ROOT_DIR}/runs}"
GT_JUDGE_PROVIDER="${GT_JUDGE_PROVIDER:-openai}"
GT_JUDGE_MODEL="${GT_JUDGE_MODEL:-gpt-5.4}"
REPEATS=(01 02)
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/run_post_submission_test05_refinement_docker.sh [--dry-run]

Runs the post-submission Test 05 refinement batch sequentially with Docker and
--resume. Results are written under:

  runs/post_submission/test05_refinement/

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
require_file evaluation/post_submission/thesis_post_submission_gemma4_26b_visualization_repair.yaml
require_file evaluation/post_submission/thesis_post_submission_gemma4_e4b_reused_test05.yaml
require_file evaluation/post_submission/thesis_post_submission_mistral_small32_24b_reused_test05.yaml

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v docker >/dev/null 2>&1 || {
    printf 'docker is required to run the post-submission batch.\n' >&2
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
  local model_slug="$1"
  local manifest="$2"
  local model="$3"
  local rep="$4"
  local save_dir="runs/post_submission/test05_refinement/${model_slug}/rep${rep}"

  printf '\n%s\n' "================================================================================"
  printf 'Post-submission Test 05 refinement: model=%s repeat=%s/02\n' "$model" "$rep"
  printf 'dataset=evaluation/benchmark_dataset.json\nmanifest=%s\nsave_dir=%s\n' "$manifest" "$save_dir"
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
    evaluation/benchmark_dataset.json
    "$manifest"
    --provider ollama
    --model "$model"
    --gt-judge-provider "$GT_JUDGE_PROVIDER"
    --gt-judge-model "$GT_JUDGE_MODEL"
    --no-gt-judge-provider ollama
    --no-gt-judge-model "$model"
    --save-dir "$save_dir"
    --repetition "${rep}/02"
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

cd "$ROOT_DIR"
printf 'Running post-submission Test 05 refinement sequentially.\n'
printf 'Docker image: %s | GPU device: %s | runs mount: %s\n' "$IMAGE_NAME" "$GPU_DEVICE" "$RUNS_HOST_DIR"

for rep in "${REPEATS[@]}"; do
  run_manifest \
    gemma4_26b \
    evaluation/post_submission/thesis_post_submission_gemma4_26b_visualization_repair.yaml \
    gemma4:26b \
    "$rep"

  run_manifest \
    gemma4_e4b \
    evaluation/post_submission/thesis_post_submission_gemma4_e4b_reused_test05.yaml \
    gemma4:e4b \
    "$rep"

  run_manifest \
    mistral_small32_24b \
    evaluation/post_submission/thesis_post_submission_mistral_small32_24b_reused_test05.yaml \
    mistral-small3.2:24b \
    "$rep"
done

printf '\nPost-submission Test 05 refinement finished.\n'
