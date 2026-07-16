#!/usr/bin/env bash
set -euo pipefail

# Run the RL validation scenarios (delayed reward and combined reward) through
# Docker, 3 repetitions each, on the prompt subset requested by Hamta
# (test_case_id 7, 10, 12, 18 of evaluation/benchmark_dataset.json).
#
# Each scenario executes the mixed-model pipeline chosen by the RL agent:
# the lookup step model is overridden per step inside the manifests, while
# decide_tool and the remaining steps use mistral-small3.2:24b.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${DATA_AGENT_IMAGE:-data-agent}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OLLAMA_ENDPOINT="${OLLAMA_HOST:-http://localhost:11434}"
RUNS_HOST_DIR="${RUNS_HOST_DIR:-${ROOT_DIR}/runs}"
GT_JUDGE_PROVIDER="${GT_JUDGE_PROVIDER:-openai}"
GT_JUDGE_MODEL="${GT_JUDGE_MODEL:-gpt-5.4}"
AGENT_MODEL="mistral-small3.2:24b"
REPEATS=(01 02 03)
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash evaluation/run_rl_validation_docker.sh [--dry-run]

Runs the two RL validation scenarios sequentially with Docker and --resume,
3 repetitions each. Results are written under:

  runs/post_submission/rl_validation/

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
require_file evaluation/benchmark_dataset_rl_validation.json
require_file evaluation/benchmark_dataset_rl_validation_p7.json
require_file evaluation/benchmark_dataset_rl_validation_p10_12_18.json
require_file evaluation/post_submission/rl_validation_delayed_reward_p7.yaml
require_file evaluation/post_submission/rl_validation_delayed_reward_p10_12_18.yaml
require_file evaluation/post_submission/rl_validation_combined_reward.yaml

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v docker >/dev/null 2>&1 || {
    printf 'docker is required to run the RL validation batch.\n' >&2
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
  local slug="$1"
  local dataset="$2"
  local manifest="$3"
  local rep="$4"
  local save_dir="runs/post_submission/rl_validation/${slug}/rep${rep}"

  printf '\n%s\n' "================================================================================"
  printf 'RL validation: scenario=%s repeat=%s/03\n' "$slug" "$rep"
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
    --model "$AGENT_MODEL"
    --gt-judge-provider "$GT_JUDGE_PROVIDER"
    --gt-judge-model "$GT_JUDGE_MODEL"
    --no-gt-judge-provider ollama
    --no-gt-judge-model "$AGENT_MODEL"
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

cd "$ROOT_DIR"
printf 'Running RL validation scenarios sequentially.\n'
printf 'Docker image: %s | GPU device: %s | runs mount: %s\n' "$IMAGE_NAME" "$GPU_DEVICE" "$RUNS_HOST_DIR"

for rep in "${REPEATS[@]}"; do
  run_manifest \
    delayed_reward_p7 \
    evaluation/benchmark_dataset_rl_validation_p7.json \
    evaluation/post_submission/rl_validation_delayed_reward_p7.yaml \
    "$rep"

  run_manifest \
    delayed_reward_p10_12_18 \
    evaluation/benchmark_dataset_rl_validation_p10_12_18.json \
    evaluation/post_submission/rl_validation_delayed_reward_p10_12_18.yaml \
    "$rep"

  run_manifest \
    combined_reward \
    evaluation/benchmark_dataset_rl_validation.json \
    evaluation/post_submission/rl_validation_combined_reward.yaml \
    "$rep"
done

printf '\nRL validation batch finished.\n'
