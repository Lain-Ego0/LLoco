#!/usr/bin/env bash
# Run the requested finite validation training for every migrated source task.
# Usage: NUM_ENVS=1024 ITERATIONS=1000 ./scripts/run_go2_validation.sh

set -u

NUM_ENVS="${NUM_ENVS:-1024}"
ITERATIONS="${ITERATIONS:-1000}"
STEPS_PER_ENV="${STEPS_PER_ENV:-24}"
STUDENT_STEPS_PER_ENV="${STUDENT_STEPS_PER_ENV:-50}"
LOG_ROOT="${LOG_ROOT:-logs/go2_validation}"
RUN_NAME="${RUN_NAME:-validation_${NUM_ENVS}x${ITERATIONS}}"
MOTION_DIR="${GO2_MOTION_DIR:-/home/lxy/下载/My_unitree_go2_gym-main/datasets/mocap_motions_go2}"
SKIP_ONNX_EXPORT="${GO2_SKIP_ONNX_EXPORT:-1}"
TASK_START_INDEX="${TASK_START_INDEX:-0}"
TASK_END_INDEX="${TASK_END_INDEX:-}"
RESUME_FIRST_TASK="${RESUME_FIRST_TASK:-0}"
RESUME_RUN="${RESUME_RUN:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-model_0.pt}"

tasks=(
  Mjlab-Trot-Flat-Unitree-Go2
  Mjlab-Jump-Flat-Unitree-Go2
  Mjlab-Spring-Jump-Flat-Unitree-Go2
  Mjlab-Backflip-Flat-Unitree-Go2
  Mjlab-Handstand-Flat-Unitree-Go2
  Mjlab-Leggedstand-Flat-Unitree-Go2
  Mjlab-DreamWaQ-Rough-Unitree-Go2
  Mjlab-AMP-DreamWaQ-Rough-Unitree-Go2
  Mjlab-CTS-Rough-Unitree-Go2
  Mjlab-AMP-CTS-Rough-Unitree-Go2
  Mjlab-AMP-TS-Rough-Unitree-Go2
  Mjlab-TS-Rough-Unitree-Go2
  Mjlab-AMP-TS-Student-Rough-Unitree-Go2
  Mjlab-TS-Student-Rough-Unitree-Go2
)

mkdir -p "${LOG_ROOT}"
status_file="${LOG_ROOT}/validation_status.tsv"
if [[ -z "${TASK_END_INDEX}" ]]; then
  TASK_END_INDEX="${#tasks[@]}"
fi
if [[ "${TASK_START_INDEX}" -eq 0 && "${TASK_END_INDEX}" -eq "${#tasks[@]}" ]]; then
  printf 'task\tstarted\tfinished\texit_code\n' > "${status_file}"
elif [[ ! -f "${status_file}" ]]; then
  printf 'task\tstarted\tfinished\texit_code\n' > "${status_file}"
fi

latest_checkpoint() {
  local root="$1"
  # A stable alias is preferred over nested historical runs.  This also
  # prevents lexical path sorting from selecting an older alias checkpoint.
  if [[ -f "${root}/model_999.pt" ]]; then
    printf '%s\n' "${root}/model_999.pt"
    return 0
  fi
  find "${root}" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

for ((task_index=TASK_START_INDEX; task_index<TASK_END_INDEX; task_index++)); do
  task="${tasks[$task_index]}"
  started="$(date -Iseconds)"
  echo "[GO2-VALIDATION] START ${task} (${NUM_ENVS} envs x ${ITERATIONS} iterations)"

  # Student variants use the completed TS teacher checkpoint when available.
  # This keeps the standalone student run source-compatible while retaining a
  # clear warning/fallback if a prior teacher run was interrupted.
  teacher_checkpoint=""
  if [[ "${task}" == *AMP-TS-Student* ]]; then
    teacher_checkpoint="$(latest_checkpoint "${LOG_ROOT}/go2_amp_ts")"
  elif [[ "${task}" == *TS-Student* ]]; then
    teacher_checkpoint="$(latest_checkpoint "${LOG_ROOT}/go2_ts")"
  fi

  task_steps_per_env="${STEPS_PER_ENV}"
  if [[ "${task}" == *TS-Student* ]]; then
    # The source recurrent distillation runner backpropagates over 50 frames.
    task_steps_per_env="${STUDENT_STEPS_PER_ENV}"
  fi
  train_args=(
    --env.scene.num-envs "${NUM_ENVS}"
    --agent.max-iterations "${ITERATIONS}"
    --agent.num-steps-per-env "${task_steps_per_env}"
    --agent.logger tensorboard --agent.upload-model False
    --log-root "${LOG_ROOT}" --agent.run-name "${RUN_NAME}"
    --agent.save-interval 100
  )
  if [[ "${RESUME_FIRST_TASK}" -eq 1 && "${task_index}" -eq "${TASK_START_INDEX}" ]]; then
    train_args+=(
      --agent.resume True
      --agent.load-run "${RESUME_RUN}"
      --agent.load-checkpoint "${RESUME_CHECKPOINT}"
    )
  fi

  if [[ -n "${teacher_checkpoint}" ]]; then
    GO2_MOTION_DIR="${MOTION_DIR}" GO2_SKIP_ONNX_EXPORT="${SKIP_ONNX_EXPORT}" GO2_TS_TEACHER_CHECKPOINT="${teacher_checkpoint}" \
      uv run train "${task}" "${train_args[@]}"
  else
    GO2_MOTION_DIR="${MOTION_DIR}" GO2_SKIP_ONNX_EXPORT="${SKIP_ONNX_EXPORT}" \
      uv run train "${task}" "${train_args[@]}"
  fi

  code=$?
  finished="$(date -Iseconds)"
  printf '%s\t%s\t%s\t%s\n' "${task}" "${started}" "${finished}" "${code}" >> "${status_file}"
  echo "[GO2-VALIDATION] END ${task} exit=${code}"

  # Publish completed TS teachers under stable paths consumed by student runs.
  if [[ "${code}" -eq 0 && "${task}" == "Mjlab-AMP-TS-Rough-Unitree-Go2" ]]; then
    source="$(find "${LOG_ROOT}/go2_amp_ts" -mindepth 2 -type f -name 'model_*.pt' \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
    if [[ -n "${source}" ]]; then
      mkdir -p "${LOG_ROOT}/go2_amp_ts"
      ln -f "${source}" "${LOG_ROOT}/go2_amp_ts/$(basename "${source}")" 2>/dev/null || true
    fi
  elif [[ "${code}" -eq 0 && "${task}" == "Mjlab-TS-Rough-Unitree-Go2" ]]; then
    source="$(find "${LOG_ROOT}/go2_ts" -mindepth 2 -type f -name 'model_*.pt' \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
    if [[ -n "${source}" ]]; then
      mkdir -p "${LOG_ROOT}/go2_ts"
      ln -f "${source}" "${LOG_ROOT}/go2_ts/$(basename "${source}")" 2>/dev/null || true
    fi
  fi
done

echo "[GO2-VALIDATION] status: ${status_file}"
