#!/usr/bin/env bash
# Run the requested finite validation training for every migrated source task.
# Usage: NUM_ENVS=1024 ITERATIONS=1000 ./scripts/run_go2_validation.sh

set -u

NUM_ENVS="${NUM_ENVS:-1024}"
ITERATIONS="${ITERATIONS:-1000}"
STEPS_PER_ENV="${STEPS_PER_ENV:-24}"
LOG_ROOT="${LOG_ROOT:-logs/go2_validation}"
RUN_NAME="${RUN_NAME:-validation_${NUM_ENVS}x${ITERATIONS}}"
MOTION_DIR="${GO2_MOTION_DIR:-/home/lxy/下载/My_unitree_go2_gym-main/datasets/mocap_motions_go2}"

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
printf 'task\tstarted\tfinished\texit_code\n' > "${status_file}"

for task in "${tasks[@]}"; do
  started="$(date -Iseconds)"
  echo "[GO2-VALIDATION] START ${task} (${NUM_ENVS} envs x ${ITERATIONS} iterations)"

  # Student variants use the completed TS teacher checkpoint when available.
  # This keeps the standalone student run source-compatible while retaining a
  # clear warning/fallback if a prior teacher run was interrupted.
  teacher_checkpoint=""
  if [[ "${task}" == *TS-Student* ]]; then
    teacher_checkpoint="$(find "${LOG_ROOT}/go2_ts" "${LOG_ROOT}/go2_amp_ts" \
      -type f -name "model_${ITERATIONS}.pt" 2>/dev/null | sort | tail -n 1)"
  fi

  if [[ -n "${teacher_checkpoint}" ]]; then
    GO2_MOTION_DIR="${MOTION_DIR}" GO2_TS_TEACHER_CHECKPOINT="${teacher_checkpoint}" \
      uv run train "${task}" \
        --env.scene.num-envs "${NUM_ENVS}" \
        --agent.max-iterations "${ITERATIONS}" \
        --agent.num-steps-per-env "${STEPS_PER_ENV}" \
        --agent.logger tensorboard --agent.upload-model False \
        --log-root "${LOG_ROOT}" --agent.run-name "${RUN_NAME}" \
        --agent.save-interval 100
  else
    GO2_MOTION_DIR="${MOTION_DIR}" \
      uv run train "${task}" \
        --env.scene.num-envs "${NUM_ENVS}" \
        --agent.max-iterations "${ITERATIONS}" \
        --agent.num-steps-per-env "${STEPS_PER_ENV}" \
        --agent.logger tensorboard --agent.upload-model False \
        --log-root "${LOG_ROOT}" --agent.run-name "${RUN_NAME}" \
        --agent.save-interval 100
  fi

  code=$?
  finished="$(date -Iseconds)"
  printf '%s\t%s\t%s\t%s\n' "${task}" "${started}" "${finished}" "${code}" >> "${status_file}"
  echo "[GO2-VALIDATION] END ${task} exit=${code}"
done

echo "[GO2-VALIDATION] status: ${status_file}"
