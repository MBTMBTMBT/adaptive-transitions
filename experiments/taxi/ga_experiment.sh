#!/usr/bin/env bash
# ga_experiment_taxi.sh - driver for ga_experiment.py
#
# Layout assumptions (this file lives in ./experiments/taxi):
#   - script & ga_experiment.py:  ./experiments/taxi
#   - repo root:                             ./../..
#   - local container (SIF):                 ./container/container.sif
#
# -----------------------------------------------------------------------------
# USAGE EXAMPLES
# 1) Local (Apptainer), run 3 replicas:
#      experiments/taxi/ga_experiment.sh --exp taxi_test --count 1
#
# 2) Force host venv (NO container):
#      experiments/taxi/ga_experiment.sh --exp taxi_test --no-container
#
# 3) Use SLURM (one job per replica):
#      experiments/taxi/ga_experiment.sh --exp taxi_test --count 1 --use-slurm
#
# 4) Pass extra args through to ga_experiment.py (after “--”):
#      experiments/taxi/ga_experiment.sh --exp taxi_test -- --wandb-mode offline --skip-vis
#
# Output structure:
#   LOCAL  -> <LOCAL_EXP_OUTROOT>/<exp>/r<k>/           (default: ./experiment_output/<exp>/r<k>/)
#   SLURM  -> <SLURM_EXP_OUTROOT>/<exp>/r<k>/           (default: /scratch/users/$USER/experiment_output/<exp>/r<k>/)
#   W&B    -> <...>/wandb_runs/                         (kept away from top-level to avoid import shadowing)
#   SLURM stdout/err -> /scratch/users/$USER/slurm_out/ (customizable)
# -----------------------------------------------------------------------------

set -euo pipefail

# -----------------------------
# Repo-aware paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../experiments/taxi
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"                 # repo root
DEFAULT_SIF="${PROJ_ROOT}/container/container.sif"
PYTHONPATH_EXTRA="${PROJ_ROOT}"

# -----------------------------
# Defaults
# -----------------------------
USE_SLURM=false
CONTAINER="${DEFAULT_SIF}"          # --no-container to force host venv
PYTHON="python3"
SCRIPT_PATH="${SCRIPT_DIR}/ga_experiment.py"
WANDB_MODE="online"

# experiment outputs (NEW: fully separated)
LOCAL_EXP_OUTROOT_DEFAULT="${PROJ_ROOT}/experiment_output"
SLURM_EXP_OUTROOT_DEFAULT="/scratch/users/${USER}/experiment_output"

# SLURM resources
SLURM_PARTITION="cpu"
SLURM_GRES=""
SLURM_MEM="31G"
SLURM_CPUS="63"
SLURM_TIME_DAYS="2.0"   # supports decimal (e.g., 1.5 -> 36h)
SLURM_EXCLUDE=""

# SLURM stdout/err directory
SLURM_STDOUT_DIR_DEFAULT="/scratch/users/${USER}/slurm_out"

# Singularity cache/tmp on scratch
SLUR_CACHE_DEFAULT="/scratch/users/${USER}/singularity/cache"

# -----------------------------
# CLI
# -----------------------------
EXP_NAME=""
COUNT=1                     # number of replicas
EXTRA_ARGS=()
LOCAL_EXP_OUTROOT="${LOCAL_EXP_OUTROOT_DEFAULT}"
SLURM_EXP_OUTROOT="${SLURM_EXP_OUTROOT_DEFAULT}"
SLURM_STDOUT_DIR="${SLURM_STDOUT_DIR_DEFAULT}"
LOGDIR_CLI=""  # Optional override for driver logs.

usage() {
  cat <<EOF
Usage: $0 --exp <name> [--count <N>] [options]
Options:
  --use-slurm              Submit one SLURM job per replica.
  --no-container           Run on host venv.
  --container <path>       Use a specific container .sif.
  --python <exe>           Python executable.
  --wandb-mode <mode>      online|offline (default: online).
  --exp-outroot-local <p>  Local output root (default: ${LOCAL_EXP_OUTROOT_DEFAULT}).
  --exp-outroot-slurm <p>  SLURM output root (default: ${SLURM_EXP_OUTROOT_DEFAULT}).
  --slurm-stdout-dir <p>   SLURM stdout/err dir (default: ${SLURM_STDOUT_DIR_DEFAULT}).
  --partition <name>       SLURM partition (default: cpu).
  --gres <spec>            SLURM GRES (e.g., gpu:1).
  --mem <mem>              SLURM memory (default: ${SLURM_MEM}).
  --cpus <n>               SLURM CPUs (default: ${SLURM_CPUS}).
  --days <float>           Walltime in DAYS (default: ${SLURM_TIME_DAYS}).
  --exclude <nodes>        SLURM node exclude list.
  --script <path>          Python script path (default: ${SCRIPT_PATH}).
  --logdir <path>          Driver logs dir override.
  -- <args...>             Extra args passed to Python script.
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)                 EXP_NAME="${2:-}"; shift ;;
    --count)               COUNT="${2:-1}"; shift ;;
    --use-slurm)           USE_SLURM=true ;;
    --no-container)        CONTAINER="" ;;
    --container)           CONTAINER="${2:-}"; shift ;;
    --python)              PYTHON="${2:-}"; shift ;;
    --wandb-mode)          WANDB_MODE="${2:-}"; shift ;;
    --exp-outroot-local)   LOCAL_EXP_OUTROOT="$(cd "${2:-}" 2>/dev/null && pwd || echo "${2:-}")"; shift ;;
    --exp-outroot-slurm)   SLURM_EXP_OUTROOT="${2:-}"; shift ;;
    --slurm-stdout-dir)    SLURM_STDOUT_DIR="${2:-}"; shift ;;
    --partition)           SLURM_PARTITION="${2:-}"; shift ;;
    --gres)                SLURM_GRES="${2:-}"; shift ;;
    --mem)                 SLURM_MEM="${2:-}"; shift ;;
    --cpus)                SLURM_CPUS="${2:-}"; shift ;;
    --days)                SLURM_TIME_DAYS="${2:-}"; shift ;;
    --exclude)             SLURM_EXCLUDE="${2:-}"; shift ;;
    --script)              SCRIPT_PATH="$(cd "$(dirname "${2:-}")" && pwd)/$(basename "${2:-}")"; shift ;;
    --logdir)              LOGDIR_CLI="${2:-}"; shift ;;
    --)                    shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)             usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

[[ -z "${EXP_NAME}" ]] && { echo "Error: --exp is required."; usage; }
[[ -f "${SCRIPT_PATH}" ]] || { echo "Not found: ${SCRIPT_PATH}"; exit 2; }
[[ "${COUNT}" =~ ^[0-9]+$ ]] || { echo "Error: --count must be an integer."; exit 2; }

# -----------------------------
# Derived paths
# -----------------------------
if ! ${USE_SLURM}; then
  EXP_OUTROOT="${LOCAL_EXP_OUTROOT}"
  ENGINE="apptainer"
else
  EXP_OUTROOT="${SLURM_EXP_OUTROOT}"
  ENGINE="singularity"
fi

WB_DIR="${EXP_OUTROOT%/}/wandb_runs"
mkdir -p "${EXP_OUTROOT}" "${WB_DIR}"

# Driver logs
if [[ -n "${LOGDIR_CLI}" ]]; then
  LOGDIR="$(cd "${LOGDIR_CLI}" 2>/dev/null && pwd || echo "${LOGDIR_CLI}")"
else
  LOGDIR="${EXP_OUTROOT%/}/${EXP_NAME}/_driver_logs"
fi
mkdir -p "${LOGDIR}"

# Container presence (if requested)
if [[ -n "${CONTAINER}" && ! -f "${CONTAINER}" ]]; then
  echo "Warning: container not found at ${CONTAINER}; falling back to host venv." >&2
  CONTAINER=""
fi

# -----------------------------
# SLURM time string (supports decimal days)
# -----------------------------
days_str="${SLURM_TIME_DAYS}"
total_min=$(awk -v d="$days_str" 'BEGIN{printf("%d", d*24*60 + 0.5)}')
d=$(( total_min / (24*60) ))
rem=$(( total_min % (24*60) ))
h=$(( rem / 60 ))
m=$(( rem % 60 ))
SLURM_TIME=$(printf "%d-%02d:%02d:00" "$d" "$h" "$m")
echo "[SLURM] Requested days=${SLURM_TIME_DAYS} -> --time=${SLURM_TIME}"

# -----------------------------
# NV flag logic
# -----------------------------
NV_FLAG=""
if ${USE_SLURM}; then
  if [[ "${SLURM_GRES}" =~ gpu ]] || [[ "${SLURM_PARTITION}" =~ gpu ]]; then
    NV_FLAG="--nv"
  fi
else
  NV_FLAG="--nv"
fi

# -----------------------------
# Command builder
# -----------------------------
CMD_ARR=()
build_cmd_for_rep() {
  local rep_idx="$1"
  local run_name="${EXP_NAME}_r${rep_idx}"
  local outdir="${EXP_OUTROOT%/}/${EXP_NAME}/r${rep_idx}"

  mkdir -p "${outdir}"

  if [[ -n "${CONTAINER}" ]]; then
    if [[ "${ENGINE}" == "apptainer" ]]; then
      # Local: Apptainer
      CMD_ARR=(apptainer exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
    else
      # SLURM: Singularity
      CMD_ARR=(singularity exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
    fi
  else
    # Host venv
    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
    export WANDB_DIR="${WB_DIR}"
    CMD_ARR=("${PYTHON}" "${SCRIPT_PATH}"
             --run-name "${run_name}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
  fi

  if ((${#EXTRA_ARGS[@]})); then
    CMD_ARR+=("${EXTRA_ARGS[@]}")
  fi
}

print_cmd_line() { printf "%q " "$@"; }

# -----------------------------
# Execute locally or submit SLURM
# -----------------------------
if ! ${USE_SLURM}; then
  echo "[Local/${ENGINE:-host}] Running ${COUNT} replica(s)"
  echo "[Local] EXP_OUTROOT = ${EXP_OUTROOT}"
  echo "[Local] WANDB_DIR   = ${WB_DIR}"
  echo "[Local] LOGDIR      = ${LOGDIR}"
  for ((i=0; i<COUNT; i++)); do
    echo "==> REPLICA r${i}"
    build_cmd_for_rep "${i}"
    echo "[CMD] $(print_cmd_line "${CMD_ARR[@]}")"
    "${CMD_ARR[@]}" 2>&1 | tee "${LOGDIR}/ga_${EXP_NAME}_r${i}.log"
  done
  echo "[Local] All replicas finished."
  exit 0
fi

# -------- SLURM submission path (one job per replica) --------
mkdir -p "${SLURM_STDOUT_DIR}"
echo "[SLURM] Submitting ${COUNT} replica(s)"
echo "[SLURM] EXP_OUTROOT = ${EXP_OUTROOT}"
echo "[SLURM] STDOUT_DIR  = ${SLURM_STDOUT_DIR}"
echo "[SLURM] DRIVER LOGS = ${LOGDIR}"

for ((i=0; i<COUNT; i++)); do
  job_script="$(mktemp)"
  job_name="ga_${EXP_NAME}_r${i}"
  out_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.out"
  err_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.err"

  cat > "${job_script}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${out_file}
#SBATCH --error=${err_file}
#SBATCH --nodes=1
#SBATCH --chdir=${SCRIPT_DIR}
EOF
  [[ -n "${SLURM_GRES}" ]]    && echo "#SBATCH --gres=${SLURM_GRES}"        >> "${job_script}"
  [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}"  >> "${job_script}"

  cat >> "${job_script}" <<'EOF'
set -euo pipefail
# Reset environment modules (ignore failures quietly).
module purge >/dev/null 2>&1 || true

# 1) Ensure container runtime modules are available (common on clusters); try loading both.
module load apptainer  >/dev/null 2>&1 || true
module load singularity >/dev/null 2>&1 || true

# 2) Pre-flight check: can this node resolve my UID? (critical)
if ! getent passwd "${UID}" >/dev/null 2>&1; then
  echo "[WARN] $(hostname): no passwd entry for UID=${UID}; requeue job ${SLURM_JOB_ID}"
  # Try to requeue (so the scheduler places us on another node).
  scontrol requeue "${SLURM_JOB_ID}" || exit 1
  exit 0
fi
EOF

  cat >> "${job_script}" <<EOF
export SINGULARITY_CACHEDIR="${SLUR_CACHE_DEFAULT}"
export SINGULARITY_TMPDIR="/scratch/users/${USER}/\${SLURM_JOB_ID}/tmp"
mkdir -p "\${SINGULARITY_CACHEDIR}" "\${SINGULARITY_TMPDIR}"
export PYTHONPATH="${PYTHONPATH_EXTRA}:\${PYTHONPATH:-}"
export WANDB_DIR="${WB_DIR}"
EOF

  build_cmd_for_rep "${i}"
  echo "$(print_cmd_line "${CMD_ARR[@]}")" >> "${job_script}"

  if sb_out=$(sbatch "${job_script}"); then
    echo "Submitted: ${sb_out}"
  else
    echo "Failed to submit ${job_name}" >&2
  fi
  rm -f "${job_script}"
  sleep 0.3
done

echo "[SLURM] All jobs submitted."
