#!/usr/bin/env bash
# ga_experiment.sh - driver for ga_experiment.py
#
# Layout assumptions (this file lives in ./experiments/frozenlake):
#   - script & ga_experiment.py:   ./experiments/frozenlake
#   - repo root:                   ./../..
#   - local container (SIF):       ./container/container.sif
#
# ---------------------------------------------------------------------------
# USAGE EXAMPLES (only set --exp and --maps; everything else has sane defaults)
#
# 1) Local (Apptainer, default maps env0..env4; outputs -> ./experiment_output):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk
#
# 2) Local with specific maps:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --maps env2,env4
#
# 3) Read maps from file (one per line):
#      printf "env0\nenvX\n" > maps.txt
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --maps-file maps.txt
#
# 4) Force host venv (NO container):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --no-container
#
# 5) Use a specific local container path:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --container /abs/path/container.sif
#
# 6) Submit one SLURM job per map (Singularity; outputs -> /scratch/users/$USER/experiment_output):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --use-slurm
#      # custom resources:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --use-slurm \
#         --partition gpu --gres gpu:1 --mem 48G --cpus 12 --days 2
#
# 7) Override experiment output roots:
#      # local:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk \
#         --exp-outroot-local /data/exp_out
#      # slurm:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --use-slurm \
#         --exp-outroot-slurm /scratch/users/$USER/myproj/exp_out
#
# 8) Pass extra args through to ga_experiment.py (after “--”):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk -- --wandb-mode offline --skip-train
#
# Output structure:
#   LOCAL  -> <LOCAL_EXP_OUTROOT>/<exp>/<map>/ (default: ./experiment_output/<exp>/<map>/)
#   SLURM  -> <SLURM_EXP_OUTROOT>/<exp>/<map>/ (default: /scratch/users/$USER/experiment_output/<exp>/<map>/)
#   W&B    -> <...>/wandb_runs/  (kept away from top-level to avoid import shadowing)
#   SLURM stdout/err -> /scratch/users/$USER/slurm_out/  (customizable)
# ---------------------------------------------------------------------------

set -euo pipefail

# -----------------------------
# Repo-aware paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../experiments/frozenlake
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

# local logs (small text logs for driver)
LOGDIR="${SCRIPT_DIR}/logs"

# SLURM resources
# SLURM_PARTITION="gpu"
# SLURM_GRES="gpu:1"
SLURM_PARTITION="cpu"
SLURM_GRES=""
SLURM_MEM="32G"
SLURM_CPUS="32"
SLURM_TIME_DAYS=1
SLURM_EXCLUDE=""

# SLURM stdout/err directory (on scratch, recommended by site docs)
SLURM_STDOUT_DIR_DEFAULT="/scratch/users/${USER}/slurm_out"

# Singularity cache/tmp on scratch (per docs)
SLUR_CACHE_DEFAULT="/scratch/users/${USER}/singularity/cache"

# -----------------------------
# CLI
# -----------------------------
EXP_NAME=""
MAPS_INPUT="8x8,env0,env1,env2,env3,env4"
EXTRA_ARGS=()
LOCAL_EXP_OUTROOT="${LOCAL_EXP_OUTROOT_DEFAULT}"
SLURM_EXP_OUTROOT="${SLURM_EXP_OUTROOT_DEFAULT}"
SLURM_STDOUT_DIR="${SLURM_STDOUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $0 --exp <name> [--maps <csv>|--maps-file <path>] [options]
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)                 EXP_NAME="${2:-}"; shift ;;
    --maps)                MAPS_INPUT="${2:-}"; shift ;;
    --maps-file)           MAPS_INPUT="FILE:${2:-}"; shift ;;
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
    --)                    shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)             usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

[[ -z "${EXP_NAME}" ]] && { echo "Error: --exp is required."; usage; }
[[ -f "${SCRIPT_PATH}" ]] || { echo "Not found: ${SCRIPT_PATH}"; exit 2; }

# -----------------------------
# Build map list
# -----------------------------
declare -a MAPS=()
if [[ "${MAPS_INPUT}" == FILE:* ]]; then
  MAP_FILE="${MAPS_INPUT#FILE:}"
  [[ -f "${MAP_FILE}" ]] || { echo "Map file not found: ${MAP_FILE}"; exit 2; }
  while IFS= read -r line; do
    m="$(echo "$line" | tr -d '[:space:]')"
    [[ -n "${m}" ]] && MAPS+=("${m}")
  done < "${MAP_FILE}"
else
  IFS=',' read -r -a MAPS <<< "${MAPS_INPUT}"
fi
[[ "${#MAPS[@]}" -gt 0 ]] || { echo "No maps parsed."; exit 2; }

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

WB_DIR="${EXP_OUTROOT%/}/wandb_runs"          # keep W&B artifacts outside code tree
mkdir -p "${LOGDIR}" "${EXP_OUTROOT}" "${WB_DIR}"

# sanity: container file presence (if requested)
if [[ -n "${CONTAINER}" && ! -f "${CONTAINER}" ]]; then
  echo "Warning: container not found at ${CONTAINER}; falling back to host venv." >&2
  CONTAINER=""
fi

# SLURM time string
case "${SLURM_TIME_DAYS}" in
  1) SLURM_TIME="0-24:00" ;;
  2) SLURM_TIME="0-48:00" ;;
  *) SLURM_TIME="0-$((SLURM_TIME_DAYS*24)):00" ;;
esac

# -----------------------------
# Command builder (array-safe)
# -----------------------------
CMD_ARR=()
build_cmd_for_map() {
  local map="$1"
  local run_name="${EXP_NAME}_${map}"
  local outdir="${EXP_OUTROOT%/}/${EXP_NAME}/${map}"

  mkdir -p "${outdir}"

  if [[ -n "${CONTAINER}" ]]; then
    if [[ "${ENGINE}" == "apptainer" ]]; then
      # Local: Apptainer
      CMD_ARR=(apptainer exec --nv --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
    else
      # SLURM: Singularity
      CMD_ARR=(singularity exec --nv --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
    fi
  else
    # Host venv
    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
    export WANDB_DIR="${WB_DIR}"
    CMD_ARR=("${PYTHON}" "${SCRIPT_PATH}"
             --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}")
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
  echo "[Local/${ENGINE:-host}] Running ${#MAPS[@]} map(s): ${MAPS[*]}"
  echo "[Local] EXP_OUTROOT = ${EXP_OUTROOT}"
  echo "[Local] WANDB_DIR   = ${WB_DIR}"
  for map in "${MAPS[@]}"; do
    echo "==> MAP=${map}"
    build_cmd_for_map "${map}"
    echo "[CMD] $(print_cmd_line "${CMD_ARR[@]}")"
    "${CMD_ARR[@]}" 2>&1 | tee "${LOGDIR}/ga_${EXP_NAME}_${map}.log"
  done
  echo "[Local] All maps finished."
  exit 0
fi

# -------- SLURM submission path (one job per map) --------
mkdir -p "${SLURM_STDOUT_DIR}"
echo "[SLURM] Submitting ${#MAPS[@]} map(s): ${MAPS[*]}"
echo "[SLURM] EXP_OUTROOT = ${EXP_OUTROOT}"
echo "[SLURM] STDOUT_DIR  = ${SLURM_STDOUT_DIR}"

for map in "${MAPS[@]}"; do
  job_script="$(mktemp)"
  job_name="ga_${EXP_NAME}_${map}"
  out_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.out"
  err_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.err"

  # NB: bash -l as per site docs (to enable Environment Modules)
  cat > "${job_script}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${out_file}
#SBATCH --error=${err_file}
#SBATCH --nodes=1
#SBATCH --chdir=${SCRIPT_DIR}
EOF
  [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}" >> "${job_script}"

  # job body
  cat >> "${job_script}" <<'EOF'
set -euo pipefail
module purge >/dev/null 2>&1 || true
EOF

  # per-site recommended Singularity cache/tmp on scratch
  cat >> "${job_script}" <<EOF
export SINGULARITY_CACHEDIR="${SLUR_CACHE_DEFAULT}"
export SINGULARITY_TMPDIR="/scratch/users/${USER}/\${SLURM_JOB_ID}/tmp"
mkdir -p "\${SINGULARITY_CACHEDIR}" "\${SINGULARITY_TMPDIR}"
EOF

  # expose PYTHONPATH / WANDB_DIR to the job (singularity will also get them via --env)
  cat >> "${job_script}" <<EOF
export PYTHONPATH="${PYTHONPATH_EXTRA}:\${PYTHONPATH:-}"
export WANDB_DIR="${WB_DIR}"
EOF

  # actual command
  build_cmd_for_map "${map}"
  echo "$(print_cmd_line "${CMD_ARR[@]}")" >> "${job_script}"

  # submit
  if sb_out=$(sbatch "${job_script}"); then
    echo "Submitted: ${sb_out}"
  else
    echo "Failed to submit ${job_name}" >&2
  fi
  rm -f "${job_script}"
  sleep 0.3
done

echo "[SLURM] All jobs submitted."
