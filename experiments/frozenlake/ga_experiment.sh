#!/usr/bin/env bash
# ga_experiment.sh - driver for ga_experiment.py
#
# Repo-aware paths (assumes this file lives in ./experiments/frozenlake):
#   - script & ga_experiment.py:   ./experiments/frozenlake
#   - outputs & logs:              ./experiments/frozenlake/{outputs,logs}
#   - container (Apptainer SIF):   ./container/container.sif
#
# -----------------------------------------------------------------------------
# USAGE EXAMPLES (only set --exp and --maps; everything else uses defaults)
#
# 1) Local run (default maps = env0..env4, using the default container):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk
#
# 2) Local run with specific maps:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --maps env2,env4
#
# 3) Read maps from a file (one map per line):
#      printf "env0\nenvX\ncustom_map42\n" > maps.txt
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --maps-file maps.txt
#
# 4) Force host venv (NO container):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --no-container
#
# 5) Use a specific container path:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --container /abs/path/container.sif
#
# 6) Submit one SLURM job per map (defaults: 1 GPU / 32GB / 8 CPUs / 1 day):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --use-slurm
#      # change resources:
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --use-slurm --partition gpuA --gres gpu:1 --mem 48G --cpus 12 --days 2
#
# 7) Pass extra args through to ga_experiment.py (after “--”):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk -- --wandb-mode offline --skip-train --json-max 5
#
# Outputs go to:
#   ./experiments/frozenlake/outputs/<exp>/<map>/
# Logs go to:
#   ./experiments/frozenlake/logs/
# -----------------------------------------------------------------------------

set -euo pipefail

# -----------------------------
# Repo-aware path resolution
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../experiments/frozenlake
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # repo root
DEFAULT_CONTAINER="${PROJ_ROOT}/container/container.sif"
PYTHONPATH_EXTRA="${PROJ_ROOT}"                                 # <- make in-repo packages importable

# -----------------------------
# Defaults (edit if you must)
# -----------------------------
USE_SLURM=false
CONTAINER="${DEFAULT_CONTAINER}"    # use --no-container to force host venv
PYTHON="python3"
SCRIPT_PATH="${SCRIPT_DIR}/ga_experiment.py"
WANDB_MODE="online"
WORKDIR="${SCRIPT_DIR}"
LOGDIR="${SCRIPT_DIR}/logs"
OUTROOT="${SCRIPT_DIR}/outputs"

# SLURM defaults
SLURM_PARTITION="gpu"
SLURM_GRES="gpu:1"
SLURM_MEM="32G"
SLURM_CPUS="8"
SLURM_TIME_DAYS=1
SLURM_EXCLUDE=""

# -----------------------------
# Parse CLI
# -----------------------------
EXP_NAME=""
MAPS_INPUT="env0,env1,env2,env3,env4"   # keep your 8x8 if you want
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 --exp <name> [--maps <csv>|--maps-file <path>] [options]
(See header comments for examples.)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)           EXP_NAME="${2:-}"; shift ;;
    --maps)          MAPS_INPUT="${2:-}"; shift ;;
    --maps-file)     MAPS_INPUT="FILE:${2:-}"; shift ;;
    --use-slurm)     USE_SLURM=true ;;
    --no-container)  CONTAINER="" ;;
    --container)     CONTAINER="${2:-}"; shift ;;
    --python)        PYTHON="${2:-}"; shift ;;
    --wandb-mode)    WANDB_MODE="${2:-}"; shift ;;
    --workdir)       WORKDIR="$(cd "${2:-}" && pwd)"; shift ;;
    --outroot)       OUTROOT="$(mkdir -p "${2:-}" && cd "${2:-}" && pwd)"; shift ;;
    --logdir)        LOGDIR="$(mkdir -p "${2:-}" && cd "${2:-}" && pwd)"; shift ;;
    --script)        SCRIPT_PATH="$(cd "$(dirname "${2:-}")" && pwd)/$(basename "${2:-}")"; shift ;;
    --partition)     SLURM_PARTITION="${2:-}"; shift ;;
    --gres)          SLURM_GRES="${2:-}"; shift ;;
    --mem)           SLURM_MEM="${2:-}"; shift ;;
    --cpus)          SLURM_CPUS="${2:-}"; shift ;;
    --days)          SLURM_TIME_DAYS="${2:-}"; shift ;;
    --exclude)       SLURM_EXCLUDE="${2:-}"; shift ;;
    --)              shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)       usage ;;
    *)               echo "Unknown option: $1"; usage ;;
  esac
  shift
done

[[ -z "${EXP_NAME}" ]] && { echo "Error: --exp is required."; usage; }
[[ -f "${SCRIPT_PATH}" ]] || { echo "Not found: ${SCRIPT_PATH}"; exit 2; }
mkdir -p "${LOGDIR}" "${OUTROOT}"

# If container path is set but missing, warn & fall back to host
if [[ -n "${CONTAINER}" && ! -f "${CONTAINER}" ]]; then
  echo "Warning: container not found at ${CONTAINER}; falling back to host venv." >&2
  CONTAINER=""
fi

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

# SLURM time format
case "${SLURM_TIME_DAYS}" in
  1) SLURM_TIME="0-24:00" ;;
  2) SLURM_TIME="0-48:00" ;;
  *) SLURM_TIME="0-$((SLURM_TIME_DAYS*24)):00" ;;
esac

# -----------------------------
# Build command (array-safe)
# -----------------------------
CMD_ARR=()
build_cmd_for_map() {
  local map="$1"
  local run_name="${EXP_NAME}_${map}"
  local outdir="${OUTROOT%/}/${EXP_NAME}/${map}"

  if [[ -n "${CONTAINER}" ]]; then
    # Container: bind repo root and inject PYTHONPATH inside container
    CMD_ARR=(apptainer exec --nv --pwd "$WORKDIR" --bind "${PROJ_ROOT}:${PROJ_ROOT}"
             --env PYTHONPATH="${PYTHONPATH_EXTRA}"
             "$CONTAINER" "$PYTHON" "$SCRIPT_PATH"
             --run-name "$run_name" --map "$map" --outdir "$outdir" --wandb-mode "$WANDB_MODE")
  else
    # Host venv: export PYTHONPATH here
    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
    CMD_ARR=("$PYTHON" "$SCRIPT_PATH"
             --run-name "$run_name" --map "$map" --outdir "$outdir" --wandb-mode "$WANDB_MODE")
  fi
  if ((${#EXTRA_ARGS[@]})); then
    CMD_ARR+=("${EXTRA_ARGS[@]}")
  fi
}

print_cmd_line() { printf "%q " "$@"; }

# -----------------------------
# Local or SLURM execution
# -----------------------------
if ! ${USE_SLURM}; then
  echo "[Local] Running ${#MAPS[@]} map(s): ${MAPS[*]}"
  for map in "${MAPS[@]}"; do
    echo "==> MAP=${map}"
    mkdir -p "${OUTROOT%/}/${EXP_NAME}/${map}"
    build_cmd_for_map "${map}"
    echo "[PYTHONPATH] ${PYTHONPATH_EXTRA}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[CMD] $(print_cmd_line "${CMD_ARR[@]}")"
    "${CMD_ARR[@]}" 2>&1 | tee "${LOGDIR}/ga_${EXP_NAME}_${map}.log"
  done
  echo "[Local] All maps finished."
  exit 0
fi

# ---- SLURM path ----
echo "[SLURM] Submitting ${#MAPS[@]} map(s): ${MAPS[*]}"

for map in "${MAPS[@]}"; do
  job_script="$(mktemp)"
  job_name="ga_${EXP_NAME}_${map}"
  out_file="${LOGDIR}/${job_name}.out"
  err_file="${LOGDIR}/${job_name}.err"

  cat > "${job_script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${out_file}
#SBATCH --error=${err_file}
#SBATCH --nodes=1
#SBATCH --chdir=${WORKDIR}
EOF

  [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}" >> "${job_script}"

  cat >> "${job_script}" <<EOF
set -euo pipefail
module purge >/dev/null 2>&1 || true
# module load cuda || true    # uncomment if your cluster needs it
export PYTHONPATH="${PYTHONPATH_EXTRA}:\${PYTHONPATH:-}"   # <- make in-repo packages visible
EOF

  # append the actual command
  build_cmd_for_map "${map}"
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
