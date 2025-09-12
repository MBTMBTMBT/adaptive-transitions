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
# 0) What I actually run on create:
#      bash experiments/frozenlake/ga_experiment.sh --exp ga-frozenlake --use-slurm
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
#         --partition gpu --gres gpu:1 --mem 48G --cpus 12 --days 2.5
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
# 9) Select objective groups (CSV or file; default = ALL groups):
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --obj-groups perf_source_target,kl
#      experiments/frozenlake/ga_experiment.sh --exp my_flk --obj-groups-file groups.txt
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
SCRIPT_PATH="${SCRIPT_DIR}/ga_experiment_ray.py"
WANDB_MODE="online"

# experiment outputs (NEW: fully separated)
LOCAL_EXP_OUTROOT_DEFAULT="${PROJ_ROOT}/experiment_output"
SLURM_EXP_OUTROOT_DEFAULT="/scratch/users/${USER}/experiment_output"

# SLURM resources
SLURM_PARTITION="cpu,gpu,nmes_gpu"
SLURM_GRES=""
SLURM_MEM="31G"
SLURM_CPUS="21"          # per-node CPUs (default 21)
SLURM_NODES="5"          # total nodes (default 5)
SLURM_TIME_DAYS="2.0"    # supports decimal now, e.g., 1.5
SLURM_EXCLUDE=""

# SLURM stdout/err directory (on scratch, recommended by site docs)
SLURM_STDOUT_DIR_DEFAULT="/scratch/users/${USER}/slurm_out"

# Singularity cache/tmp on scratch (per docs)
SLUR_CACHE_DEFAULT="/scratch/users/${USER}/singularity/cache"

# -----------------------------
# CLI
# -----------------------------
EXP_NAME=""
MAPS_INPUT="8x8,env3,env4"  # env0,env1,env2
OBJ_GROUPS_INPUT="auc_source_target,perf_source_target,perf_kl_source_target,kl,auc_source_value_diff,auc_source_target_kl,auc_source_target_value_diff"
EXTRA_ARGS=()
LOCAL_EXP_OUTROOT="${LOCAL_EXP_OUTROOT_DEFAULT}"
SLURM_EXP_OUTROOT="${SLURM_EXP_OUTROOT_DEFAULT}"
SLURM_STDOUT_DIR="${SLURM_STDOUT_DIR_DEFAULT}"
LOGDIR_CLI=""  # Optional override for driver logs.

usage() {
  cat <<EOF
Usage: $0 --exp <name> [--maps <csv>|--maps-file <path>] [--obj-groups <csv>|--obj-groups-file <path>] [options]
Options:
  --days <float>     Walltime in DAYS (supports decimals, e.g., 0.5 -> 12h).
  --nodes <int>      Total nodes for a Ray-backed job (default: 5).
  --cpus <int>       CPUs per node (default: 21).
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)                 EXP_NAME="${2:-}"; shift ;;
    --maps)                MAPS_INPUT="${2:-}"; shift ;;
    --maps-file)           MAPS_INPUT="FILE:${2:-}"; shift ;;
    --obj-groups)          OBJ_GROUPS_INPUT="${2:-}"; shift ;;
    --obj-groups-file)     OBJ_GROUPS_INPUT="FILE:${2:-}"; shift ;;
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
    --nodes)               SLURM_NODES="${2:-}"; shift ;;
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
# Build objective group list
# -----------------------------
declare -a OBJ_GROUPS=()
if [[ "${OBJ_GROUPS_INPUT}" == FILE:* ]]; then
  OG_FILE="${OBJ_GROUPS_INPUT#FILE:}"
  [[ -f "${OG_FILE}" ]] || { echo "Obj-groups file not found: ${OG_FILE}"; exit 2; }
  while IFS= read -r line; do
    g="$(echo "$line" | tr -d '[:space:]')"
    [[ -n "${g}" ]] && OBJ_GROUPS+=("${g}")
  done < "${OG_FILE}"
else
  IFS=',' read -r -a OBJ_GROUPS <<< "${OBJ_GROUPS_INPUT}"
fi
[[ "${#OBJ_GROUPS[@]}" -gt 0 ]] || { echo "No objective groups parsed."; exit 2; }

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
mkdir -p "${EXP_OUTROOT}" "${WB_DIR}"

# Unify driver LOGDIR under EXP_OUTROOT, but allow user override via --logdir.
if [[ -n "${LOGDIR_CLI}" ]]; then
  LOGDIR="$(cd "${LOGDIR_CLI}" 2>/dev/null && pwd || echo "${LOGDIR_CLI}")"
else
  LOGDIR="${EXP_OUTROOT%/}/${EXP_NAME}/_driver_logs"
fi
mkdir -p "${LOGDIR}"

# sanity: container file presence (if requested)
if [[ -n "${CONTAINER}" && ! -f "${CONTAINER}" ]]; then
  echo "Warning: container not found at ${CONTAINER}; falling back to host venv." >&2
  CONTAINER=""
fi

# -----------------------------
# SLURM time string (supports decimal days)
# -----------------------------
days_str="${SLURM_TIME_DAYS}"
# compute total minutes with rounding using awk (avoids bash float)
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
  # Only add --nv for Singularity when GPUs were requested
  if [[ "${SLURM_GRES}" =~ gpu ]] || [[ "${SLURM_PARTITION}" =~ gpu ]]; then
    NV_FLAG="--nv"
  fi
else
  # Keep --nv for local Apptainer to match previous behavior
  NV_FLAG="--nv"
fi

# -----------------------------
# Command builder (array-safe)
# -----------------------------
CMD_ARR=()
build_cmd_for_map() {
  local map="$1"
  local obj_group="$2"
  local run_name="${EXP_NAME}_${map}_${obj_group}"
  local outdir="${EXP_OUTROOT%/}/${EXP_NAME}/${map}"

  mkdir -p "${outdir}"

  if [[ -n "${CONTAINER}" ]]; then
    if [[ "${ENGINE}" == "apptainer" ]]; then
      # Local: Apptainer
      CMD_ARR=(apptainer exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}" --obj-group "${obj_group}")
    else
      # SLURM: Singularity
      CMD_ARR=(singularity exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}" --obj-group "${obj_group}")
    fi
  else
    # Host venv
    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
    export WANDB_DIR="${WB_DIR}"
    CMD_ARR=("${PYTHON}" "${SCRIPT_PATH}"
             --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}" --obj-group "${obj_group}")
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
  echo "[Local] Objective groups = ${OBJ_GROUPS[*]}"
  echo "[Local] EXP_OUTROOT = ${EXP_OUTROOT}"
  echo "[Local] WANDB_DIR   = ${WB_DIR}"
  echo "[Local] LOGDIR      = ${LOGDIR}"
  for map in "${MAPS[@]}"; do
    for g in "${OBJ_GROUPS[@]}"; do
      echo "==> MAP=${map}  OBJ_GROUP=${g}"
      build_cmd_for_map "${map}" "${g}"
      echo "[CMD] $(print_cmd_line "${CMD_ARR[@]}")"
      "${CMD_ARR[@]}" 2>&1 | tee "${LOGDIR}/ga_${EXP_NAME}_${map}_${g}.log"
    done
  done
  echo "[Local] All (map, obj_group) combinations finished."
  exit 0
fi

# -------- SLURM submission path (one job per (map, obj_group)) --------
mkdir -p "${SLURM_STDOUT_DIR}"
echo "[SLURM] Submitting ${#MAPS[@]} map(s): ${MAPS[*]}"
echo "[SLURM] Objective groups = ${OBJ_GROUPS[*]}"
echo "[SLURM] EXP_OUTROOT = ${EXP_OUTROOT}"
echo "[SLURM] STDOUT_DIR  = ${SLURM_STDOUT_DIR}"
echo "[SLURM] DRIVER LOGS = ${LOGDIR}"

for map in "${MAPS[@]}"; do
  for g in "${OBJ_GROUPS[@]}"; do
    job_script="$(mktemp)"
    job_name="ga_${EXP_NAME}_${map}_${g}"
    out_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.out"
    err_file="${SLURM_STDOUT_DIR%/}/${job_name}_%j.err"

    # Build the Python command now (stringified) for later srun on head node.
    build_cmd_for_map "${map}" "${g}"
    CMD_STR="$(print_cmd_line "${CMD_ARR[@]}")"

    # NB: bash -l as per site docs (to enable Environment Modules)
    cat > "${job_script}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${out_file}
#SBATCH --error=${err_file}
#SBATCH --chdir=${SCRIPT_DIR}
EOF
    [[ -n "${SLURM_GRES}" ]]    && echo "#SBATCH --gres=${SLURM_GRES}"     >> "${job_script}"
    [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}" >> "${job_script}"

    # --- Bake variables into the job script (avoid set -u issues) ---
    cat >> "${job_script}" <<EOF
CONTAINER="${CONTAINER}"
ENGINE="${ENGINE}"
NV_FLAG="${NV_FLAG}"
SCRIPT_DIR="${SCRIPT_DIR}"
PROJ_ROOT="${PROJ_ROOT}"
EXP_OUTROOT="${EXP_OUTROOT}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA}"
WB_DIR="${WB_DIR}"
export SINGULARITY_CACHEDIR="${SLUR_CACHE_DEFAULT}"
export SINGULARITY_TMPDIR="/scratch/users/${USER}/\${SLURM_JOB_ID}/tmp"
EOF

    # --- Job body: start Ray head+workers, then run Python on head ---
    cat >> "${job_script}" <<'EOF'
set -euo pipefail
module purge >/dev/null 2>&1 || true

# Basic threading hygiene
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -n 65536 || true

# Derive node list and choose head
mapfile -t NODE_ARR < <(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE="${NODE_ARR[0]}"
echo "[RAY] Nodes: ${NODE_ARR[*]}"
echo "[RAY] Head:  ${HEAD_NODE}"

# Ray ports (override with env if needed)
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# Build container exec prefix once
RAY_LAUNCH=""
if [[ -n "${CONTAINER}" ]]; then
  if [[ "${ENGINE}" == "singularity" ]]; then
    RAY_LAUNCH="singularity exec ${NV_FLAG:+$NV_FLAG} --pwd ${SCRIPT_DIR} \
      --bind ${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER} \
      --env PYTHONPATH=${PYTHONPATH_EXTRA} --env WANDB_DIR=${WB_DIR}"
  else
    RAY_LAUNCH="apptainer exec ${NV_FLAG:+$NV_FLAG} --pwd ${SCRIPT_DIR} \
      --bind ${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT} \
      --env PYTHONPATH=${PYTHONPATH_EXTRA} --env WANDB_DIR=${WB_DIR}"
  fi
fi

echo "[RAY] Starting head on ${HEAD_NODE} ..."
srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "${RAY_LAUNCH} ray start --head --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-cpus=${SLURM_CPUS_PER_TASK:-1} --disable-usage-stats"
wait

# Start workers on remaining nodes
if (( ${#NODE_ARR[@]} > 1 )); then
  echo "[RAY] Starting workers ..."
  for w in "${NODE_ARR[@]:1}"; do
    srun -N1 -n1 -w "$w" bash -lc "${RAY_LAUNCH} ray start --address ${HEAD_NODE}:${RAY_PORT} --num-cpus=${SLURM_CPUS_PER_TASK:-1} --disable-usage-stats" &
  done
  wait
fi

export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"

# Ensure cache/tmp dirs exist (were baked above)
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# Expose PYTHONPATH / WANDB_DIR (also passed in container env)
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export WANDB_DIR="${WB_DIR}"
EOF

    # Inject the resolved Python command and run on head
    printf "RUN_CMD='%s'\n" "${CMD_STR//\'/\'\"\'\"\'}" >> "${job_script}"
    cat >> "${job_script}" <<'EOF'
echo "[CMD] ${RUN_CMD}"
srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "${RUN_CMD}"

# Graceful Ray shutdown (best-effort)
echo "[RAY] Stopping cluster ..."
for n in "${NODE_ARR[@]}"; do
  srun -N1 -n1 -w "$n" bash -lc 'ray stop >/dev/null 2>&1 || true' || true
done
EOF

    # submit
    if sb_out=$(sbatch "${job_script}"); then
      echo "Submitted: ${sb_out}"
    else
      echo "Failed to submit ${job_name}" >&2
    fi
    rm -f "${job_script}"
    sleep 0.3
  done
done

echo "[SLURM] All jobs submitted."
