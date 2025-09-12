#!/usr/bin/env bash
# ga_experiment.sh - driver for ga_experiment.py / ga_experiment_ray.py
#
# Layout assumptions (this file lives in ./experiments/frozenlake):
#   - this script & ga_experiment*.py: ./experiments/frozenlake
#   - repo root:                      ./../..
#   - local container (SIF):          ./container/container.sif
#
# See usage block for examples.

set -euo pipefail

# -----------------------------
# Repo-aware paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SIF="${PROJ_ROOT}/container/container.sif"
PYTHONPATH_EXTRA="${PROJ_ROOT}"

# -----------------------------
# Defaults
# -----------------------------
USE_SLURM=false
CONTAINER="${DEFAULT_SIF}"        # --no-container to force host venv
PYTHON="python3"
SCRIPT_PATH="${SCRIPT_DIR}/ga_experiment_ray.py"
WANDB_MODE="online"

LOCAL_EXP_OUTROOT_DEFAULT="${PROJ_ROOT}/experiment_output"
SLURM_EXP_OUTROOT_DEFAULT="/scratch/users/${USER}/experiment_output"

SLURM_PARTITION="cpu,gpu,nmes_gpu"
SLURM_GRES=""
SLURM_MEM="5G"
SLURM_CPUS="1"
SLURM_NODES="5"
SLURM_TIME_DAYS="2.0"
SLURM_EXCLUDE=""

SLURM_STDOUT_DIR_DEFAULT="/scratch/users/${USER}/slurm_out"
SLUR_CACHE_DEFAULT="/scratch/users/${USER}/singularity/cache"

# -----------------------------
# CLI
# -----------------------------
EXP_NAME=""
MAPS_INPUT="8x8,env3,env4"
OBJ_GROUPS_INPUT="auc_source_target,perf_source_target,perf_kl_source_target,kl,auc_source_value_diff,auc_source_target_kl,auc_source_target_value_diff"

EXTRA_ARGS=()
LOCAL_EXP_OUTROOT="${LOCAL_EXP_OUTROOT_DEFAULT}"
SLURM_EXP_OUTROOT="${SLURM_EXP_OUTROOT_DEFAULT}"
SLURM_STDOUT_DIR="${SLURM_STDOUT_DIR_DEFAULT}"
LOGDIR_CLI=""

usage() {
  cat <<EOF
Usage: $0 --exp <name> [--maps <csv>|--maps-file <path>] [--obj-groups <csv>|--obj-groups-file <path>] [options]
Options:
  --use-slurm
  --no-container | --container <path> | --python <exe>
  --partition <p> --gres <g> --mem <mem> --cpus <n> --nodes <n> --days <float> --exclude <nodes>
  --exp-outroot-local <dir> --exp-outroot-slurm <dir> --slurm-stdout-dir <dir>
  --wandb-mode <online|offline>
  --script <path>  # path to ga_experiment_*.py
  --logdir <dir>   # driver logs dir
  --               # everything after this goes to Python script verbatim
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
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help)             usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

[[ -z "${EXP_NAME}" ]] && { echo "Error: --exp is required."; usage; }
[[ -f "${SCRIPT_PATH}" ]] || { echo "Not found: ${SCRIPT_PATH}"; exit 2; }

# -----------------------------
# Parse map list
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
# Parse objective group list
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

WB_DIR="${EXP_OUTROOT%/}/wandb_runs"
mkdir -p "${EXP_OUTROOT}" "${WB_DIR}"

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
# SLURM time string (support decimal days)
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
# NV flag at submit time (only for local Apptainer convenience)
# -----------------------------
NV_FLAG=""
if ! ${USE_SLURM}; then
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
      CMD_ARR=(apptainer exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}" --obj-group "${obj_group}")
    else
      CMD_ARR=(singularity exec ${NV_FLAG:+$NV_FLAG} --pwd "${SCRIPT_DIR}"
               --bind "${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER}"
               --env PYTHONPATH="${PYTHONPATH_EXTRA}"
               --env WANDB_DIR="${WB_DIR}"
               "${CONTAINER}" "${PYTHON}" "${SCRIPT_PATH}"
               --run-name "${run_name}" --map "${map}" --outdir "${outdir}" --wandb-mode "${WANDB_MODE}" --obj-group "${obj_group}")
    fi
  else
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

    build_cmd_for_map "${map}" "${g}"
    CMD_STR="$(print_cmd_line "${CMD_ARR[@]}")"

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
    [[ -n "${SLURM_GRES}" ]]    && echo "#SBATCH --gres=${SLURM_GRES}"         >> "${job_script}"
    [[ -n "${SLURM_EXCLUDE}" ]] && echo "#SBATCH --exclude=${SLURM_EXCLUDE}"   >> "${job_script}"

    cat >> "${job_script}" <<EOF
CONTAINER="${CONTAINER}"
ENGINE="${ENGINE}"
SCRIPT_DIR="${SCRIPT_DIR}"
PROJ_ROOT="${PROJ_ROOT}"
EXP_OUTROOT="${EXP_OUTROOT}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA}"
WB_DIR="${WB_DIR}"
export SINGULARITY_CACHEDIR="${SLUR_CACHE_DEFAULT}"
export SINGULARITY_TMPDIR="/scratch/users/${USER}/\${SLURM_JOB_ID}/tmp"
export RAY_TMPDIR="/scratch/users/${USER}/\${SLURM_JOB_ID}/ray"
EOF

    cat >> "${job_script}" <<'EOF'
set -euo pipefail

module purge >/dev/null 2>&1 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ulimit -n 65536 || true

# Prepare scratch dirs before any singularity exec
: "${SINGULARITY_CACHEDIR:="/scratch/users/${USER}/singularity/cache"}"
: "${SINGULARITY_TMPDIR:="/scratch/users/${USER}/${SLURM_JOB_ID}/tmp"}"
: "${RAY_TMPDIR:="/scratch/users/${USER}/${SLURM_JOB_ID}/ray"}"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}" "${RAY_TMPDIR}"

# Build container exec base (runtime --nv only if devices exist)
RUNTIME_NV_FLAG=""
if [[ -e /dev/nvidiactl || -e /dev/nvidia0 ]]; then
  RUNTIME_NV_FLAG="--nv"
fi
if [[ -n "${CONTAINER}" ]]; then
  if [[ "${ENGINE}" == "singularity" ]]; then
    BASE_CMD="singularity exec ${RUNTIME_NV_FLAG} --pwd \"${SCRIPT_DIR}\" \
      --bind \"${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT},/scratch/users/${USER}:/scratch/users/${USER}\" \
      --env PYTHONPATH=\"${PYTHONPATH_EXTRA}\" --env WANDB_DIR=\"${WB_DIR}\" \
      \"${CONTAINER}\""
  else
    BASE_CMD="apptainer exec ${RUNTIME_NV_FLAG} --pwd \"${SCRIPT_DIR}\" \
      --bind \"${PROJ_ROOT}:${PROJ_ROOT},${EXP_OUTROOT}:${EXP_OUTROOT}\" \
      --env PYTHONPATH=\"${PYTHONPATH_EXTRA}\" --env WANDB_DIR=\"${WB_DIR}\" \
      \"${CONTAINER}\""
  fi
else
  BASE_CMD=""
fi

# Resolve node list and head
mapfile -t NODE_ARR < <(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE="${NODE_ARR[0]}"
echo "[RAY] Nodes: ${NODE_ARR[*]}"
echo "[RAY] Head:  ${HEAD_NODE}"

# Resolve a concrete IP to bind for --node-ip-address (hostname first)
HEAD_IP="$(srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "getent ahostsv4 ${HEAD_NODE} | awk '{print \$1; exit}' || hostname -I | tr ' ' '\n' | grep -E '^(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.)' | grep -v '^172\\.17\\.' | head -n1" | tr -d '\r')"
echo "[RAY] Head IP: ${HEAD_IP:-<none>}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# Stop any leftover Ray on allocated nodes
for n in "${NODE_ARR[@]}"; do
  if [[ -n "${BASE_CMD}" ]]; then
    srun -N1 -n1 -w "$n" bash -lc "${BASE_CMD} ray stop >/dev/null 2>&1 || true" || true
  else
    srun -N1 -n1 -w "$n" bash -lc "ray stop >/dev/null 2>&1 || true" || true
  fi
done

# Start head (bind to HEAD_IP); Ray temp-dir to scratch
if [[ -n "${BASE_CMD}" ]]; then
  HEAD_CMD="${BASE_CMD} ray start --head \
    --node-ip-address=${HEAD_IP} \
    --port=${RAY_PORT} \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --dashboard-host=0.0.0.0 \
    --num-cpus=${SLURM_CPUS_PER_TASK:-1} \
    --disable-usage-stats \
    --temp-dir=${RAY_TMPDIR}"
else
  HEAD_CMD="ray start --head \
    --node-ip-address=${HEAD_IP} \
    --port=${RAY_PORT} \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --dashboard-host=0.0.0.0 \
    --num-cpus=${SLURM_CPUS_PER_TASK:-1} \
    --disable-usage-stats \
    --temp-dir=${RAY_TMPDIR}"
fi

echo "[RAY] Starting head on ${HEAD_NODE} ..."
srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "${HEAD_CMD}"

# Export RAY_ADDRESS as hostname:port for all subsequent steps
export RAY_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
echo "[RAY] Waiting for GCS @ ${RAY_ADDRESS} ..."

# Quiet, reliable health check (socket connect) to avoid srun error spam
READY=0
for i in {1..90}; do
  HC_OUT="$(srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "python3 - <<'PY'
import socket, sys, os
host, port = os.environ.get('RAY_ADDRESS','').split(':')
ok = False
try:
    with socket.create_connection((host, int(port)), timeout=1):
        ok = True
except Exception:
    pass
print('OK' if ok else 'WAIT')
PY" 2>/dev/null || true)"
  if echo "${HC_OUT}" | grep -q 'OK'; then READY=1; break; fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  echo "[RAY] GCS not reachable in time; tail head logs for hints:"
  srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "ls -1 ${RAY_TMPDIR}/session_latest/logs 2>/dev/null | tail -n +1 | sed 's/^/  /' || true"
  exit 1
fi
echo "[RAY] GCS is ready."

# Worker base; connect via hostname:port; bind each worker to its own IP
if [[ -n "${BASE_CMD}" ]]; then
  WORKER_BASE="${BASE_CMD} ray start --address ${RAY_ADDRESS} --num-cpus=${SLURM_CPUS_PER_TASK:-1} --disable-usage-stats --temp-dir=${RAY_TMPDIR}"
else
  WORKER_BASE="ray start --address ${RAY_ADDRESS} --num-cpus=${SLURM_CPUS_PER_TASK:-1} --disable-usage-stats --temp-dir=${RAY_TMPDIR}"
fi

if (( ${#NODE_ARR[@]} > 1 )); then
  echo "[RAY] Starting workers ..."
  for w in "${NODE_ARR[@]:1}"; do
    srun -N1 -n1 -w "$w" bash -lc "mkdir -p \"${SINGULARITY_CACHEDIR}\" \"${SINGULARITY_TMPDIR}\" \"${RAY_TMPDIR}\"; \
      WIP=\$(getent ahostsv4 ${w} | awk '{print \$1; exit}' || hostname -I | tr ' ' '\n' | grep -E '^(10\\.|192\\.168\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.)' | grep -v '^172\\.17\\.' | head -n1); \
      ${WORKER_BASE} --node-ip-address=\${WIP}" &
  done
  wait
fi

# Optional quick sanity print of cluster resources (non-fatal)
srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "python3 - <<'PY' || true
import os, json
try:
    import ray
    ray.init(address=os.environ.get('RAY_ADDRESS','auto'), log_to_driver=False, namespace='sanity')
    print('[SANITY] Resources:', json.dumps(ray.cluster_resources(), sort_keys=True))
except Exception as e:
    print('[SANITY] Skip:', e)
PY"

# Make sure experiment sees RAY_ADDRESS
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
export WANDB_DIR="${WB_DIR}"
export RAY_ADDRESS="${RAY_ADDRESS}"

EOF

    printf "RUN_CMD='%s'\n" "${CMD_STR//\'/\'\"\'\"\'}" >> "${job_script}"
    cat >> "${job_script}" <<'EOF'
echo "[CMD] ${RUN_CMD}"
srun -N1 -n1 -w "${HEAD_NODE}" bash -lc "${RUN_CMD}"

# Graceful Ray shutdown (best-effort)
echo "[RAY] Stopping cluster ..."
for n in "${NODE_ARR[@]}"; do
  if [[ -n "${BASE_CMD}" ]]; then
    srun -N1 -n1 -w "$n" bash -lc "${BASE_CMD} ray stop >/dev/null 2>&1 || true" || true
  else
    srun -N1 -n1 -w "$n" bash -lc "ray stop >/dev/null 2>&1 || true" || true
  fi
done
EOF

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
