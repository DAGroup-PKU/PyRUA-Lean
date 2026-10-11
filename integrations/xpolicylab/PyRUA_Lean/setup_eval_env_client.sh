#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

# The simulator client reads what an observation holds from env_cfg.  With `observation: rgbd`
# in deploy.yml it is started from a copy of RoboDojo's code and configuration in which the
# depth image and the camera matrices are switched on (client_root.py); the checkout itself
# is not edited, and eval_result/ in the copy is the checkout's.  The debug client has no
# simulator and runs where it is.
# shellcheck source=../../utils/resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
client_root="${BENCH_ROOT}"
if [[ "$(resolve_eval_env_type)" == "sim" ]]; then
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/pyrualean-client.XXXXXX")"
    trap 'rm -rf "${scratch}"' EXIT
    client_root="$(
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${eval_env_conda_env}"
        python "${SCRIPT_DIR}/client_root.py" "${BENCH_ROOT}" "${scratch}/root" \
            --env-cfg "${env_cfg_type}" --deploy "${yaml_file}"
    )"
    echo "[CLIENT] simulator client root: ${client_root}"
fi

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${client_root}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
