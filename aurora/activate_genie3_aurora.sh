#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Source this script from an Aurora login or compute node before using Genie3.
# It deliberately redirects every relevant cache and runtime-home path away
# from the user's actual home directory.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script: source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

frameworks_module="${AURORA_FRAMEWORKS_MODULE:-frameworks}"
set +u
module load "${frameworks_module}"
set -u

genie3_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
genie3_project_root=${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}
genie3_env=${GENIE3_AURORA_ENV:-${genie3_project_root}/envs/genie3-aurora}
genie3_cache_root=${GENIE3_CACHE_ROOT:-${genie3_project_root}/cache/genie3-aurora}
genie3_runtime_home=${GENIE3_RUNTIME_HOME:-${genie3_cache_root}/runtime-home}
genie3_colabfold_env=${GENIE3_COLABFOLD_ENV:-${genie3_project_root}/envs/genie3-colabfold-aurora}
genie3_boltz_env=${GENIE3_BOLTZ_ENV:-${genie3_project_root}/envs/genie3-boltz-aurora}
genie3_tools_env=${GENIE3_TOOLS_ENV:-${genie3_project_root}/conda_envs/genie3-tools-aurora}

if [[ ! -x "${genie3_env}/bin/python" ]]; then
    echo "Genie3 environment is missing: ${genie3_env}" >&2
    return 2
fi

mkdir -p \
    "${genie3_runtime_home}" \
    "${genie3_cache_root}/pip" \
    "${genie3_cache_root}/tmp" \
    "${genie3_cache_root}/xdg" \
    "${genie3_cache_root}/huggingface" \
    "${genie3_cache_root}/torch" \
    "${genie3_cache_root}/triton" \
    "${genie3_cache_root}/torchinductor" \
    "${genie3_cache_root}/boltz" \
    "${genie3_cache_root}/wandb" \
    "${genie3_cache_root}/numba" \
    "${genie3_cache_root}/jax" \
    "${genie3_cache_root}/matplotlib" \
    "${genie3_cache_root}/python-userbase" \
    "${genie3_cache_root}/conda-pkgs"

export GENIE3_AURORA_ENV="${genie3_env}"
export GENIE3_CACHE_ROOT="${genie3_cache_root}"
export GENIE3_RUNTIME_HOME="${genie3_runtime_home}"
export HOME="${genie3_runtime_home}"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${genie3_cache_root}/python-userbase"
export PIP_CACHE_DIR="${genie3_cache_root}/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="${genie3_cache_root}/tmp"
export XDG_CACHE_HOME="${genie3_cache_root}/xdg"
export HF_HOME="${genie3_cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${genie3_cache_root}/torch"
export TORCHINDUCTOR_CACHE_DIR="${genie3_cache_root}/torchinductor"
export TRITON_CACHE_DIR="${genie3_cache_root}/triton"
export WANDB_DIR="${genie3_cache_root}/wandb"
export WANDB_CACHE_DIR="${genie3_cache_root}/wandb"
export NUMBA_CACHE_DIR="${genie3_cache_root}/numba"
export JAX_COMPILATION_CACHE_DIR="${genie3_cache_root}/jax"
export MPLCONFIGDIR="${genie3_cache_root}/matplotlib"
export GENIE3_ACCELERATOR=xpu
export GENIE3_XPU_DISTRIBUTED_BACKEND=xccl
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export GENIE3_ESMFOLD_MODEL_DIR="${genie3_cache_root}/models/esmfold_v1"
export GENIE3_COLABFOLD_ENV="${genie3_colabfold_env}"
export GENIE3_COLABFOLD_BATCH="${GENIE3_COLABFOLD_BATCH:-${genie3_repo_root}/aurora/colabfold_batch_xpu.sh}"
export GENIE3_COLABFOLD_DATA_DIR="${XDG_CACHE_HOME}/colabfold"
export GENIE3_BOLTZ_ENV="${genie3_boltz_env}"
export GENIE3_BOLTZ_BIN="${GENIE3_BOLTZ_BIN:-${genie3_repo_root}/aurora/boltz_xpu.sh}"
export GENIE3_BOLTZ_CACHE="${genie3_cache_root}/boltz"
export BOLTZ_CACHE="${GENIE3_BOLTZ_CACHE}"
export GENIE3_TOOLS_ENV="${genie3_tools_env}"
if [[ -d "${genie3_tools_env}/bin" ]]; then
    export PATH="${genie3_tools_env}/bin:${PATH}"
fi

source "${genie3_env}/bin/activate"
unset genie3_repo_root genie3_project_root genie3_env genie3_cache_root genie3_runtime_home genie3_colabfold_env genie3_boltz_env genie3_tools_env
