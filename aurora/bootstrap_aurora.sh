#!/usr/bin/env bash
# Create Genie3's Aurora environment without replacing the vendor XPU PyTorch.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
project_root="${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}"
env_prefix="${GENIE3_AURORA_ENV:-${project_root}/envs/genie3-aurora}"
cache_root="${GENIE3_CACHE_ROOT:-${project_root}/cache/genie3-aurora}"
pretrained_root="${GENIE3_PRETRAINED_ROOT:-}"
fetch_weights=false
setup_colabfold=false
setup_boltz=false

usage() {
    cat <<'EOF'
Usage: aurora/bootstrap_aurora.sh [options]

Creates a Python 3.12 venv over Aurora's frameworks module and installs Genie3
without resolving or replacing the vendor-provided XPU PyTorch. Model weights
are reused from pretrained/ or GENIE3_PRETRAINED_ROOT; --weights downloads the
pretrained generation weights when no checkpoint is staged.

Options:
  --env PREFIX          Python environment path
  --cache-root PATH     Project-local runtime/cache root
  --pretrained-root DIR Existing directory containing v1/config.yaml
  --weights             Download Genie3 pretrained weights if needed
  --colabfold           Also set up the isolated ColabFold environment
  --boltz               Also set up the isolated Boltz2 environment
  -h, --help            Show this help
EOF
}

while (($#)); do
    case "$1" in
        --env) env_prefix="$2"; shift 2 ;;
        --cache-root) cache_root="$2"; shift 2 ;;
        --pretrained-root) pretrained_root="$2"; shift 2 ;;
        --weights) fetch_weights=true; shift ;;
        --colabfold) setup_colabfold=true; shift ;;
        --boltz) setup_boltz=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v module >/dev/null 2>&1; then
    echo "Aurora's module command is required to load the vendor XPU runtime." >&2
    exit 1
fi
set +u
frameworks_module="${AURORA_FRAMEWORKS_MODULE:-frameworks}"
module load "${frameworks_module}"
set -u

python_candidate="$(command -v python3 || command -v python)"
"${python_candidate}" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
    echo "${frameworks_module} must provide Python 3.12." >&2
    exit 1
}

mkdir -p "$(dirname -- "${env_prefix}")" "${cache_root}"
if [[ -e "${env_prefix}" && ! -x "${env_prefix}/bin/python" ]]; then
    echo "Environment path exists but is not a usable Python environment: ${env_prefix}" >&2
    exit 1
fi
if [[ ! -x "${env_prefix}/bin/python" ]]; then
    "${python_candidate}" -m venv --system-site-packages "${env_prefix}"
fi

export GENIE3_AURORA_ENV="${env_prefix}"
export GENIE3_CACHE_ROOT="${cache_root}"
source "${script_dir}/activate_genie3_aurora.sh"

python_bin="${env_prefix}/bin/python"
"${python_bin}" -m pip install --disable-pip-version-check --no-deps \
    -r "${script_dir}/requirements-aurora.txt"
"${python_bin}" -m pip install --disable-pip-version-check --no-deps \
    --no-build-isolation --editable "${repo_dir}"

bash "${script_dir}/install_tools_aurora.sh"

# The framework module supplies these runtime dependencies. Installing the
# package with --no-deps above prevents pip from replacing its XPU PyTorch.
"${python_bin}" - <<'PY'
import importlib.metadata
import importlib.util

modules = {
    "tqdm": "tqdm",
    "numpy": "numpy",
    "torch": "torch",
    "scipy": "scipy",
    "wandb": "wandb",
    "pandas": "pandas",
    "lightning": "lightning",
    "pytorch_lightning": "pytorch-lightning",
    "Bio": "biopython",
    "tensorboard": "tensorboard",
    "ml_collections": "ml-collections",
    "zstandard": "zstandard",
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
}
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Aurora framework dependencies: " + ", ".join(missing))

import lightning.pytorch
import pytorch_lightning
import tensorboard
import wandb
import zstandard

import torch
if not hasattr(torch, "xpu"):
    raise SystemExit("The loaded PyTorch build does not expose torch.xpu.")
print("PyTorch:", importlib.metadata.version("torch"), torch.__file__)
print("torch.xpu available on this host:", torch.xpu.is_available())
PY

if [[ ! -e "${repo_dir}/pretrained" && ! -L "${repo_dir}/pretrained" ]]; then
    if [[ -n "${pretrained_root}" ]]; then
        if [[ ! -f "${pretrained_root}/v1/config.yaml" ]]; then
            echo "Pretrained root lacks v1/config.yaml: ${pretrained_root}" >&2
            exit 1
        fi
        ln -s "${pretrained_root}" "${repo_dir}/pretrained"
    elif [[ -d "${project_root}/software/genie3/pretrained" && "${repo_dir}" != "${project_root}/software/genie3" ]]; then
        ln -s "${project_root}/software/genie3/pretrained" "${repo_dir}/pretrained"
    elif [[ "${fetch_weights}" == true ]]; then
        "${env_prefix}/bin/hf" download yeqinglin/genie3 \
            --include 'pretrained/**' --local-dir "${repo_dir}"
    fi
fi

if [[ "${setup_colabfold}" == true ]]; then
    bash "${script_dir}/setup_colabfold_xpu.sh"
fi
if [[ "${setup_boltz}" == true ]]; then
    bash "${script_dir}/setup_boltz_xpu.sh"
fi

"${env_prefix}/bin/genie3" --help >/dev/null
echo "Genie3 Aurora environment installed: ${env_prefix}"
echo "Runtime cache: ${cache_root}"
if [[ -e "${repo_dir}/pretrained/v1/checkpoints/step=600000.ckpt" ]]; then
    echo "Generation checkpoint: ${repo_dir}/pretrained/v1/checkpoints/step=600000.ckpt"
else
    echo "Generation weights are not staged; rerun with --weights before inference."
fi
