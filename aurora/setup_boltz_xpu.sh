#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Install the patched official Boltz source without replacing Aurora PyTorch.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${GENIE3_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
project_root="${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}"
boltz_root=${GENIE3_BOLTZ_SOURCE:-${project_root}/software/boltz}
boltz_env=${GENIE3_BOLTZ_ENV:-${project_root}/envs/genie3-boltz-aurora}

source "${repo_root}/aurora/activate_genie3_aurora.sh"

if [[ ! -x "${boltz_env}/bin/python" ]]; then
    python -m venv --system-site-packages "${boltz_env}"
fi

"${boltz_env}/bin/python" -m pip install --upgrade pip
"${boltz_env}/bin/python" -m pip install --no-cache-dir -e "${boltz_root}"
