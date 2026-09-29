#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Install the ColabFold JAX stack that is compatible with Intel OpenXLA 0.6.0.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${GENIE3_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
project_root="${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}"
colabfold_env=${GENIE3_COLABFOLD_ENV:-${project_root}/envs/genie3-colabfold-aurora}

source "${repo_root}/aurora/activate_genie3_aurora.sh"

if [[ ! -x "${colabfold_env}/bin/python" ]]; then
    python -m venv "${colabfold_env}"
fi

"${colabfold_env}/bin/python" -m pip install --upgrade pip
"${colabfold_env}/bin/python" -m pip install --only-binary=:all: \
    "numpy==1.26.4" "scipy==1.11.4" \
    "jax==0.4.38" "jaxlib==0.4.38" "intel-extension-for-openxla==0.6.0" \
    "absl-py==1.4.0" "appdirs==1.4.4" "biopython==1.82" \
    "importlib-metadata>=4.8.2,<5" "tensorflow-cpu>=2.12.1,<3" \
    "dm-haiku==0.0.11" "dm-tree==0.1.8" "chex==0.1.88" \
    "immutabledict==4.2.1" "ml-collections==0.1.0" "pandas==2.2.3" \
    "matplotlib==3.8.4" "py3Dmol==2.4.2" "requests==2.32.3" "tqdm==4.67.1"

# ColabFold 1.5.4's published pandas cap predates Python 3.12 wheels. The
# compatible pinned runtime above is installed first; keep its package metadata
# from replacing the Intel-JAX dependency set.
"${colabfold_env}/bin/python" -m pip install --no-deps \
    "alphafold-colabfold==2.3.6" "colabfold==1.5.4"

"${colabfold_env}/bin/python" -c '
from pathlib import Path
from colabfold.download import download_alphafold_params
import os
download_alphafold_params(
    "alphafold2_multimer_v3",
    data_dir=Path(os.environ["GENIE3_COLABFOLD_DATA_DIR"]),
)
'
