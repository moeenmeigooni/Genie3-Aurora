#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Execute the isolated ColabFold/JAX runtime with the cache environment passed
# through from activate_genie3_aurora.sh.
set -euo pipefail

project_root="${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}"
colabfold_env=${GENIE3_COLABFOLD_ENV:-${project_root}/envs/genie3-colabfold-aurora}

if [[ ! -x "${colabfold_env}/bin/colabfold_batch" ]]; then
    echo "ColabFold XPU environment is missing: ${colabfold_env}" >&2
    exit 2
fi

exec "${colabfold_env}/bin/colabfold_batch" "$@"
