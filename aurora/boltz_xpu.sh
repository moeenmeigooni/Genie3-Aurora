#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Execute the isolated Boltz runtime with the cache environment passed through
# from activate_genie3_aurora.sh.
set -euo pipefail

project_root="${GENIE3_PROJECT_ROOT:-${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}}"
boltz_env=${GENIE3_BOLTZ_ENV:-${project_root}/envs/genie3-boltz-aurora}

if [[ ! -x "${boltz_env}/bin/boltz" ]]; then
    echo "Boltz XPU environment is missing: ${boltz_env}" >&2
    exit 2
fi

exec "${boltz_env}/bin/boltz" "$@"
