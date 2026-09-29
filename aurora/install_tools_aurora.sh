#!/usr/bin/env bash
# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
# Install Genie3's small CPU evaluation helpers without the upstream CUDA setup.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${GENIE3_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
packages_dir="${GENIE3_PACKAGES_DIR:-${repo_dir}/packages}"
mkdir -p "${packages_dir}"

clone_public_repo() {
    local url="$1"
    local destination="$2"
    if [[ -d "${destination}/.git" ]]; then
        echo "Using existing source: ${destination}"
    elif [[ -e "${destination}" ]]; then
        echo "Refusing to replace non-Git path: ${destination}" >&2
        exit 1
    else
        git clone --depth 1 "${url}" "${destination}"
    fi
}

clone_public_repo \
    "https://github.com/dauparas/ProteinMPNN.git" \
    "${packages_dir}/ProteinMPNN"
clone_public_repo \
    "https://github.com/DunbrackLab/IPSAE.git" \
    "${packages_dir}/IPSAE"

if [[ ! -x "${packages_dir}/TMscore/TMalign" || ! -x "${packages_dir}/TMscore/TMscore" ]]; then
    mkdir -p "${packages_dir}/TMscore"
    (
        cd "${packages_dir}/TMscore"
        wget --quiet --output-document TMscore.cpp https://zhanggroup.org/TM-score/TMscore.cpp
        g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp
        wget --quiet --output-document TMalign.cpp https://zhanggroup.org/TM-align/TMalign.cpp
        g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
    )
fi

if [[ ! -x "${packages_dir}/dssp-2.3.0/mkdssp" ]]; then
    mkdir -p "${packages_dir}/dssp-2.3.0"
    wget --quiet --output-document "${packages_dir}/dssp-2.3.0/mkdssp" \
        https://github.com/martinpacesa/BindCraft/raw/refs/heads/main/functions/dssp
    chmod 0755 "${packages_dir}/dssp-2.3.0/mkdssp"
fi

test -f "${packages_dir}/ProteinMPNN/protein_mpnn_utils.py"
test -f "${packages_dir}/IPSAE/ipsae.py"
test -x "${packages_dir}/TMscore/TMalign"
test -x "${packages_dir}/TMscore/TMscore"
test -x "${packages_dir}/dssp-2.3.0/mkdssp"
echo "Genie3 Aurora evaluation helpers are ready under ${packages_dir}"
