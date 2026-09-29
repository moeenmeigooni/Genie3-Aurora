# Genie3 on Aurora XPU

Clone the Aurora port and enter its checkout:

```bash
git clone https://github.com/moeenmeigooni/Genie3-Aurora.git
cd Genie3-Aurora
```

This is the Aurora port of the AlQurayshi Lab's Genie3 repository. It uses
the ALCF default `frameworks` module (currently `frameworks/2026.1.0` on
Aurora), which provides native
`torch.xpu` and the `xccl` distributed backend. No CUDA PyTorch wheel or
Intel Extension for PyTorch is installed.

Install the Aurora environment and the Genie3 source package on an Aurora
login node. `--weights` downloads the pretrained generation checkpoint; omit it
when `pretrained/` is already available in the checkout or is linked from a
shared project cache:

```bash
bash aurora/bootstrap_aurora.sh --weights
```

For a test or alternate installation path, set the environment and cache roots
before running the installer:

```bash
GENIE3_AURORA_ENV=/path/to/test/envs/genie3 \
GENIE3_CACHE_ROOT=/path/to/project/cache/genie3 \
bash aurora/bootstrap_aurora.sh
```

Set `AURORA_PROJECT_ROOT` to a shared project directory visible from both the
login and compute nodes. By default, the scripts use
`/lus/flare/projects/FRAME-IDP/$USER`. Source the activation script from the
checkout root before any run:

```bash
export AURORA_PROJECT_ROOT="/lus/flare/projects/FRAME-IDP/${USER}"
source aurora/activate_genie3_aurora.sh
```

It redirects `HOME`, Hugging Face, PyTorch, compiler, Python, pip, temporary,
and tool caches to `${AURORA_PROJECT_ROOT}/cache/genie3-aurora`.
Nothing in this setup writes a cache to the user's actual home directory.

## Environments and tools

| Component | Aurora implementation | Location |
| --- | --- | --- |
| Genie3 generation and ESMFold | Project venv inheriting ALCF's native PyTorch XPU | `envs/genie3-aurora` |
| ColabFold | Isolated ColabFold 1.5.4 + JAX 0.4.38 + Intel OpenXLA 0.6.0 | `envs/genie3-colabfold-aurora` |
| Boltz2 | Patched official Boltz v2.2.1, isolated from Genie3 dependencies | `envs/genie3-boltz-aurora` |
| MMseqs2, HHsuite, Kalign, Foldseek | Project conda tools environment | `conda_envs/genie3-tools-aurora` |
| ProteinMPNN, IPSAE, TM-align, DSSP | Genie3 project tools | `software/genie3/packages` |

The generation workflow uses the native Lightning XPU adapter. ESMFold uses
the pure-PyTorch Transformers implementation in place of the upstream
OpenFold/CUDA extension. ColabFold runs through its isolated JAX/OpenXLA
environment and its normal file-based backend. Boltz2 is invoked through its
isolated XPU-aware Lightning runtime with CUDA-only custom kernels disabled;
this preserves the standard Boltz2 prediction outputs while using portable
PyTorch operators.

Evaluation workers use Python's `spawn` multiprocessing context on XPU. This
avoids PyTorch's restriction against initializing XPU from a forked child and
keeps the normal per-device parallel mapping workflow.

The main installer also fetches ProteinMPNN and IPSAE and builds the TM-align,
TMscore, and DSSP helpers used by evaluation. Those tools live under the
checkout's ignored `packages/` directory. The default installer does not run
the upstream CUDA environment setup.

The patched Boltz source is fetched by `aurora/setup_boltz_xpu.sh`, which
recreates its isolated environment after sourcing the activation script.

Likewise, `aurora/setup_colabfold_xpu.sh` recreates the JAX 0.4.38 / Intel
OpenXLA 0.6.0 ColabFold environment and caches the multimer parameters in the
project cache.

## Validation and use

Run these on allocated Aurora nodes, not on a UAN:

```bash
repo="$PWD"
qsub -A FRAME-IDP -q debug "${repo}/aurora/genie3_xpu_smoke.pbs"
qsub -A FRAME-IDP -q debug "${repo}/aurora/colabfold_xpu_smoke.pbs"
qsub -A FRAME-IDP -q debug "${repo}/aurora/esmfold_xpu_smoke.pbs"
qsub -A FRAME-IDP -q debug "${repo}/aurora/boltz_xpu_smoke.pbs"
```

After the Boltz adapter smoke passes, `aurora/boltz_xpu_predict_smoke.pbs`
runs a real small Boltz2 prediction and caches its checkpoint under the project
cache. It has a one-hour debug allocation because the first model load and
download can be substantial.

Genie3 pretrained generation weights, the Transformers ESMFold checkpoint, and
ColabFold multimer parameters are stored in the project cache. Submit the
bounded generation validation with:

```bash
cd /path/to/Genie3-Aurora
qsub -A FRAME-IDP -q debug -v \
  GENIE3_CONFIG="${PWD}/aurora/xpu_validation.yaml" \
  aurora/genie3_generate.pbs
```

The matching full validation starts from that generated backbone and runs
ProteinMPNN, ESMFold, structural post-processing, and reduction:

```bash
qsub -A FRAME-IDP -q debug aurora/genie3_evaluate.pbs
```

For normal work, source `activate_genie3_aurora.sh` and use the usual `genie3
generate`, `genie3 evaluate`, or `genie3 run` commands. Begin with one XPU
(`GENIE3_NUM_DEVICES=1`). The Lightning port uses `xccl` when more than one
XPU is requested; validate the target configuration on one XPU before scaling
out.
