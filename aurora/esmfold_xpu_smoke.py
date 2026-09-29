# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""Exercise Genie3's ESMFold evaluation handler on one Aurora XPU tile."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from genie3.evaluation.model.fold.esmfold import ESMFoldHandler


if not torch.xpu.is_available():
    raise RuntimeError("No XPU is visible. Submit this through Aurora PBS.")

with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tempdir:
    structures_dir = Path(tempdir)
    (structures_dir / "input.fasta").write_text(
        ">esmfold_xpu_smoke-resample_0\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n"
    )
    handler = ESMFoldHandler(
        version="unconditional",
        mode="single",
        device="xpu:0",
        preload=True,
    )
    handler.fold(str(structures_dir))
    pdb = structures_dir / "esmfold_xpu_smoke-resample_0" / "esmfold_xpu_smoke-resample_0.pdb"
    scores = structures_dir / "esmfold_xpu_smoke-resample_0" / "scores_esmfold_xpu_smoke-resample_0.json"
    if not pdb.is_file() or pdb.stat().st_size == 0:
        raise RuntimeError("ESMFold did not produce a PDB output.")
    if not scores.is_file() or scores.stat().st_size == 0:
        raise RuntimeError("ESMFold did not produce confidence scores.")

torch.xpu.synchronize()
print("Genie3 ESMFold XPU smoke passed.")
