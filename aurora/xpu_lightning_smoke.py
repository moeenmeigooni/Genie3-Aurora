# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""Run a minimal real Lightning test loop on an Aurora Intel XPU."""

import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from genie3.runtime.device import xpu_lightning_strategy
from genie3.runtime.xpu import XPUAccelerator


class SmokeModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(8, 4)

    def test_step(self, batch, batch_idx):
        del batch_idx
        x, = batch
        output = self.layer(x)
        if output.device.type != "xpu":
            raise RuntimeError(f"Expected XPU output, got {output.device}")
        self.log("xpu_smoke_mean", output.mean())


if not torch.xpu.is_available():
    raise RuntimeError("No XPU is visible. Submit this through Aurora PBS, not on a UAN.")

print(f"torch={torch.__version__}")
print(f"xpu_count={torch.xpu.device_count()}")
print(f"xpu_0={torch.xpu.get_device_name(0)}")

tensor = torch.randn(128, 128, device="xpu")
assert (tensor @ tensor.T).device.type == "xpu"

loader = DataLoader(TensorDataset(torch.randn(16, 8)), batch_size=4)
accelerator = XPUAccelerator()
trainer = Trainer(
    strategy=xpu_lightning_strategy(accelerator, devices=1),
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)
trainer.test(SmokeModule(), dataloaders=loader, verbose=False)
torch.xpu.synchronize()
print("Genie3 Lightning XPU smoke passed.")
