# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""Exercise the Boltz-specific Lightning XPU adapter on an Aurora compute node."""

from __future__ import annotations

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch.utils.data import DataLoader, TensorDataset

from boltz.runtime.xpu import XPUAccelerator


class TinyModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, _batch_index):
        features, targets = batch
        return torch.nn.functional.mse_loss(self.layer(features), targets)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if not torch.xpu.is_available():
    raise RuntimeError("No XPU is visible. Submit this through Aurora PBS.")

model = TinyModule()
data = TensorDataset(torch.ones(4, 4), torch.zeros(4, 1))
strategy = SingleDeviceStrategy(
    device=torch.device("xpu", 0), accelerator=XPUAccelerator()
)
trainer = Trainer(
    accelerator="auto",
    strategy=strategy,
    devices=1,
    precision="bf16-mixed",
    max_epochs=1,
    limit_train_batches=1,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(model, DataLoader(data, batch_size=2))
with torch.autocast("cuda", enabled=False):
    torch.ones(1, device="xpu").add_(1)
torch.xpu.synchronize()
if model.layer.weight.device.type != "xpu":
    raise RuntimeError(f"Boltz model did not reach XPU: {model.layer.weight.device}")

print("Boltz Lightning XPU smoke passed.")
