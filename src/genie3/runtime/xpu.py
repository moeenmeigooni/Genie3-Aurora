# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""PyTorch Lightning accelerator adapter for Intel XPU devices."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class XPUAccelerator(Accelerator):
    """Run Lightning modules on PyTorch's native Intel XPU backend."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device} instead")
        torch.xpu.set_device(device)

    def get_device_stats(self, device: torch.device) -> dict[str, Any]:
        return torch.xpu.memory_stats(device)

    def teardown(self) -> None:
        if self.is_available():
            torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: int | str | list[int] | None) -> list[int] | None:
        """Parse Lightning's standard integer or comma-delimited device syntax."""
        if devices is None:
            return None
        available = XPUAccelerator.auto_device_count()
        if isinstance(devices, str):
            value = devices.strip().lower()
            if value == "auto":
                devices = -1
            elif "," in value:
                devices = [int(part.strip()) for part in value.split(",") if part.strip()]
            else:
                devices = int(value)
        if isinstance(devices, int):
            if devices == 0:
                return None
            ids = list(range(available)) if devices == -1 else list(range(devices))
        else:
            ids = list(devices)
        if not ids:
            return None
        if len(set(ids)) != len(ids):
            raise MisconfigurationException(f"Duplicate XPU device IDs: {ids}")
        if any(index < 0 or index >= available for index in ids):
            raise MisconfigurationException(
                f"Requested XPU devices {ids}, but {available} XPU device(s) are visible."
            )
        return ids

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", index) for index in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(cls.name(), cls, description=cls.__name__)
