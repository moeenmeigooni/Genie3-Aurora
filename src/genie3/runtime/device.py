# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""Backend-neutral device helpers used by Genie3 runtime workflows."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch


def accelerator_name() -> str:
    """Return the requested usable accelerator (``xpu``, ``cuda``, or ``cpu``).

    ``GENIE3_ACCELERATOR=auto`` (the default) prefers an Intel XPU when one is
    visible, then CUDA, then CPU. On Aurora, setting ``GENIE3_ACCELERATOR=xpu``
    makes an accidental CPU fallback an immediate, clear error.
    """
    requested = os.environ.get("GENIE3_ACCELERATOR", "auto").lower().strip()
    if requested not in {"auto", "xpu", "cuda", "cpu"}:
        raise ValueError(
            "GENIE3_ACCELERATOR must be one of auto, xpu, cuda, or cpu; "
            f"received {requested!r}."
        )

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        if xpu_available:
            return "xpu"
        if cuda_available:
            return "cuda"
        return "cpu"
    if requested == "xpu" and not xpu_available:
        raise RuntimeError(
            "GENIE3_ACCELERATOR=xpu was requested, but no Intel XPU is visible. "
            "Run inside an Aurora GPU PBS allocation with the frameworks module loaded."
        )
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("GENIE3_ACCELERATOR=cuda was requested, but CUDA is unavailable.")
    return requested


def lightning_accelerator():
    """Return a Lightning accelerator compatible with the selected backend."""
    name = accelerator_name()
    if name == "xpu":
        from genie3.runtime.xpu import XPUAccelerator

        return XPUAccelerator()
    return name


def xpu_lightning_strategy(
    accelerator: Any,
    devices: int | None,
    *,
    timeout: timedelta | None = None,
    find_unused_parameters: bool = False,
):
    """Build a Lightning strategy with explicit XPU root devices.

    Lightning's built-in single-device strategy defaults its root device to
    CPU for an unregistered accelerator instance. Supplying the XPU device
    explicitly avoids that fallback. The DDP variant also supplies every
    per-node XPU device and Aurora's ``xccl`` backend.
    """
    from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

    count = 1 if devices is None else int(devices)
    if count < 1:
        raise ValueError(f"XPU device count must be positive, received {count}.")
    parallel_devices = [torch.device("xpu", index) for index in range(count)]
    if count == 1:
        return SingleDeviceStrategy(device=parallel_devices[0], accelerator=accelerator)

    kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "parallel_devices": parallel_devices,
        "process_group_backend": ddp_backend(),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if find_unused_parameters:
        kwargs["find_unused_parameters"] = True
    return DDPStrategy(**kwargs)


def ddp_backend() -> str:
    """Return the appropriate distributed backend for the selected device."""
    name = accelerator_name()
    if name == "xpu":
        return os.environ.get("GENIE3_XPU_DISTRIBUTED_BACKEND", "xccl")
    if name == "cuda":
        return "nccl"
    return "gloo"


def device_string(index: int) -> str:
    """Return a fully-qualified device string for a worker index."""
    name = accelerator_name()
    return "cpu" if name == "cpu" else f"{name}:{index}"


def synchronize_and_empty_cache() -> None:
    """Synchronize and release the active accelerator's unused memory cache."""
    name = accelerator_name()
    if name == "cpu":
        return
    module = getattr(torch, name)
    module.synchronize()
    module.empty_cache()
