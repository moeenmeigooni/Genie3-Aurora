# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""Verify that the isolated ColabFold JAX stack executes on an Aurora XPU."""

from __future__ import annotations

import jax
import jax.numpy as jnp


devices = jax.devices("sycl")
if not devices:
    raise RuntimeError("Intel OpenXLA did not expose an Aurora SYCL device.")

matrix = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
result = jax.jit(lambda value: value @ value)(matrix)
result.block_until_ready()
if result.device.platform != "sycl":
    raise RuntimeError(f"Expected a SYCL XPU result, received {result.device!s}")

print(f"ColabFold OpenXLA XPU smoke passed on {result.device!s}.")
