from __future__ import annotations

import torch
import triton
import triton.language as tl

from .base import BaseOp, OpConfig, register_op


# ===================== Kernel ===================== #

@triton.jit
def vec_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    assert x.dtype == y.dtype

    n_elements = x.numel()
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, block_size),)

    vec_add_kernel[grid](
        x, y, out,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


# ===================== Op ===================== #

@register_op
class VecAddOp(BaseOp):
    """
    Elementwise vector addition.
    z = x + y
    """
    config = OpConfig(
        name="vec_add",
        description="out = x + y, elementwise",
    )

    @classmethod
    def add_arguments(cls, parser) -> None:
        parser.add_argument("--numel", type=int, default=1_000_000,
                            help="Number of elements of x and y")

    @staticmethod
    def make_inputs(args):
        device = torch.device(args.device)
        dtype = getattr(torch, args.dtype)
        numel = args.numel

        x = torch.randn(numel, device=device, dtype=dtype)
        y = torch.randn(numel, device=device, dtype=dtype)
        return (x, y)

    @staticmethod
    def torch_impl(x, y):
        return x + y

    @staticmethod
    def triton_impl(x, y):
        return triton_add(x, y)
