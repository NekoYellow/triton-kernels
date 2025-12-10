from __future__ import annotations

import torch
import triton
import triton.language as tl

from .base import BaseOp, OpConfig, register_op

DEVICE = torch.device("cuda:0")
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
PROGRAM_PER_SM = 4


# ===================== Kernel ===================== #

@triton.autotune(
    configs=[
        triton.Config(
            {}, # {"BLOCK_SIZE": 1024},
            num_warps=4,
            num_stages=2,
            num_ctas=1,
        ),
        triton.Config(
            {}, # {"BLOCK_SIZE": 1024},
            num_warps=8,
            num_stages=2,
            num_ctas=2,
        ),
        triton.Config(
            {}, # {"BLOCK_SIZE": 1024},
            num_warps=8,
            num_stages=4,
            num_ctas=4,
        ),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def softmax_kernel(
    in_ptr, out_ptr,
    in_row_stride, out_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = in_ptr + row_idx * in_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        in_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(in_ptrs, mask=mask, other=-float("inf"))
        row = row - tl.max(row, axis=0)
        numerator = tl.exp(row)
        denumerator = tl.sum(numerator, axis=0)
        out = numerator / denumerator
        out_row_start_ptr = out_ptr + row_idx * out_row_stride
        out_ptrs = out_row_start_ptr + col_offsets
        tl.store(out_ptrs, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    y = torch.empty_like(x)

    def grid(meta):
        num_programs = NUM_SM * PROGRAM_PER_SM
        num_programs = min(num_programs, n_rows)
        return (num_programs, )

    softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ===================== Op ===================== #

@register_op
class SoftmaxOp(BaseOp):
    """
    Row-wise softmax.
    For each row x, computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    config = OpConfig(
        name="softmax",
        description="row-wise softmax",
    )

    @classmethod
    def add_arguments(cls, parser) -> None:
        parser.add_argument("--N", type=int, default=1024,
                            help="Number of rows of x")
        parser.add_argument("--M", type=int, default=1024,
                            help="Number of columns of x")

    @staticmethod
    def make_inputs(args):
        device = torch.device(args.device)
        dtype = getattr(torch, args.dtype)
        N, M = args.N, args.M

        x = torch.randn((N, M), device=device, dtype=dtype)
        return (x, )

    @staticmethod
    def torch_impl(x):
        return torch.softmax(x, dim=-1)

    @staticmethod
    def triton_impl(x):
        return triton_softmax(x)
