from __future__ import annotations

import argparse
import time

import torch

from triton_kernels.ops import OP_REGISTRY, BaseOp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Triton op profiler")

    parser.add_argument("--op", type=str, required=True,
                        help="Name of the op to benchmark")
    parser.add_argument("--backend", type=str, default="both",
                        choices=["torch", "triton", "both"],
                        help="Which backend(s) to benchmark")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Torch dtype name: float16, float32, bfloat16, etc.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device, e.g., cuda or cuda:0")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of timed iterations")
    parser.add_argument("--check", action="store_true",
                        help="Whether to run correctness check")

    # common shape params
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--S", type=int, default=2048)
    parser.add_argument("--numel", type=int, default=1_000_000)

    return parser


def benchmark(fn, inputs, warmup: int, iters: int) -> float:
    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timed
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = fn(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / iters
    return avg_ms


def tensors_close(a, b, rtol=1e-3, atol=1e-3) -> float:
    """
    Simple correctness check, returning max abs diff.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        diff = (a - b).abs()
        return float(diff.max().item())
    # 可以根据需要扩展到 tuple/list
    raise TypeError("Only single-tensor outputs are currently supported")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.op not in OP_REGISTRY:
        raise ValueError(f"Unknown op: {args.op}. "
                         f"Available: {list(OP_REGISTRY.keys())}")

    op_cls: type[BaseOp] = OP_REGISTRY[args.op]

    device = torch.device(args.device)
    args.device = device

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    inputs = op_cls.make_inputs(args)

    if args.check or args.backend == "both":
        ref = op_cls.torch_impl(*inputs)
        out = op_cls.triton_impl(*inputs)
        max_diff = tensors_close(out, ref)
        print(f"[check] max abs diff: {max_diff:.3e}")

    if args.backend in ("torch", "both"):
        torch_time = benchmark(
            lambda *xs: op_cls.torch_impl(*xs),
            inputs,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"[torch]   {args.op}: {torch_time:.3f} ms")

    if args.backend in ("triton", "both"):
        triton_time = benchmark(
            lambda *xs: op_cls.triton_impl(*xs),
            inputs,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"[triton]  {args.op}: {triton_time:.3f} ms")

    if args.backend == "both":
        speedup = torch_time / triton_time
        print(f"[summary] speedup triton / torch: {speedup:.2f}x")


if __name__ == "__main__":
    main()
