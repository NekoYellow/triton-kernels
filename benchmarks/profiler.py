from __future__ import annotations

import argparse
import os
import time

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule as profiler_schedule,
    tensorboard_trace_handler,
)

from kernels.ops import OP_REGISTRY, BaseOp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Triton op profiler (simple)")

    # Core options
    parser.add_argument("--op", type=str, required=True,
                        help="Name of the op to benchmark")
    parser.add_argument("--backend", type=str, default="both",
                        choices=["torch", "triton", "both"],
                        help="Which backend(s) to benchmark")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Torch dtype name: float16, float32, bfloat16, etc.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device, e.g., cuda or cuda:0")
    parser.add_argument("--check", action="store_true",
                        help="Whether to run correctness check")

    # Simple microbenchmark options
    parser.add_argument("--warmup-iters", type=int, default=10,
                        help="Warmup iterations for simple wall-clock timing")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations for simple wall-clock timing")

    # Common shape parameters (ops can decide which ones to use)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--S", type=int, default=2048)
    parser.add_argument("--numel", type=int, default=1_000_000)

    # Profiler schedule options (simple)
    parser.add_argument("--profiler-warmup-steps", type=int, default=2,
                        help="Warmup steps used by torch.profiler schedule")
    parser.add_argument("--profiler-active-steps", type=int, default=8,
                        help="Active steps used by torch.profiler schedule")

    # Trace output options
    parser.add_argument("--trace-dir", type=str, default="traces",
                        help="Base directory to export Chrome traces "
                             "(TensorBoard-compatible format)")

    return parser


def benchmark(fn, inputs, warmup: int, iters: int) -> float:
    """Simple wall-clock benchmark with warmup and timed iterations."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fn(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed region
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
    Supports single tensor or a tuple/list of tensors.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        diff = (a - b).abs()
        return float(diff.max().item())

    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise ValueError("Output tuples have different lengths.")
        max_diff = 0.0
        for t1, t2 in zip(a, b):
            d = tensors_close(t1, t2, rtol=rtol, atol=atol)
            max_diff = max(max_diff, d)
        return max_diff

    raise TypeError("Unsupported output type for tensors_close.")


def run_profiler(fn, inputs, args, label: str) -> None:
    """
    Run torch.profiler with a simple schedule and export Chrome trace.
    By default:
      - wait steps: 1
      - warmup steps: args.profiler_warmup_steps
      - active steps: args.profiler_active_steps
      - repeat: 1
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    sched = profiler_schedule(
        wait=1,
        warmup=args.profiler_warmup_steps,
        active=args.profiler_active_steps,
        repeat=1,
    )

    # trace_dir/elementwise_add_triton/ will contain the profiler traces
    base_dir = args.trace_dir
    trace_path = os.path.join(base_dir, label)
    os.makedirs(trace_path, exist_ok=True)

    trace_handler = tensorboard_trace_handler(trace_path)

    total_steps = 1 + args.profiler_warmup_steps + args.profiler_active_steps

    with profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=trace_handler,
        record_shapes=True,      # shapes are useful in practice
        profile_memory=False,    # can turn on later if needed
        with_stack=False,        # keep overhead low by default
    ) as prof:
        for step in range(total_steps):
            with torch.no_grad():
                _ = fn(*inputs)
            prof.step()

    print(f"[profiler] Finished profiling {label}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.op not in OP_REGISTRY:
        raise ValueError(
            f"Unknown op: {args.op}. "
            f"Available: {list(OP_REGISTRY.keys())}"
        )

    op_cls: type[BaseOp] = OP_REGISTRY[args.op]

    device = torch.device(args.device)
    args.device = device

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Inputs are created by the op implementation
    inputs = op_cls.make_inputs(args)

    # Correctness check
    if args.check or args.backend == "both":
        ref = op_cls.torch_impl(*inputs)
        out = op_cls.triton_impl(*inputs)
        max_diff = tensors_close(out, ref)
        print(f"[check] max abs diff: {max_diff:.3e}")

    # Simple microbenchmark
    if args.backend in ("torch", "both"):
        torch_time = benchmark(
            lambda *xs: op_cls.torch_impl(*xs),
            inputs,
            warmup=args.warmup_iters,
            iters=args.iters,
        )
        print(f"[torch]   {args.op}: {torch_time:.3f} ms")

    if args.backend in ("triton", "both"):
        triton_time = benchmark(
            lambda *xs: op_cls.triton_impl(*xs),
            inputs,
            warmup=args.warmup_iters,
            iters=args.iters,
        )
        print(f"[triton]  {args.op}: {triton_time:.3f} ms")

    if args.backend == "both":
        speedup = torch_time / triton_time
        print(f"[summary] speedup triton / torch: {speedup:.2f}x")

    # Always run profiler and export traces
    if args.backend in ("torch", "both"):
        run_profiler(
            lambda *xs: op_cls.torch_impl(*xs),
            inputs,
            args,
            label=f"{args.op}_torch",
        )

    if args.backend in ("triton", "both"):
        run_profiler(
            lambda *xs: op_cls.triton_impl(*xs),
            inputs,
            args,
            label=f"{args.op}_triton",
        )


if __name__ == "__main__":
    main()
