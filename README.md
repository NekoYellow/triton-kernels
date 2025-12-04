# triton-kernels

# triton_kernels

A small personal library of Triton kernels, organized like a competitive programming template repo.
Each operator (op) has:

* A Triton implementation
* A naive or baseline PyTorch implementation
* A common benchmarking entry point

Which enables quick compare of correctness and performance.

## Directory structure

Rough layout:

```text
triton_kernels/
  README.md

  kernels/
    __init__.py

    ops/
      __init__.py
      base.py           # BaseOp class and operator registry
      matmul.py
      softmax.py
      # more ops...

  benchmarks/
    profiler.py         # Generic profiling script

  tests/
    test_matmul.py
    test_softmax.py
    # simple correctness tests
```

## How operators are defined

Each op is a Python class that:

* Inherits from `BaseOp`
* Is registered via `@register_op`
* Implements three static methods:

```python
@register_op
class SomeOp(BaseOp):
    config = OpConfig(
        name="some_op",
        description="Short text explaining the op",
    )

    @classmethod
    def add_arguments(cls, parser):
        # Add op-specific CLI arguments if needed
        parser.add_argument("--M", type=int, default=1024)

    @staticmethod
    def make_inputs(args):
        # Return a tuple of input tensors for this op
        ...

    @staticmethod
    def torch_impl(*inputs):
        # Reference / baseline PyTorch implementation
        ...

    @staticmethod
    def triton_impl(*inputs):
        # Triton implementation wrapping your kernel
        ...
```

The registry in `base.py` keeps a mapping `name -> op class`, which is used by the profiling script.

## Profiling an operator

Use the generic profiler:

```bash
python -m benchmarks.profiler \
  --op vec_add \
  --backend both \
  --dtype float16 \
  --device cuda \
  --numel 10000000 \
  --check
```

Key arguments:

* `--op`: operator name, must match `OpConfig.name`
* `--backend`: `torch`, `triton`, or `both`
* `--dtype`: torch dtype name (`float16`, `float32`, `bfloat16`, ...)
* `--device`: `cuda`, `cuda:0`, etc.
* `--warmup`: warmup iterations before timing
* `--iters`: timed iterations
* Shape-related args such as `--M`, `--N`, `--K`, `--B`, `--S`, `--numel` are passed to `make_inputs`

For example, if you define a `matmul` op:

```bash
python -m benchmarks.profiler \
  --op matmul \
  --backend both \
  --dtype float16 \
  --device cuda \
  --M 2048 --N 2048 --K 2048 \
  --check
```

The script will:

1. Build input tensors via `make_inputs`
2. Optionally check correctness (`torch_impl` vs `triton_impl`)
3. Benchmark both implementations and print timing and speedup

## Adding a new operator

1. Create a new file in `kernels/ops`, for example `my_op.py`
2. Implement your Triton kernel and a Python wrapper
3. Define an op class inheriting `BaseOp` and decorated with `@register_op`
4. Import your class in `triton_kernels/ops/__init__.py` so it gets registered
5. Run `profiler.py` with `--op my_op` to profile it

