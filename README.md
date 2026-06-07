# PINN from scratch

> A minimal automatic-differentiation engine and GPU runtime, built from scratch
> (NumPy + Numba/CUDA) to **understand what lives under PyTorch** — then used to
> solve a 2-D heat equation with a Physics-Informed Neural Network (PINN).

This is **not** an attempt to compete with PyTorch or cuBLAS. It is a teaching
implementation: a reverse-mode autograd `tensor`, a hand-written GPU memory
allocator, and hand-written CUDA kernels (matmul, reductions, elementwise),
assembled into an MLP that learns the solution of a PDE from its residual alone.
A short [`Production perspective`](#production-perspective) section documents
exactly where this sits relative to the modern stack and why production looks
different.

## The problem

2-D heat equation with a source term:

```
∂u/∂t = Δu + x·t²·sin(y),   0 < x < 1,   0 < y < π/2,   t > 0
∂u/∂x|_{x=0} = ∂u/∂x|_{x=1} = 0        (Neumann)
u|_{y=0} = 0,   ∂u/∂y|_{y=π/2} = 0      (Dirichlet / Neumann)
u|_{t=0} = 0                            (initial condition)
```

The PINN minimizes the PDE residual plus the boundary/initial residuals, with no
labelled data. Results are validated against a closed-form Fourier-series
solution.

## Status

🚧 Work in progress.

## Quickstart

```bash
poetry install                 # core (CPU) deps
poetry run pytest              # runs on CPU, no GPU required
```

## Layout

```
src/pinn/backend/   CPU (NumPy) and CUDA (Numba) compute backends + GPU allocator
src/pinn/core/      reverse-mode autograd tensor
src/pinn/nn/        Linear, activations, MLP, Adam
src/pinn/pde/       heat-equation PINN loss + analytic reference
tests/              gradient checks, kernel/allocator/accuracy tests
examples/           runnable demo + a compact PyTorch reference
benchmarks/         matmul: hand-written Numba vs CuPy/cuBLAS vs Triton vs NumPy
docs/report/        LaTeX report (RU + EN translation)
```

## Production perspective

_(to be written — the honest map: cuBLAS/cuDNN/Triton/CUTLASS, the thread-vs-tile
programming model, and why this is slower than PyTorch — orchestration and tiny
matrices, not the kernel language.)_

## License

MIT — see [LICENSE](LICENSE).
