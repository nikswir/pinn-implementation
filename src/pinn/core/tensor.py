"""A minimal reverse-mode automatic-differentiation engine.

``Tensor`` wraps a 2-D ``float32`` array and records the operations applied to
it, so that gradients can be obtained by backpropagation. Crucially the
backward functions are themselves expressed with ``Tensor`` operations, so the
gradient is *also* a differentiable graph — that is what lets us take the
**second** derivatives a PINN needs (``u_xx``, ``u_yy``).

The design mirrors PyTorch's ``autograd`` in miniature:

- :meth:`Tensor.backward` accumulates ``.grad`` on leaves (first order).
- :func:`grad` returns gradients as ``Tensor`` objects and, with
  ``create_graph=True``, keeps them differentiable for higher-order derivatives.

All number crunching is delegated to the active compute backend
(:mod:`pinn.backend`), so the same graph runs on NumPy (CPU) or CUDA.
"""

from __future__ import annotations

import numpy as np

from types import ModuleType
from typing import Any, overload
from collections.abc import Callable, Sequence

from pinn import backend


def _b() -> ModuleType:
    return backend.active()


class Tensor:
    """A node in the computation graph holding a 2-D float32 array."""

    __slots__ = ("data", "requires_grad", "grad", "_parents")

    data: np.ndarray
    requires_grad: bool
    grad: Tensor | None
    _parents: list[tuple[Tensor, Callable[[Tensor], Tensor]]]

    def __init__(self, value: Any, requires_grad: bool = False) -> None:
        self.data = _b().asarray(value)
        self.requires_grad = requires_grad
        self.grad = None
        # list of (parent_tensor, grad_fn); grad_fn(upstream) -> Tensor
        self._parents = []

    ########################################
    #         construction helpers         #
    ########################################

    @classmethod
    def _wrap(
        cls,
        data: np.ndarray,
        parents: list[tuple[Tensor, Callable[[Tensor], Tensor]]],
        requires_grad: bool,
    ) -> Tensor:
        t = cls.__new__(cls)
        t.data = data
        t.requires_grad = requires_grad
        t.grad = None
        t._parents = parents
        return t

    def _ensure(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    ########################################
    #             numpy / repr             #
    ########################################

    def numpy(self) -> np.ndarray:
        result: np.ndarray = _b().to_numpy(self.data)
        return result

    def item(self) -> float:
        return float(self.numpy().reshape(-1)[0])

    def detach(self) -> Tensor:
        """Return a graph-free copy sharing the same values."""
        return Tensor._wrap(self.data, parents=[], requires_grad=False)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    ########################################
    #        elementwise binary ops        #
    ########################################

    def __add__(self, other: Any) -> Tensor:
        other = self._ensure(other)
        out = Tensor._wrap(
            _b().add(self.data, other.data),
            parents=[],
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: _unbroadcast(g, self.shape)))
        if other.requires_grad:
            out._parents.append((other, lambda g: _unbroadcast(g, other.shape)))
        return out

    def __mul__(self, other: Any) -> Tensor:
        other = self._ensure(other)
        out = Tensor._wrap(
            _b().mul(self.data, other.data),
            parents=[],
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if self.requires_grad:
            out._parents.append(
                (self, lambda g: _unbroadcast(g * other, self.shape)),
            )
        if other.requires_grad:
            out._parents.append(
                (other, lambda g: _unbroadcast(g * self, other.shape)),
            )
        return out

    def __pow__(self, power: int | float) -> Tensor:
        if not isinstance(power, (int | float)):
            raise TypeError("only scalar exponents are supported")
        out = Tensor._wrap(
            _b().power(self.data, float(power)),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            out._parents.append(
                (self, lambda g: g * (self ** (power - 1)) * power),
            )
        return out

    def __truediv__(self, other: Any) -> Tensor:
        denom = self._ensure(other)
        return self * (denom**-1)

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._ensure(other) * (self**-1)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Any) -> Tensor:
        return self + (-self._ensure(other))

    # reflected scalar ops
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: Any) -> Tensor:
        return self._ensure(other) + (-self)

    ########################################
    #          matmul / transpose          #
    ########################################

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor._wrap(
            _b().matmul(self.data, other.data),
            parents=[],
            requires_grad=self.requires_grad or other.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: g @ other.T()))
        if other.requires_grad:
            out._parents.append((other, lambda g: self.T() @ g))
        return out

    def T(self) -> Tensor:
        out = Tensor._wrap(
            _b().transpose(self.data),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: g.T()))
        return out

    ########################################
    #           unary functions            #
    ########################################

    def sin(self) -> Tensor:
        out = Tensor._wrap(
            _b().sin(self.data),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: g * self.cos()))
        return out

    def cos(self) -> Tensor:
        out = Tensor._wrap(
            _b().cos(self.data),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: g * (-self.sin())))
        return out

    def ln(self) -> Tensor:
        out = Tensor._wrap(
            _b().log(self.data),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            out._parents.append((self, lambda g: g * (self**-1)))
        return out

    def sigmoid(self) -> Tensor:
        out = Tensor._wrap(
            _b().sigmoid(self.data),
            parents=[],
            requires_grad=self.requires_grad,
        )
        if self.requires_grad:
            # d/dx sigma = sigma * (1 - sigma); reuse the forward value `out`.
            out._parents.append((self, lambda g: g * (out * (1.0 - out))))
        return out

    ########################################
    #              reductions              #
    ########################################

    def sum(self, axis: int | None = None) -> Tensor:
        if axis is None:
            data = _b().sum_all(self.data)
        elif axis in (0, 1):
            data = _b().sum_axis(self.data, axis)
        else:
            raise ValueError(f"unsupported sum axis {axis!r}")
        out = Tensor._wrap(data, parents=[], requires_grad=self.requires_grad)
        if self.requires_grad:
            ones = Tensor(_b().full(self.shape, 1.0), requires_grad=False)
            out._parents.append((self, lambda g: g * ones))
        return out

    ########################################
    #               indexing               #
    ########################################

    def __getitem__(self, item: Any) -> Tensor:
        """Support column extraction ``t[:, j]`` -> shape (rows, 1).

        Implemented as a matmul with a one-hot selector, so the gradient (a
        scatter back into the original shape) is handled by the matmul rule and
        remains differentiable for higher-order derivatives.
        """
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], int)
        ):
            rows_sel, col = item
            if not (isinstance(rows_sel, slice) and rows_sel == slice(None)):
                raise IndexError(
                    "only full-row column selection t[:, j] is supported",
                )
            selector = np.zeros((self.shape[1], 1), dtype=np.float32)
            selector[col, 0] = 1.0
            return self @ Tensor(selector, requires_grad=False)
        raise IndexError("only t[:, j] indexing is supported")

    ########################################
    #               backward               #
    ########################################

    def backward(self, grad_output: Tensor | None = None) -> None:
        """Populate ``.grad`` on every leaf reachable from ``self``."""
        seed = (
            grad_output
            if grad_output is not None
            else Tensor(_b().full(self.shape, 1.0))
        )
        grads: dict[int, Tensor] = {id(self): seed}
        for t in reversed(_topo(self)):
            g = grads.get(id(t))
            if g is None:
                continue
            for parent, grad_fn in t._parents:
                contrib = grad_fn(g)
                prev = grads.get(id(parent))
                grads[id(parent)] = contrib if prev is None else prev + contrib
            if not t._parents and t.requires_grad:  # leaf
                t.grad = grads[id(t)].detach()


########################################
#         broadcasting helper          #
########################################


def _unbroadcast(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Sum ``grad`` over dimensions that were broadcast to match ``shape``."""
    if grad.shape == shape:
        return grad
    if shape[0] == 1 and grad.shape[0] != 1:
        grad = grad.sum(axis=0)
    if shape[1] == 1 and grad.shape[1] != 1:
        grad = grad.sum(axis=1)
    return grad


########################################
#         gradient computation         #
########################################


def _topo(output: Tensor) -> list[Tensor]:
    order: list[Tensor] = []
    seen: set[int] = set()

    def visit(t: Tensor) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        for parent, _ in t._parents:
            visit(parent)
        order.append(t)

    visit(output)
    return order


@overload
def grad(
    output: Tensor,
    inputs: Tensor,
    grad_output: Tensor | None = ...,
    create_graph: bool = ...,
) -> Tensor: ...


@overload
def grad(
    output: Tensor,
    inputs: Sequence[Tensor],
    grad_output: Tensor | None = ...,
    create_graph: bool = ...,
) -> list[Tensor | None]: ...


def grad(
    output: Tensor,
    inputs: Tensor | Sequence[Tensor],
    grad_output: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor | list[Tensor | None] | None:
    """Return d(output)/d(input) for each tensor in ``inputs`` as ``Tensor``.

    Gradients incoming to a node are summed *before* propagating further (proper
    reverse-mode accumulation). With ``create_graph=True`` the returned
    gradients stay attached to the graph, enabling second-order derivatives.
    """
    if isinstance(inputs, Tensor):
        input_list = [inputs]
        single = True
    else:
        input_list = list(inputs)
        single = False

    seed = (
        grad_output
        if grad_output is not None
        else Tensor(_b().full(output.shape, 1.0))
    )
    grads: dict[int, Tensor] = {id(output): seed}

    for t in reversed(_topo(output)):
        g = grads.get(id(t))
        if g is None:
            continue
        for parent, grad_fn in t._parents:
            contrib = grad_fn(g)
            prev = grads.get(id(parent))
            grads[id(parent)] = contrib if prev is None else prev + contrib

    out: list[Tensor | None] = []
    for inp in input_list:
        g = grads.get(id(inp))
        if g is not None and not create_graph:
            g = g.detach()
        out.append(g)
    return out[0] if single else out
