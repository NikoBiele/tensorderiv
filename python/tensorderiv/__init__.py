"""tensorderiv: gradient, divergence, curl and Laplacian on scattered points, a Python
interface to a Rust core."""

import operator

import numpy as np
from scipy.spatial import cKDTree

from ._core import (
    gradient_kernel,
    laplacian_kernel,
    minimum_neighbours,
    stencil_set_weights,
    thread_count,
)

__all__ = [
    "StencilSet",
    "minimum_neighbours",
    "gradient",
    "divergence",
    "curl",
    "laplacian",
    "thread_count",
]


class StencilSet:
    """The nearest neighbours and stencil weights of every point in a point cloud, computed
    once, for use by the derivative operators.

    Parameters
    ----------
    points : array_like, shape (n_points, dims)
        The point coordinates, one row per point.
    order : {2, 1}, optional
        Accuracy order of the operators. With 2, the default, the weights also cancel the
        leading error term, which makes every operator second-order accurate. 1 gives
        first-order operators from smaller stencils, for small point clouds or cheaper builds.
    k : int, optional
        Neighbours per point. The minimum is ``minimum_neighbours(dims, order)``: 9 in 2D and
        19 in 3D for order 2, 5 in 2D and 9 in 3D for order 1. The default is twice the
        minimum, because stencils at the minimum size are ill-conditioned.
    threaded : bool, optional
        Find the neighbours and compute the weights on all cores (default True). Set it to
        False when calling from code that is already parallel.

    Attributes
    ----------
    points : ndarray, shape (n_points, dims)
        The point coordinates, as float64.
    neighbours : ndarray, shape (n_points, k)
        Indices of each point's neighbours, nearest first, excluding the point itself.
    weights : ndarray, shape (n_points, k)
        The weights of each point's neighbours, in the same order.
    order : int
        The accuracy order the weights were computed for.
    """

    def __init__(self, points, order=2, k=None, threaded=True):
        # the coordinates as a float64 array with one row per point
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"points must be a 2D array with one row per point, got {points.ndim} dimensions")
        n_points, dims = points.shape
        if not np.all(np.isfinite(points)):
            raise ValueError("points must not contain NaN or infinite coordinates")
        # minimum_neighbours also rejects any order other than 1 or 2
        k_min = minimum_neighbours(dims, order)
        k = 2 * k_min if k is None else operator.index(k)  # operator.index accepts integers only
        if k < k_min:
            raise ValueError(f"k = {k} is below the minimum {k_min} for order {order} in {dims} dimensions")
        if k >= n_points:
            raise ValueError(f"k = {k} must be smaller than the number of points ({n_points})")

        # k + 1 nearest per point, the point itself included, sorted by distance
        workers = -1 if threaded else 1                     # -1: all cores
        _, candidates = cKDTree(points).query(points, k + 1, workers=workers)
        neighbours = _exclude_self(candidates, k)

        self.points = points
        self.neighbours = neighbours
        self.weights = stencil_set_weights(points, neighbours, order, threaded)
        self.order = order

    def __repr__(self):
        n_points, dims = self.points.shape
        k = self.neighbours.shape[1]
        return f"StencilSet({n_points} points in {dims}D, order={self.order}, k={k})"


def _exclude_self(candidates, k):
    # Each row of candidates holds a point's k + 1 nearest points, normally starting with the
    # point itself. Remove the point itself; if another point sits at exactly the same
    # coordinates, the point itself may not be among the candidates, and the furthest
    # candidate is dropped instead, as in the Julia package.
    n_points = candidates.shape[0]
    keep = candidates != np.arange(n_points)[:, None]      # False where the point itself is
    keep[keep.sum(axis=1) > k, -1] = False                  # rows without the point itself: drop the last
    return candidates[keep].reshape(n_points, k).astype(np.int64)


def gradient(stencils, values, at=None, threaded=True):
    """Gradient ∇⊗Y ≈ Σ_k a_k Δx_k ⊗ ΔY_k of a scalar, vector or tensor field.

    Parameters
    ----------
    stencils : StencilSet
        The stencils of the points the field is sampled at.
    values : array_like, shape (n_points, ...)
        The field, one value per point along the first axis: shape (n_points,) for a scalar
        field, (n_points, n) for a vector field, (n_points, n, m) for a matrix field, and so on.
    at : int, slice or array of ints, optional
        The points to evaluate at, indexed as in NumPy (negative indices count from the end).
        By default, every point.
    threaded : bool, optional
        Spread the points over all cores (default True).

    Returns
    -------
    ndarray, shape (n_targets, dims, ...)
        The derivative direction comes right after the point index:
        ``result[p, a, j...] = ∂Y_j.../∂x_a`` at point p. For a vector field this is the
        transpose of the usual Jacobian. With a single integer ``at``, the point axis is left out.

    The estimate is exact for linear fields with order-1 stencils, and also for quadratic
    fields with order-2 stencils.
    """
    flat, field_shape, targets, single = _prepare(stencils, values, at)
    dims = stencils.points.shape[1]
    result = gradient_kernel(stencils.points, stencils.neighbours, stencils.weights, flat, targets, threaded)
    result = result.reshape(len(targets), dims, *field_shape)  # derivative direction, then the field's own axes
    return result[0] if single else result


def divergence(stencils, values, at=None, threaded=True):
    """Divergence ∇·Y ≈ Σ_k a_k Δx_k · ΔY_k of a vector or tensor field, contracting the
    derivative with the field's first axis.

    Parameters are as for `gradient`; the field's first axis (after the point axis) must have
    the spatial dimension.

    Returns
    -------
    ndarray, shape (n_targets, ...)
        For a vector field, one value per point; for a matrix field of shape (n_points, dims, m),
        shape (n_targets, m) with ``result[p, j] = Σ_a ∂Y_aj/∂x_a``. With a single integer
        ``at``, the point axis is left out, so a vector field gives a single number.

    The divergence is a contraction of the gradient and has the same accuracy.
    """
    _check_stencils(stencils)
    dims = stencils.points.shape[1]
    field_shape = np.shape(values)[1:]
    if len(field_shape) == 0 or field_shape[0] != dims:
        raise ValueError(
            f"divergence needs a field whose first axis after the points has length {dims}, "
            f"but one point's value has shape {field_shape}"
        )
    grad = gradient(stencils, values, at, threaded)
    # contract the derivative direction with the field's first axis; with a single target the
    # point axis is absent, so the two axes to contract come first
    first = 0 if np.ndim(grad) == len(field_shape) + 1 else 1
    return np.trace(grad, axis1=first, axis2=first + 1)


def curl(stencils, values, at=None, threaded=True):
    """Curl ∇×Y ≈ Σ_k a_k Δx_k × ΔY_k of a vector or tensor field in three dimensions, taking
    the cross product with the field's first axis.

    Parameters are as for `gradient`; the points must be three-dimensional, and the field's
    first axis (after the point axis) must have length 3.

    Returns
    -------
    ndarray
        The same shape as the field (for the evaluated points), with
        ``result[p, i, r...] = ε_ijk ∂Y_kr.../∂x_j``. With a single integer ``at``, the point
        axis is left out.

    The curl is a contraction of the gradient and has the same accuracy.
    """
    _check_stencils(stencils)
    if stencils.points.shape[1] != 3:
        raise ValueError(f"curl is only defined in 3 dimensions, but the points have {stencils.points.shape[1]}")
    field_shape = np.shape(values)[1:]
    if len(field_shape) == 0 or field_shape[0] != 3:
        raise ValueError(f"curl needs a field whose first axis after the points has length 3, but one point's value has shape {field_shape}")
    grad = gradient(stencils, values, at, threaded)
    single = np.ndim(grad) == len(field_shape) + 1    # a single target: no point axis
    if single:
        grad = grad[None]                              # add a point axis, removed again below
    # grad[p, j, k, ...] = ∂Y_k.../∂x_j; the curl's components ε_ijk ∂_j Y_k
    result = np.stack([
        grad[:, 1, 2] - grad[:, 2, 1],                 # ∂Y_3/∂x_2 − ∂Y_2/∂x_3
        grad[:, 2, 0] - grad[:, 0, 2],                 # ∂Y_1/∂x_3 − ∂Y_3/∂x_1
        grad[:, 0, 1] - grad[:, 1, 0],                 # ∂Y_2/∂x_1 − ∂Y_1/∂x_2
    ], axis=1)
    return result[0] if single else result


def laplacian(stencils, values, at=None, threaded=True):
    """Laplacian ∇²Y ≈ 2 Σ_k a_k ΔY_k of a scalar, vector or tensor field, applied to each
    component.

    Parameters are as for `gradient`.

    Returns
    -------
    ndarray
        The same shape as the field (for the evaluated points). With a single integer ``at``,
        the point axis is left out, so a scalar field gives a single number.

    The estimate is exact for quadratic fields with order-1 stencils, and also for cubic fields
    with order-2 stencils.
    """
    flat, field_shape, targets, single = _prepare(stencils, values, at)
    result = laplacian_kernel(stencils.points, stencils.neighbours, stencils.weights, flat, targets, threaded)
    result = result.reshape(len(targets), *field_shape)  # the field's own axes
    return result[0] if single else result


def _prepare(stencils, values, at):
    # Check the inputs of an operator and bring them into the form the Rust kernels take:
    # the field flattened to one row per point, its own shape, the indices of the points to
    # evaluate at, and whether a single integer index was given
    _check_stencils(stencils)
    values = np.asarray(values, dtype=np.float64)
    n_points = stencils.points.shape[0]
    if values.ndim == 0 or values.shape[0] != n_points:
        raise ValueError(
            f"values must have one entry per point along the first axis ({n_points} points), "
            f"got shape {values.shape}"
        )
    field_shape = values.shape[1:]                       # one point's value: () for a scalar field
    flat = values.reshape(n_points, -1)                   # one row of components per point
    # NumPy indexing turns `at` into point indices, with the usual meaning of negative indices,
    # slices and boolean masks, and raises IndexError for points that don't exist
    targets = np.arange(n_points)[slice(None) if at is None else at]
    single = np.ndim(targets) == 0
    return flat, field_shape, np.atleast_1d(targets).astype(np.int64), single


def _check_stencils(stencils):
    # The operators need the arrays of a StencilSet
    if not isinstance(stencils, StencilSet):
        raise TypeError(f"expected a StencilSet, got {type(stencils).__name__}")