"""tensorderiv: gradient, divergence, curl and Laplacian on scattered points, a Python
interface to a Rust core."""

import operator

import numpy as np
from scipy.spatial import cKDTree

from ._core import minimum_neighbours, stencil_set_weights, sum_of_squares, thread_count

__all__ = ["StencilSet", "minimum_neighbours", "sum_of_squares", "thread_count"]


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