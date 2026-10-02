import numpy as np
import pytest
from scipy.spatial import cKDTree

from tensorderiv._core import minimum_neighbours, stencil_weights


def lcg_points(n_points, dims, seed=0x12345678):
    # Deterministic pseudo-random points in the unit cube, one row per point: the same 32-bit
    # generator as the Julia package's tests, so both test suites use identical point sets
    state = seed
    points = np.empty((n_points, dims))
    for p in range(n_points):
        for d in range(dims):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF  # Numerical Recipes LCG step
            points[p, d] = state / 2.0**32                       # map to [0, 1)
    return points


def stencils(points, k):
    # Offsets of each point's k nearest neighbours, one (k, dims) array per point
    _, nearest = cKDTree(points).query(points, k + 1)            # k + 1: each point finds itself first
    return [points[row[1:]] - points[row[0]] for row in nearest]


CASES = [(dims, order) for dims in (2, 3) for order in (1, 2)]


@pytest.mark.parametrize("dims, order", CASES)
@pytest.mark.parametrize("scale", [1.0, 1e-4])
def test_moment_conditions(dims, order, scale):
    # The weights must satisfy every moment condition, also for tiny stencils
    k = 2 * minimum_neighbours(dims, order)
    for dx in stencils(scale * lcg_points(300, dims), k):
        a = stencil_weights(dx, order)
        h = np.max(np.linalg.norm(dx, axis=1))                   # stencil radius
        assert np.linalg.norm(a @ dx) * h < 1e-10                 # Σ a Δx = 0, made dimensionless
        assert np.linalg.norm(dx.T @ (a[:, None] * dx) - np.eye(dims)) < 1e-10  # Σ a Δx Δxᵀ = I
        if order == 2:
            third = np.einsum("q,qi,qj,ql->ijl", a, dx, dx, dx)  # Σ a Δx ⊗ Δx ⊗ Δx
            assert np.linalg.norm(third) / h < 1e-10


@pytest.mark.parametrize("dims, order", CASES)
def test_matches_julia(dims, order):
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import DiscreteTensorDerivatives")
    # The Julia package stores one neighbour per column, so the offsets are transposed
    julia_weights = jl.seval(
        "(dX, order) -> DiscreteTensorDerivatives.coefficients(Matrix{Float64}(dX), order)"
    )
    k = 2 * minimum_neighbours(dims, order)
    for dx in stencils(lcg_points(200, dims), k):
        expected = np.array(julia_weights(dx.T, order))
        a = stencil_weights(dx, order)
        assert np.linalg.norm(a - expected) <= 1e-12 * np.linalg.norm(expected)


def test_memory_layouts():
    # Fortran-ordered and strided offsets must give the same weights as C-ordered ones
    dx = stencils(lcg_points(100, 3), 38)[0]
    a = stencil_weights(dx, 2)
    np.testing.assert_array_equal(stencil_weights(np.asfortranarray(dx), 2), a)
    wide = np.repeat(dx, 2, axis=1)                              # every column twice
    np.testing.assert_array_equal(stencil_weights(wide[:, ::2], 2), a)  # a strided view of dx


def test_minimum_neighbours():
    assert minimum_neighbours(2, 1) == 5
    assert minimum_neighbours(3, 1) == 9
    assert minimum_neighbours(2, 2) == 9
    assert minimum_neighbours(3, 2) == 19
    with pytest.raises(ValueError):
        minimum_neighbours(3, 3)                                 # unsupported order


def test_errors():
    with pytest.raises(ValueError, match="order"):
        stencil_weights(np.ones((10, 2)), 3)                     # unsupported order
    with pytest.raises(ValueError, match="coincide"):
        stencil_weights(np.zeros((10, 2)), 1)                    # all neighbours at the centre
    with pytest.raises(ValueError, match="coincide"):
        stencil_weights(np.full((10, 2), np.nan), 1)             # NaN offsets