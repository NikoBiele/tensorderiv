import numpy as np
import pytest

from tensorderiv import StencilSet, curl, divergence, gradient, laplacian
from test_weights import lcg_points  # the same deterministic points as the Julia package's tests

TOL = 1e-9  # exactness tolerance, as in the Julia package's tests


def interior(points, margin):
    # Indices of the points at least `margin` away from the boundary of the unit cube
    return np.flatnonzero(np.all((points > margin) & (points < 1 - margin), axis=1))


# ---- exactness on polynomial fields (3D), mirroring the Julia package's tests ----

POINTS = lcg_points(600, 3)
X, Y, Z = POINTS.T  # coordinate columns, one value per point


def linear():
    # f = 1 + x − 2y + 3z, with its gradient and Laplacian at every point
    f = 1 + X - 2 * Y + 3 * Z
    grad = np.tile([1.0, -2.0, 3.0], (len(X), 1))
    return f, grad, np.zeros(len(X))


def quadratic():
    # the linear field plus x² + 2xy − yz + 3z²
    f0, g0, _ = linear()
    f = f0 + X**2 + 2 * X * Y - Y * Z + 3 * Z**2
    grad = g0 + np.column_stack([2 * X + 2 * Y, 2 * X - Z, -Y + 6 * Z])
    return f, grad, np.full(len(X), 8.0)


def cubic():
    # the quadratic field plus x³ − xyz + 2y²z
    f0, g0, l0 = quadratic()
    f = f0 + X**3 - X * Y * Z + 2 * Y**2 * Z
    grad = g0 + np.column_stack([3 * X**2 - Y * Z, -X * Z + 4 * Y * Z, -X * Y + 2 * Y**2])
    return f, grad, l0 + 6 * X + 4 * Z


@pytest.fixture(scope="module")
def stencils_by_order():
    # first- and second-order stencils on the same points, built once for all tests below
    return {order: StencilSet(POINTS, order=order) for order in (1, 2)}


@pytest.mark.parametrize("order, exact_gradient, exact_laplacian, inexact_gradient", [
    (1, [linear], [linear, quadratic], quadratic),     # order 1: gradient exact for linear fields
    (2, [linear, quadratic], [quadratic, cubic], cubic),  # order 2: for quadratic fields
])
def test_scalar_exactness(stencils_by_order, order, exact_gradient, exact_laplacian, inexact_gradient):
    stencils = stencils_by_order[order]
    for field in exact_gradient:
        f, grad, _ = field()
        assert np.max(np.abs(gradient(stencils, f) - grad)) < TOL
    for field in exact_laplacian:
        f, _, lap = field()
        assert np.max(np.abs(laplacian(stencils, f) - lap)) < TOL
    # one degree higher the gradient is not exact, which confirms the order
    f, grad, _ = inexact_gradient()
    assert np.max(np.abs(gradient(stencils, f) - grad)) > 1e-3


def test_linear_vector_field(stencils_by_order):
    # v = A x + b: every operator is exact at order 1
    stencils = stencils_by_order[1]
    A = np.array([[1.0, 2.0, 0.0], [-1.0, 0.0, 3.0], [2.0, 1.0, -1.0]])
    v = POINTS @ A.T + np.array([0.5, -1.0, 2.0])                # one row v(x) = A x + b per point
    assert np.max(np.abs(gradient(stencils, v) - A.T)) < TOL     # [a, j] = ∂v_j/∂x_a = A[j, a]
    assert np.max(np.abs(divergence(stencils, v) - np.trace(A))) < TOL
    assert np.max(np.abs(curl(stencils, v) - [-2.0, -2.0, -3.0])) < TOL
    assert np.max(np.abs(laplacian(stencils, v))) < TOL


def test_quadratic_vector_field(stencils_by_order):
    # every operator is exact at order 2
    stencils = stencils_by_order[2]
    v = np.column_stack([X * Y + Z, Y**2 - X * Z, X**2 + Y * Z])
    zero, one = np.zeros(len(X)), np.ones(len(X))
    grad = np.stack([                                             # [p, a, j] = ∂v_j/∂x_a
        np.column_stack([Y, -Z, 2 * X]),
        np.column_stack([X, 2 * Y, Z]),
        np.column_stack([one, -X, Y]),
    ], axis=1)
    assert np.max(np.abs(gradient(stencils, v) - grad)) < TOL
    assert np.max(np.abs(divergence(stencils, v) - 4 * Y)) < TOL
    assert np.max(np.abs(curl(stencils, v) - np.column_stack([Z + X, 1 - 2 * X, -Z - X]))) < TOL
    assert np.max(np.abs(laplacian(stencils, v) - np.column_stack([zero, 2 * one, 2 * one]))) < TOL


def test_linear_matrix_field(stencils_by_order):
    # M_ij = Σ_a B[a, i, j] x_a + 1: gradient, divergence and curl are exact at order 1
    stencils = stencils_by_order[1]
    B = np.arange(1.0, 28.0).reshape(3, 3, 3) - 14.0
    M = np.einsum("pa,aij->pij", POINTS, B) + 1.0                 # (points, 3, 3)
    assert np.max(np.abs(gradient(stencils, M) - B)) < TOL       # [p, a, i, j] = ∂M_ij/∂x_a
    assert np.max(np.abs(divergence(stencils, M) - np.einsum("aaj->j", B))) < TOL
    curl_expected = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])  # [i, r]
    assert np.max(np.abs(curl(stencils, M) - curl_expected)) < TOL


# ---- convergence order (2D) ----

def median_errors(n_points, order):
    # Median interior gradient and Laplacian errors of a smooth 2D field
    points = lcg_points(n_points, 2)
    x, y = points.T
    f = np.sin(3 * x) * np.exp(y) + np.cos(2 * y)
    grad = np.column_stack([3 * np.cos(3 * x) * np.exp(y), np.sin(3 * x) * np.exp(y) - 2 * np.sin(2 * y)])
    lap = -8 * np.sin(3 * x) * np.exp(y) - 4 * np.cos(2 * y)
    stencils = StencilSet(points, order=order)
    inner = interior(points, 0.2)
    grad_error = np.linalg.norm(gradient(stencils, f) - grad, axis=1)[inner]
    lap_error = np.abs(laplacian(stencils, f) - lap)[inner]
    return np.array([np.median(grad_error), np.median(lap_error)])


def test_convergence_order():
    # Four times as many points halves the stencil radius in 2D
    coarse1, fine1 = median_errors(2000, 1), median_errors(8000, 1)
    coarse2, fine2 = median_errors(2000, 2), median_errors(8000, 2)
    assert np.all(coarse1 / fine1 > 1.6)   # first order: error ratio about 2
    assert np.all(coarse2 / fine2 > 3.2)   # second order: error ratio about 4
    assert np.all(fine2 < fine1 / 5)       # order 2 is much more accurate at the same resolution


# ---- shapes, evaluation points and threading ----

@pytest.fixture(scope="module")
def small():
    # a small 3D cloud with a scalar, a vector and a matrix field on it
    points = lcg_points(200, 3)
    f = points.sum(axis=1)                                      # (200,)
    v = 2 * points                                              # (200, 3)
    M = np.stack([points, -points], axis=2)                     # (200, 3, 2)
    return StencilSet(points), f, v, M


def test_shapes(small):
    stencils, f, v, M = small
    assert gradient(stencils, f).shape == (200, 3)
    assert laplacian(stencils, f).shape == (200,)
    assert gradient(stencils, v).shape == (200, 3, 3)
    assert divergence(stencils, v).shape == (200,)
    assert curl(stencils, v).shape == (200, 3)
    assert laplacian(stencils, v).shape == (200, 3)
    assert gradient(stencils, M).shape == (200, 3, 3, 2)
    assert divergence(stencils, M).shape == (200, 2)
    assert curl(stencils, M).shape == (200, 3, 2)
    assert laplacian(stencils, M).shape == (200, 3, 2)


def test_at(small):
    # Evaluating at selected points gives the same values as the whole cloud, indexed as in NumPy
    stencils, f, v, M = small
    for operator, field in [(gradient, f), (gradient, M), (divergence, v), (divergence, M),
                            (curl, v), (curl, M), (laplacian, f), (laplacian, M)]:
        everywhere = operator(stencils, field)
        for at in [7, -1, [3, 5, 199], slice(10, 20), np.arange(200) % 3 == 0]:
            np.testing.assert_array_equal(operator(stencils, field, at=at), everywhere[at])
    assert np.ndim(divergence(stencils, v, at=7)) == 0         # a single number
    assert np.ndim(laplacian(stencils, f, at=7)) == 0


def test_threaded_matches_serial(small):
    stencils, f, v, M = small
    for operator, field in [(gradient, M), (divergence, M), (curl, M), (laplacian, M)]:
        np.testing.assert_array_equal(operator(stencils, field), operator(stencils, field, threaded=False))


def test_errors(small):
    stencils, f, v, M = small
    stencils2d = StencilSet(lcg_points(100, 2), order=1)
    with pytest.raises(TypeError):
        gradient(stencils.points, f)                            # not a StencilSet
    with pytest.raises(ValueError, match="one entry per point"):
        gradient(stencils, f[:10])                              # wrong number of points
    with pytest.raises(ValueError, match="first axis"):
        divergence(stencils, f)                                 # divergence of a scalar field
    with pytest.raises(ValueError, match="first axis"):
        divergence(stencils, np.zeros((200, 2)))                # first axis is not 3
    with pytest.raises(ValueError, match="3 dimensions"):
        curl(stencils2d, np.zeros((100, 2)))                    # curl in 2D
    with pytest.raises(IndexError):
        gradient(stencils, f, at=200)                           # no such point


# ---- agreement with the Julia package ----

@pytest.mark.parametrize("operator, field_name", [
    ("gradient", "f"), ("gradient", "v"), ("gradient", "M"),
    ("divergence", "v"), ("divergence", "M"),
    ("curl", "v"), ("curl", "M"),
    ("laplacian", "f"), ("laplacian", "v"), ("laplacian", "M"),
])
def test_matches_julia(operator, field_name):
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import DiscreteTensorDerivatives")
    # Julia stores one point per column and puts the point index last, both for the field and
    # for the result; the function moves it there and back
    julia_operator = jl.seval("""
        (points, values, name) -> begin
            s = DiscreteTensorDerivatives.StencilSet(permutedims(Matrix{Float64}(points)); threaded = false)
            op = getfield(DiscreteTensorDerivatives, Symbol(name))
            op(s, Array{Float64}(values))
        end
    """)
    points = lcg_points(1000, 3)
    x, y, z = points.T
    fields = {
        "f": np.sin(x) * np.exp(y) + z**3 * x,                                 # scalar
        "v": np.column_stack([x * y, np.sin(z), x**2 * z]),                    # vector
        "M": np.stack([np.column_stack([x * y, np.sin(z), x**2 * z]),
                       np.column_stack([np.cos(x), y * z, x + y**2])], axis=2),  # matrix (3 × 2)
    }
    values = fields[field_name]
    expected = np.moveaxis(np.array(julia_operator(points, np.moveaxis(values, 0, -1), operator)), -1, 0)
    result = {"gradient": gradient, "divergence": divergence, "curl": curl, "laplacian": laplacian}[operator](
        StencilSet(points), values)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10 * np.max(np.abs(expected)))