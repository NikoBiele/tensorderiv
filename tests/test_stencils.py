import numpy as np
import pytest

from tensorderiv import StencilSet, minimum_neighbours, thread_count
from test_weights import lcg_points  # the same deterministic points as the Julia package's tests


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_defaults_and_shapes(dims, order):
    # The default k is twice the minimum, and every attribute has one row per point
    points = lcg_points(500, dims)
    stencils = StencilSet(points, order=order)
    k = 2 * minimum_neighbours(dims, order)
    assert stencils.order == order
    assert stencils.points.shape == (500, dims)
    assert stencils.neighbours.shape == (500, k)
    assert stencils.weights.shape == (500, k)
    assert repr(stencils) == f"StencilSet(500 points in {dims}D, order={order}, k={k})"


def test_second_order_is_default():
    assert StencilSet(lcg_points(200, 3)).order == 2


def test_neighbours():
    # No point is its own neighbour, and neighbours are sorted nearest first
    points = lcg_points(1000, 3)
    stencils = StencilSet(points)
    assert not np.any(stencils.neighbours == np.arange(1000)[:, None])
    distances = np.linalg.norm(points[stencils.neighbours] - points[:, None, :], axis=2)
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_duplicate_points():
    # Points at identical coordinates still get k neighbours, none of them themselves
    points = lcg_points(300, 2)
    points = np.vstack([points, points[:10]])                  # the first ten points twice
    stencils = StencilSet(points, order=1)
    assert stencils.neighbours.shape == (310, 10)
    assert not np.any(stencils.neighbours == np.arange(310)[:, None])
    assert np.all(np.isfinite(stencils.weights))


def grid(n, dims):
    # A regular grid of n points per axis in the unit cube, one row per point
    axes = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * dims, indexing="ij")
    return np.stack(axes, axis=-1).reshape(-1, dims)


@pytest.mark.parametrize("dims, order, k", [
    (2, 1, None), (2, 2, None),        # 2D grids work with the defaults
    (3, 2, None),                      # 3D grids work at order 2 with the default k
    (3, 1, 24),                        # and at order 1 with more neighbours than the default
])
def test_regular_grids(dims, order, k):
    # Stencils on a regular grid are exact for linear fields everywhere, faces included,
    # and the Laplacian is exact for quadratic ones
    from tensorderiv import gradient, laplacian
    points = grid(15, dims)
    stencils = StencilSet(points, order=order, k=k)
    slope = np.arange(1.0, dims + 1)                           # gradient of the linear field
    assert np.max(np.abs(gradient(stencils, points @ slope) - slope)) < 1e-9
    assert np.max(np.abs(laplacian(stencils, (points**2).sum(axis=1)) - 2 * dims)) < 1e-9


def sphere_surface(n_points):
    # Points spread evenly over the surface of the unit sphere: a curved surface in 3D
    rng = np.random.default_rng(0)
    theta = rng.random(n_points) * 2 * np.pi                   # longitudes
    phi = np.arccos(rng.uniform(-1.0, 1.0, n_points))          # latitudes
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


@pytest.mark.parametrize("points, order", [
    (np.column_stack([lcg_points(500, 2), np.zeros(500)]), 2),                         # a plane in 3D
    (lcg_points(500, 2) @ np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.7]]), 2),            # a tilted plane in 3D
    (sphere_surface(1000), 2),                                                        # a curved surface in 3D
    (np.column_stack([np.cos(np.linspace(0, 6.2, 500)), np.sin(np.linspace(0, 6.2, 500))]), 2),  # a circle in 2D
    (np.column_stack([np.linspace(0, 1, 500), np.zeros(500)]), 2),                     # a line in 2D
    (grid(15, 3), 1),                                                                 # a 3D grid at order 1, default k
], ids=["plane", "tilted plane", "sphere", "circle", "line", "grid at order 1"])
def test_degenerate_points_rejected(points, order):
    # Points that do not fill their space cannot satisfy the moment conditions; the stencils
    # would give meaningless derivatives, so building them raises an error instead
    with pytest.raises(ValueError, match="moment conditions"):
        StencilSet(points, order=order)


def test_threaded_matches_serial():
    # Splitting the work over threads must not change a single bit
    points = lcg_points(3000, 3)
    threaded = StencilSet(points)
    serial = StencilSet(points, threaded=False)
    np.testing.assert_array_equal(threaded.neighbours, serial.neighbours)
    np.testing.assert_array_equal(threaded.weights, serial.weights)


def test_thread_count():
    # rayon always has at least one thread
    assert thread_count() >= 1


def test_input_conversion():
    # Lists and integer coordinates are accepted and converted to float64
    points = lcg_points(200, 2)
    from_list = StencilSet(points.tolist())
    np.testing.assert_array_equal(from_list.weights, StencilSet(points).weights)
    integer = StencilSet(np.round(1000 * points).astype(int))
    assert integer.points.dtype == np.float64


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_matches_julia(dims, order):
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    jl.seval("import DiscreteTensorDerivatives")
    # The Julia StencilSet stores one point per column and counts from 1; return its neighbours
    # and weights as matrices with one row per point, to compare with the Python layout
    julia_stencils = jl.seval("""
        (points, order) -> begin
            s = DiscreteTensorDerivatives.StencilSet(permutedims(Matrix{Float64}(points));
                                                     order = order, threaded = false)
            (permutedims(stack(s.neighbours)), permutedims(stack(s.weights)))
        end
    """)
    points = lcg_points(1000, dims)
    julia_neighbours, julia_weights = julia_stencils(points, order)
    stencils = StencilSet(points, order=order)
    np.testing.assert_array_equal(stencils.neighbours + 1, np.array(julia_neighbours))
    # each point's weights agree to rounding, relative to the size of that point's weights
    expected = np.array(julia_weights)
    differences = np.linalg.norm(stencils.weights - expected, axis=1)
    assert np.all(differences <= 1e-12 * np.linalg.norm(expected, axis=1))


def test_errors():
    points = lcg_points(100, 3)
    with pytest.raises(ValueError, match="2D array"):
        StencilSet(points[:, 0])                               # one coordinate per point, not a matrix
    with pytest.raises(ValueError, match="order"):
        StencilSet(points, order=3)                            # unsupported order
    with pytest.raises(ValueError, match="minimum"):
        StencilSet(points, k=5)                                # below the minimum of 19
    with pytest.raises(ValueError, match="smaller than the number of points"):
        StencilSet(points[:30])                                # k = 38 needs more points
    with pytest.raises(ValueError, match="NaN"):
        StencilSet(np.vstack([points, [np.nan, 0.0, 0.0]]))    # a NaN coordinate
    with pytest.raises(TypeError):
        StencilSet(points, k=20.5)                             # k must be an integer