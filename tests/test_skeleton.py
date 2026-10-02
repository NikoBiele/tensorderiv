import numpy as np
import pytest

from tensorderiv import sum_of_squares, thread_count

# A fixed set of values to sum: a million evenly spaced numbers, enough to be split across threads
X = np.linspace(-2.0, 2.0, 1_000_001)


def test_matches_numpy():
    # NumPy's own sum of squares as the reference; the parallel sum adds its partial sums in a
    # thread-dependent order, so the comparison allows rounding differences
    expected = np.sum(X * X)
    np.testing.assert_allclose(sum_of_squares(X), expected, rtol=1e-12)


def test_matches_julia():
    # Skipped automatically when juliacall isn't installed
    pytest.importorskip("juliacall")
    from juliacall import Main as jl
    # Julia's sum(abs2, x) as the reference: the pattern later tests against the Julia package follow
    expected = jl.sum(jl.abs2, X)
    np.testing.assert_allclose(sum_of_squares(X), expected, rtol=1e-12)


def test_thread_count():
    # rayon always has at least one thread
    assert thread_count() >= 1