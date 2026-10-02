"""tensorderiv: gradient, divergence, curl and Laplacian on scattered points, a Python
interface to a Rust core."""

from ._core import sum_of_squares, thread_count

__all__ = ["sum_of_squares", "thread_count"]