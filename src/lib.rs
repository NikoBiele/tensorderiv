// The Rust core of tensorderiv.

// The stencil weights (src/weights.rs) become a module of this crate
mod weights;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Python-visible toolchain check: the sum of squares of a NumPy array, computed in parallel
/// by rayon. It only confirms that PyO3, rust-numpy and rayon work together, and is replaced
/// by the real functions in later steps.
#[pyfunction]
fn sum_of_squares(x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    // borrow the array's data as a plain Rust slice; this fails for non-contiguous arrays
    let x = x.as_slice()?;
    // `par_iter` splits the slice across rayon's threads; each squares its share, then the
    // partial sums are added together
    Ok(x.par_iter().map(|&v| v * v).sum())
}

/// Python-visible: the number of threads rayon uses, by default one per logical core.
#[pyfunction]
fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Python-visible: the smallest number of neighbours for which the moment conditions of
/// `order` can hold exactly in `dims` dimensions.
#[pyfunction]
fn minimum_neighbours(dims: usize, order: usize) -> PyResult<usize> {
    if order != 1 && order != 2 {
        return Err(PyValueError::new_err(format!("order must be 1 or 2, got {order}")));
    }
    Ok(weights::minimum_neighbours(dims, order))
}

/// Python-visible: the weights of one stencil. `offsets` holds the offsets x_k − x_0 of the
/// neighbours, one row per neighbour (shape (neighbours, dims)).
#[pyfunction]
fn stencil_weights<'py>(
    py: Python<'py>,
    offsets: PyReadonlyArray2<'py, f64>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = offsets.as_array(); // the NumPy array as a Rust array view, without copying
    let dims = view.ncols(); // number of dimensions: the length of each row
    // the offsets in row-major order: the view itself if it already is, otherwise a copy, so that
    // column-major (Fortran-ordered) or strided arrays are read correctly too
    let standard = view.as_standard_layout();
    let dx = standard.as_slice().expect("a standard-layout array is contiguous"); // one flat slice, row after row
    // `map_err` turns the Rust error message into a Python ValueError
    let a = weights::weights(dx, dims, order).map_err(PyValueError::new_err)?;
    Ok(a.into_pyarray(py))
}

/// The module Python imports as tensorderiv._core.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_of_squares, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(stencil_weights, m)?)?;
    Ok(())
}