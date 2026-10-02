// The Rust core of tensorderiv.

use numpy::PyReadonlyArray1;
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

/// The module Python imports as tensorderiv._core.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_of_squares, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    Ok(())
}