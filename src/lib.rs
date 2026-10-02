// The Rust core of tensorderiv.

// The stencil weights (src/weights.rs), their computation for a whole point cloud
// (src/stencils.rs) and the derivative operators (src/operators.rs) become modules of this crate
mod operators;
mod stencils;
mod weights;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

/// Python-visible: the weights of every point's stencil. `points` has one row per point
/// (shape (points, dims)), and `neighbours` one row of k neighbour indices per point
/// (shape (points, k)), excluding the point itself. Returns the weights with the shape of
/// `neighbours`: row i holds the weights of point i's neighbours, in the same order.
#[pyfunction]
fn stencil_set_weights<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    neighbours: PyReadonlyArray2<'py, i64>,
    order: usize,
    threaded: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let points = points.as_array(); // the coordinates as a Rust array view, without copying
    let dims = points.ncols(); // number of dimensions
    let points = points.as_standard_layout(); // row-major: unchanged if it already is, otherwise copied
    let neighbours = neighbours.as_array(); // the neighbour indices as a Rust array view
    let (npoints, k) = neighbours.dim(); // number of points and neighbours per point
    if npoints != points.nrows() {
        return Err(PyValueError::new_err(format!(
            "{npoints} rows of neighbours for {} points",
            points.nrows()
        )));
    }
    // NumPy indices are signed; convert them, row after row, to Rust's unsigned indices,
    // rejecting negative ones (`collect` stops at the first error)
    let indices: Vec<usize> = neighbours
        .iter()
        .map(|&j| usize::try_from(j).map_err(|_| format!("negative neighbour index {j}")))
        .collect::<Result<_, _>>()
        .map_err(PyValueError::new_err)?;
    let coordinates = points.as_slice().expect("a standard-layout array is contiguous");
    let all = stencils::all_weights(coordinates, dims, &indices, k, order, threaded)
        .map_err(PyValueError::new_err)?;
    // reshape the flat result into one row per point; the length always matches
    let all = Array2::from_shape_vec((npoints, k), all).expect("one row of k weights per point");
    Ok(all.into_pyarray(py))
}

/// The signature shared by the operator kernels in src/operators.rs.
type Kernel = fn(&operators::Stencils, &[f64], usize, &[usize], bool) -> Result<Vec<f64>, String>;

/// Run an operator kernel on the arrays of a Python StencilSet: `points` (points, dims),
/// `neighbours` and `weights` (points, k), the field `values` with one flattened row per point
/// (points, components), and the indices of the `targets` to evaluate at. Returns the result
/// as one flat array, which the Python side reshapes.
fn apply_operator<'py>(
    py: Python<'py>,
    kernel: Kernel,
    points: PyReadonlyArray2<'py, f64>,
    neighbours: PyReadonlyArray2<'py, i64>,
    weights: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray1<'py, i64>,
    threaded: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Rust array views of the NumPy arrays, made row-major where they aren't already (a copy
    // only in that case); each view needs its own `let`, so that it lives as long as its copy
    let points = points.as_array();
    let dims = points.ncols(); // number of dimensions
    let points = points.as_standard_layout();
    let neighbours = neighbours.as_array();
    let k = neighbours.ncols(); // neighbours per point
    let neighbours = neighbours.as_standard_layout();
    let weights = weights.as_array();
    let weights = weights.as_standard_layout();
    let values = values.as_array();
    let components = values.ncols(); // field values per point
    let values = values.as_standard_layout();
    // NumPy indices are signed; convert the targets to Rust's unsigned indices, rejecting
    // negative ones (`collect` stops at the first error)
    let targets: Vec<usize> = targets
        .as_array()
        .iter()
        .map(|&i| usize::try_from(i).map_err(|_| format!("negative point index {i}")))
        .collect::<Result<_, _>>()
        .map_err(PyValueError::new_err)?;
    let contiguous = "a standard-layout array is contiguous";
    let stencils = operators::Stencils {
        points: points.as_slice().expect(contiguous),
        dims,
        neighbours: neighbours.as_slice().expect(contiguous),
        weights: weights.as_slice().expect(contiguous),
        k,
    };
    let result = kernel(&stencils, values.as_slice().expect(contiguous), components, &targets, threaded)
        .map_err(PyValueError::new_err)?;
    Ok(result.into_pyarray(py))
}

/// Python-visible: the gradient Σ_k a_k Δx_k ⊗ ΔY_k at each target point, as one flat array
/// of shape (targets × dims × components), derivative direction before field component.
#[pyfunction]
fn gradient_kernel<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    neighbours: PyReadonlyArray2<'py, i64>,
    weights: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray1<'py, i64>,
    threaded: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    apply_operator(py, operators::gradient, points, neighbours, weights, values, targets, threaded)
}

/// Python-visible: the Laplacian 2 Σ_k a_k ΔY_k at each target point, as one flat array of
/// shape (targets × components).
#[pyfunction]
fn laplacian_kernel<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    neighbours: PyReadonlyArray2<'py, i64>,
    weights: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray1<'py, i64>,
    threaded: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    apply_operator(py, operators::laplacian, points, neighbours, weights, values, targets, threaded)
}

/// The module Python imports as tensorderiv._core.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_of_squares, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(stencil_weights, m)?)?;
    m.add_function(wrap_pyfunction!(stencil_set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_kernel, m)?)?;
    Ok(())
}