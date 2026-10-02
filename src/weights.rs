//! Stencil weights: the minimum-norm solution of the moment conditions, as
//! DiscreteTensorDerivatives.jl computes them (coefficients.jl).
//!
//! Offsets are stored with one row per neighbour: the offset of neighbour `q` occupies
//! `dx[q * dims .. (q + 1) * dims]`, matching a NumPy array of shape (neighbours, dims).

/// The number of moment conditions in `dims` dimensions, which is also the smallest number of
/// neighbours for which they can hold exactly: dims(dims+3)/2 for order 1 (first and second
/// moments), plus dims(dims+1)(dims+2)/6 for order 2 (third moments).
/// `order` must already be checked to be 1 or 2.
pub fn minimum_neighbours(dims: usize, order: usize) -> usize {
    let first_and_second = dims * (dims + 3) / 2; // first moment (dims) and second moment (dims(dims+1)/2)
    let third = dims * (dims + 1) * (dims + 2) / 6; // distinct entries of the symmetric third moment
    if order == 1 {
        first_and_second
    } else {
        first_and_second + third
    }
}

/// Scratch arrays for the weights of one stencil, reusable across all stencils with the same
/// dimension, number of neighbours and order, so that computing weights allocates nothing.
pub struct Workspace {
    /// The moment conditions, one row of length `neighbours` per condition, stored one row
    /// after another; Gram-Schmidt turns these rows into an orthonormal basis in place
    rows: Vec<f64>,
    /// Right-hand side of the conditions, one entry per condition
    b: Vec<f64>,
    /// Solution coefficients in the orthonormal basis, one entry per condition
    y: Vec<f64>,
    /// Number of neighbours per stencil
    neighbours: usize,
    /// Number of moment conditions
    conditions: usize,
}

impl Workspace {
    /// A workspace for stencils of `neighbours` neighbours in `dims` dimensions at `order`.
    pub fn new(dims: usize, neighbours: usize, order: usize) -> Workspace {
        let conditions = minimum_neighbours(dims, order);
        Workspace {
            rows: vec![0.0; conditions * neighbours],
            b: vec![0.0; conditions],
            y: vec![0.0; conditions],
            neighbours,
            conditions,
        }
    }
}

/// Euclidean length of a slice.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two slices of equal length.
fn dot(u: &[f64], v: &[f64]) -> f64 {
    // `zip` walks both slices side by side
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

/// Compute the weights of one stencil into `a` (one weight per neighbour), using `workspace` for
/// scratch space. `dx` holds the offsets x_k − x_0, one row of `dims` values per neighbour.
///
/// The weights satisfy Σ a_k Δx_k = 0 and Σ a_k Δx_k Δx_kᵀ = I, and for order 2 also
/// Σ a_k Δx_k ⊗ Δx_k ⊗ Δx_k = 0; among all such weights they have minimum norm.
pub fn weights_into(
    a: &mut [f64],
    workspace: &mut Workspace,
    dx: &[f64],
    dims: usize,
    order: usize,
) -> Result<(), String> {
    if order != 1 && order != 2 {
        return Err(format!("order must be 1 or 2, got {order}"));
    }
    let k = workspace.neighbours;
    if dims == 0
        || a.len() != k
        || dx.len() != k * dims
        || workspace.conditions != minimum_neighbours(dims, order)
    {
        return Err(format!(
            "weights, offsets or workspace do not match {k} neighbours in {dims} dimensions at order {order}"
        ));
    }
    // stencil radius: the largest offset length (`chunks_exact` splits dx into one row per neighbour)
    let h = dx.chunks_exact(dims).map(norm).fold(0.0, f64::max);
    // `!(h > 0.0)` is also true when h is NaN, so NaN coordinates are rejected here too
    if !(h > 0.0) {
        return Err("all neighbours coincide with the central point".to_string());
    }
    moment_system(workspace, dx, dims, h, order);
    minimum_norm_solve(a, workspace);
    // undo the scaling, since Σ a_k Δx_k Δx_kᵀ = I makes a ∝ 1/h²
    let h2 = h * h;
    for w in a.iter_mut() {
        *w /= h2;
    }
    Ok(())
}

/// The weights of one stencil as a new vector, with a one-off workspace.
/// `dx` holds the offsets, one row of `dims` values per neighbour.
pub fn weights(dx: &[f64], dims: usize, order: usize) -> Result<Vec<f64>, String> {
    if order != 1 && order != 2 {
        return Err(format!("order must be 1 or 2, got {order}"));
    }
    if dims == 0 || dx.len() % dims != 0 {
        return Err(format!("{} offset values do not form rows of {dims}", dx.len()));
    }
    let k = dx.len() / dims; // number of neighbours
    let mut workspace = Workspace::new(dims, k, order);
    let mut a = vec![0.0; k];
    weights_into(&mut a, &mut workspace, dx, dims, order)?; // `?` returns early on an error
    Ok(a)
}

/// Fill the workspace with the moment conditions for offsets scaled by `h`, so that all
/// conditions are of similar size: first moments zero (dims rows), second moment the identity
/// (one row per entry i ≤ j) and, for order 2, third moments zero (one row per entry i ≤ j ≤ l).
fn moment_system(workspace: &mut Workspace, dx: &[f64], dims: usize, h: f64, order: usize) {
    let k = workspace.neighbours;
    // scaled offset component i of neighbour q
    let s = |q: usize, i: usize| dx[q * dims + i] / h;
    workspace.b.fill(0.0); // most conditions have a zero right-hand side
    let mut r = 0; // index of the current condition
    for i in 0..dims {
        // Σ a_q S_qi = 0
        let row = &mut workspace.rows[r * k..(r + 1) * k];
        for q in 0..k {
            row[q] = s(q, i);
        }
        r += 1;
    }
    for i in 0..dims {
        for j in i..dims {
            // Σ a_q S_qi S_qj = δ_ij
            let row = &mut workspace.rows[r * k..(r + 1) * k];
            for q in 0..k {
                row[q] = s(q, i) * s(q, j);
            }
            workspace.b[r] = if i == j { 1.0 } else { 0.0 };
            r += 1;
        }
    }
    if order == 2 {
        for i in 0..dims {
            for j in i..dims {
                for l in j..dims {
                    // Σ a_q S_qi S_qj S_ql = 0
                    let row = &mut workspace.rows[r * k..(r + 1) * k];
                    for q in 0..k {
                        row[q] = s(q, i) * s(q, j) * s(q, l);
                    }
                    r += 1;
                }
            }
        }
    }
}

/// Minimum-norm solution of the conditions in the workspace, written into `a`. The condition
/// rows are orthonormalized in place by Gram-Schmidt with two passes (enough for full
/// orthogonality), while the right-hand side is forward-substituted alongside; the solution is
/// then a combination of the orthonormal rows. Conditions that are linearly dependent on
/// earlier ones leave a negligible remainder and are dropped.
fn minimum_norm_solve(a: &mut [f64], workspace: &mut Workspace) {
    let k = workspace.neighbours;
    let m = workspace.conditions;
    let tol = 100.0 * k as f64 * f64::EPSILON; // relative size below which a condition is dependent
    for r in 0..m {
        // Split the rows into the finished ones (before r) and the rest, so that Rust allows
        // reading the finished rows while modifying row r
        let (done, rest) = workspace.rows.split_at_mut(r * k);
        let row = &mut rest[..k];
        let mut t = workspace.b[r]; // right-hand side, reduced by earlier conditions
        let original = norm(row); // size of the condition before orthogonalization
        for _pass in 0..2 {
            for s in 0..r {
                let basis = &done[s * k..(s + 1) * k]; // finished orthonormal row s
                let c = dot(basis, row); // component of row r along it
                for q in 0..k {
                    row[q] -= c * basis[q]; // remove that component
                }
                t -= c * workspace.y[s]; // and its share of the right-hand side
            }
        }
        let remaining = norm(row); // what is left of the condition
        if remaining <= tol * original {
            row.fill(0.0); // dependent condition: drop it
            workspace.y[r] = 0.0;
        } else {
            for v in row.iter_mut() {
                *v /= remaining; // normalize the new basis row
            }
            workspace.y[r] = t / remaining;
        }
    }
    a.fill(0.0); // a = Σ_r y_r q_r over the basis rows
    for r in 0..m {
        let basis = &workspace.rows[r * k..(r + 1) * k];
        let yr = workspace.y[r];
        for q in 0..k {
            a[q] += basis[q] * yr;
        }
    }
}