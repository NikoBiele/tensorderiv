//! The derivative operators as weighted sums over each point's stencil, as
//! DiscreteTensorDerivatives.jl computes them (gradient.jl and laplacian.jl):
//!
//!   gradient   ∇⊗Y ≈ Σ_k a_k Δx_k ⊗ ΔY_k
//!   Laplacian  ∇²Y ≈ 2 Σ_k a_k ΔY_k
//!
//! Divergence and curl are contractions of the gradient, done on the Python side.
//!
//! Field values are stored with one row of `components` values per point: a scalar field has
//! one component, a vector field `dims`, a matrix field the product of its dimensions, all
//! flattened row-major as NumPy stores them.

use rayon::prelude::*;

/// A point cloud's stencils, borrowed from the NumPy arrays of a Python StencilSet, all stored
/// with one row per point.
pub struct Stencils<'a> {
    /// Coordinates, `dims` per point
    pub points: &'a [f64],
    /// Number of dimensions
    pub dims: usize,
    /// Neighbour indices, `k` per point (signed, as NumPy stores them)
    pub neighbours: &'a [i64],
    /// Weights, `k` per point, in the order of the neighbours
    pub weights: &'a [f64],
    /// Neighbours per point
    pub k: usize,
}

impl Stencils<'_> {
    /// Number of points in the cloud.
    fn npoints(&self) -> usize {
        self.points.len() / self.dims
    }

    /// Check that all arrays fit together and that every neighbour index points into the
    /// cloud, so that the kernels can index without further checks.
    fn validate(&self) -> Result<(), String> {
        if self.dims == 0 || self.k == 0 || self.points.len() % self.dims != 0 {
            return Err("the stencil arrays are empty or misshapen".to_string());
        }
        let n = self.npoints();
        if self.neighbours.len() != n * self.k || self.weights.len() != n * self.k {
            return Err(format!("neighbours and weights must have {} entries for {n} points", n * self.k));
        }
        // `any` stops at the first index outside 0..n
        if self.neighbours.iter().any(|&j| j < 0 || j as usize >= n) {
            return Err(format!("a neighbour index is out of range for {n} points"));
        }
        Ok(())
    }
}

/// Check the field values and target points against the stencils.
fn check(stencils: &Stencils, values: &[f64], components: usize, targets: &[usize]) -> Result<(), String> {
    stencils.validate()?;
    let n = stencils.npoints();
    if components == 0 || values.len() != n * components {
        return Err(format!("the field must have one value per point ({n} points)"));
    }
    if let Some(&bad) = targets.iter().find(|&&i| i >= n) {
        return Err(format!("point index {bad} is out of range for {n} points"));
    }
    Ok(())
}

/// The gradient Σ_k a_k Δx_k ⊗ ΔY_k at each target point: one row of dims × components values
/// per target, with the derivative direction first, [a, c] = ∂Y_c/∂x_a.
pub fn gradient(
    stencils: &Stencils,
    values: &[f64],
    components: usize,
    targets: &[usize],
    threaded: bool,
) -> Result<Vec<f64>, String> {
    check(stencils, values, components, targets)?;
    let (dims, k) = (stencils.dims, stencils.k);
    let row = dims * components; // result values per target point
    let mut result = vec![0.0; targets.len() * row];

    // The gradient at point i, accumulated into its row `out`; only reads shared data, so
    // several threads can run it at once
    let fill = |(out, &i): (&mut [f64], &usize)| {
        let xi = &stencils.points[i * dims..(i + 1) * dims]; // coordinates of point i
        let yi = &values[i * components..(i + 1) * components]; // field value at point i
        for q in 0..k {
            let j = stencils.neighbours[i * k + q] as usize; // neighbour q of point i
            let w = stencils.weights[i * k + q]; // its weight
            let xj = &stencils.points[j * dims..(j + 1) * dims];
            let yj = &values[j * components..(j + 1) * components];
            for a in 0..dims {
                let wa = w * (xj[a] - xi[a]); // a_q Δx_qa
                let out_a = &mut out[a * components..(a + 1) * components];
                for c in 0..components {
                    out_a[c] += wa * (yj[c] - yi[c]); // += a_q Δx_qa ΔY_qc
                }
            }
        }
    };

    if threaded {
        // rows of the result paired with their target points, handed out to rayon's threads
        result.par_chunks_mut(row).zip(targets.par_iter()).for_each(fill);
    } else {
        result.chunks_mut(row).zip(targets.iter()).for_each(fill);
    }
    Ok(result)
}

/// The Laplacian 2 Σ_k a_k ΔY_k of each component at each target point: one row of
/// `components` values per target.
pub fn laplacian(
    stencils: &Stencils,
    values: &[f64],
    components: usize,
    targets: &[usize],
    threaded: bool,
) -> Result<Vec<f64>, String> {
    check(stencils, values, components, targets)?;
    let k = stencils.k;
    let mut result = vec![0.0; targets.len() * components];

    // The Laplacian at point i, accumulated into its row `out`
    let fill = |(out, &i): (&mut [f64], &usize)| {
        let yi = &values[i * components..(i + 1) * components]; // field value at point i
        for q in 0..k {
            let j = stencils.neighbours[i * k + q] as usize; // neighbour q of point i
            let w = stencils.weights[i * k + q]; // its weight
            let yj = &values[j * components..(j + 1) * components];
            for c in 0..components {
                out[c] += w * (yj[c] - yi[c]); // += a_q ΔY_qc
            }
        }
        for v in out.iter_mut() {
            *v *= 2.0; // the factor 2 of 2 Σ a_q ΔY_q
        }
    };

    if threaded {
        result.par_chunks_mut(components).zip(targets.par_iter()).for_each(fill);
    } else {
        result.chunks_mut(components).zip(targets.iter()).for_each(fill);
    }
    Ok(result)
}