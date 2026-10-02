//! Weights of every stencil in a point cloud, computed in parallel, as
//! DiscreteTensorDerivatives.jl's StencilSet does it (stencils.jl).
//!
//! Points are stored with one row per point: point `i` occupies `points[i * dims .. (i + 1) * dims]`.
//! Neighbours are stored with one row per point too: the `k` neighbour indices of point `i`
//! occupy `neighbours[i * k .. (i + 1) * k]`, and its weights the same positions of the result.

use rayon::prelude::*;

use crate::weights::{self, Workspace};

/// The weights of every point's stencil, one row of `k` weights per point, in the order of the
/// neighbour indices. With `threaded`, the points are spread over rayon's threads, each with its
/// own workspace and offset buffer, so that no allocation happens per point.
pub fn all_weights(
    points: &[f64],
    dims: usize,
    neighbours: &[usize],
    k: usize,
    order: usize,
    threaded: bool,
) -> Result<Vec<f64>, String> {
    if order != 1 && order != 2 {
        return Err(format!("order must be 1 or 2, got {order}"));
    }
    if dims == 0 || points.len() % dims != 0 {
        return Err(format!("{} coordinates do not form points of {dims} dimensions", points.len()));
    }
    let npoints = points.len() / dims;
    if k == 0 || neighbours.len() != npoints * k {
        return Err(format!(
            "{} neighbour indices do not form rows of {k} for {npoints} points",
            neighbours.len()
        ));
    }
    // `find` returns the first index that is out of range, if any
    if let Some(&bad) = neighbours.iter().find(|&&j| j >= npoints) {
        return Err(format!("neighbour index {bad} is out of range for {npoints} points"));
    }

    let mut all = vec![0.0; npoints * k]; // the result: one row of weights per point

    // The weights of point i, written into its row `a`, with a workspace and an offset buffer
    // as scratch space. The closure only reads `points` and `neighbours`, so several threads
    // can run it at once.
    let fill = |scratch: &mut (Workspace, Vec<f64>), (i, a): (usize, &mut [f64])| -> Result<(), String> {
        let (workspace, dx) = scratch;
        let xi = &points[i * dims..(i + 1) * dims]; // coordinates of point i
        for (q, &j) in neighbours[i * k..(i + 1) * k].iter().enumerate() {
            for d in 0..dims {
                dx[q * dims + d] = points[j * dims + d] - xi[d]; // offset Δx_q = x_j − x_i
            }
        }
        weights::weights_into(a, workspace, dx, dims, order).map_err(|e| format!("point {i}: {e}"))
    };
    // Fresh scratch space: called once per thread by rayon, and once for the serial loop
    let init = || (Workspace::new(dims, k, order), vec![0.0; k * dims]);

    if threaded {
        // `par_chunks_mut(k)` hands out the rows of the result to rayon's threads;
        // `try_for_each_init` gives each thread its own scratch space and stops at the first error
        all.par_chunks_mut(k).enumerate().try_for_each_init(init, fill)?;
    } else {
        let mut scratch = init();
        for row in all.chunks_mut(k).enumerate() {
            fill(&mut scratch, row)?;
        }
    }
    Ok(all)
}