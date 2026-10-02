# tensorderiv

Gradient, divergence, curl and Laplacian of scalar, vector and tensor fields sampled at scattered points, for Python. It needs no mesh and no connectivity: each operator is a weighted sum of differences over a point's nearest neighbours, with weights computed once per point cloud. The method is second-order accurate by default, and the core is written in Rust.

tensorderiv is a port of the Julia package [DiscreteTensorDerivatives.jl](https://github.com/NikoBiele/DiscreteTensorDerivatives.jl), by the same author, and is tested against it.

**Status: alpha.** All four operators work in any dimension (the curl in 3D), for fields of any rank.

## Installation

```
pip install tensorderiv
```

## Example

Build the stencils once for a point cloud, then apply any operator to any field sampled at those points:

```python
import numpy as np                                   # arrays
import tensorderiv as td                             # the package

points = np.random.rand(5000, 3)                     # 5000 random points in the unit cube, one row per point
stencils = td.StencilSet(points)                     # find neighbours and compute weights once (second order by default)

x, y, z = points.T                                   # the coordinate columns
v = np.column_stack([x * y, np.sin(z), x**2 * z])    # a vector field, one row per point

G = td.gradient(stencils, v)                         # shape (5000, 3, 3): G[p, a, j] = ∂v_j/∂x_a at point p
D = td.divergence(stencils, v)                       # shape (5000,): ∇·v at every point
C = td.curl(stencils, v)                             # shape (5000, 3): ∇×v at every point
L = td.laplacian(stencils, v)                        # shape (5000, 3): ∇²v, component by component
```

Every operator can also evaluate at selected points only, chosen with `at` as in NumPy indexing:

```python
td.gradient(stencils, v, at=42)                      # shape (3, 3): the gradient at point 42 only
td.divergence(stencils, v, at=[0, 1, 2])             # shape (3,): the divergence at three points
```

Because the gradient of any field is itself a field, derivatives can be chained. For example, the Hessian of a scalar field is the gradient of its gradient:

```python
f = np.sin(x) * np.exp(y)                            # a scalar field, one value per point
H = td.gradient(stencils, td.gradient(stencils, f))  # shape (5000, 3, 3): H[p, a, b] = ∂²f/∂x_a∂x_b
```

## Field layout

Points and field values always have the point index first, as SciPy and NumPy users expect:

| Field | `values` | `gradient` | `divergence` | `curl` (3D) | `laplacian` |
|---|---|---|---|---|---|
| Scalar | `(P,)` | `(P, N)` | – | – | `(P,)` |
| Vector | `(P, N)` | `(P, N, N)` | `(P,)` | `(P, N)` | `(P, N)` |
| Matrix | `(P, N, m)` | `(P, N, N, m)` | `(P, m)` | `(P, N, m)` | `(P, N, m)` |

Here `P` is the number of points and `N` the spatial dimension. The gradient puts the derivative direction right after the point index, so for a vector field `G[p, a, j]` $= \partial v_j / \partial x_a$, which is the transpose of the usual Jacobian. Divergence and curl act on the field's first axis after the points, which must therefore have length `N`.

## The method

Let $\mathbf{x}_0$ be a point and $\mathbf{x}_k$ its neighbours, with offsets $\Delta\mathbf{x}_k = \mathbf{x}_k - \mathbf{x}_0$ and field differences $\Delta Y_k = Y_k - Y_0$. A Taylor expansion about $\mathbf{x}_0$ gives

$$
\Delta Y_k = \Delta\mathbf{x}_k \cdot \nabla Y + \tfrac{1}{2}\, \Delta\mathbf{x}_k^\mathsf{T} \mathbf{H}\, \Delta\mathbf{x}_k + \mathcal{O}(h^3),
$$

where $\mathbf{H}$ is the Hessian and $h$ is the stencil radius. Now choose weights $a_k$ for the neighbours so that the first two moments of the stencil satisfy

$$
\sum_k a_k\, \Delta\mathbf{x}_k = \mathbf{0}, \qquad \sum_k a_k\, \Delta\mathbf{x}_k \Delta\mathbf{x}_k^\mathsf{T} = \mathbf{I}.
$$

Multiplying the Taylor expansion by $a_k \Delta\mathbf{x}_k$ and summing leaves the gradient, because the second moment is the identity. Multiplying by $a_k$ alone and summing removes the gradient term, because the first moment vanishes, and leaves half the trace of the Hessian. This gives all four operators as one-line sums:

$$
\nabla \otimes Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \otimes \Delta Y_k, \qquad
\nabla \cdot Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \cdot \Delta Y_k,
$$

$$
\nabla \times Y \approx \sum_k a_k\, \Delta\mathbf{x}_k \times \Delta Y_k, \qquad
\nabla^2 Y \approx 2 \sum_k a_k\, \Delta Y_k.
$$

The same weights serve every operator and every field, whether scalar, vector or tensor.

### Computing the weights

The moment conditions are linear in the weights. Written out component by component, with $\Delta x_{k,i}$ the $i$-th component of the offset $\Delta\mathbf{x}_k$, they are

$$
\sum_k a_k\, \Delta x_{k,i} = 0, \qquad \sum_k a_k\, \Delta x_{k,i}\, \Delta x_{k,j} = \delta_{ij} \quad (i \le j),
$$

which is $N(N+3)/2$ equations in $N$ dimensions, since the second moment is symmetric. Collecting them gives a small linear system $\mathbf{A}\mathbf{a} = \mathbf{b}$ with one column per neighbour. With more neighbours than equations, the system has many solutions, and the package picks the one with minimum norm:

$$
\mathbf{a} = \arg\min_{\mathbf{a}} \|\mathbf{a}\| \quad \text{subject to} \quad \mathbf{A}\mathbf{a} = \mathbf{b}.
$$

Small weights keep the operators from amplifying rounding errors and noise in the field values. The system is solved by orthonormalizing the rows of $\mathbf{A}$ with two passes of Gram–Schmidt, after scaling the offsets by the stencil radius so that all equations are of similar size. The conditions hold exactly when there are at least $N(N+3)/2$ neighbours in general position.

### Second order

The leading error of every operator comes from the third moment of the stencil. With `order=2`, the default, the weights are also required to satisfy

$$
\sum_k a_k\, \Delta x_{k,i}\, \Delta x_{k,j}\, \Delta x_{k,l} = 0 \quad (i \le j \le l).
$$

These $N(N+1)(N+2)/6$ equations are added to the same linear system, and they make all four operators second-order accurate.

## Accuracy and neighbour counts

| | Gradient, divergence, curl exact for | Laplacian exact for | Convergence | Minimum neighbours (2D / 3D) | Default `k` (2D / 3D) |
|---|---|---|---|---|---|
| `order=1` | linear fields | quadratic fields | first order | 5 / 9 | 10 / 18 |
| `order=2` (default) | quadratic fields | cubic fields | second order | 9 / 19 | 18 / 38 |

The default number of neighbours `k` is twice the minimum, because stencils at the minimum size are ill-conditioned. Set it explicitly with `StencilSet(points, k=30)`, and use `minimum_neighbours(N, order)` to find the lower bound. Accuracy is lower near the boundary of the point cloud, where stencils are one-sided.

## Performance

`StencilSet` does the expensive work, namely the neighbour search (with SciPy's `cKDTree`) and the weights of every point, and both run on all cores; `thread_count()` reports how many threads the Rust core uses. Pass `threaded=False` to `StencilSet` or to any operator when calling from code that is already parallel. On a 20-core desktop CPU, building second-order stencils for 100000 points in 3D takes about 0.13 seconds, and the gradient of a vector field on them about 7 milliseconds, so one `StencilSet` can be reused cheaply for any number of fields on the same points.

## Acknowledgements

This method descends from a 2016 Stack Overflow answer by a user named Hans, which introduced the moment conditions and the four operators above. That answer computed the weights through the Gram matrix $\mathbf{G} = \mathbf{B}^\mathsf{T}\mathbf{B}$ of the offset matrix $\mathbf{B}$, as $\mathbf{a} = \mathbf{P}\, \big( (\mathbf{G} \circ \mathbf{G})\, \mathbf{P} \big)^+ \mathrm{diag}(\mathbf{G})$, where $\mathbf{P}$ projects onto the null space of $\mathbf{B}$. This formula gives exactly the same weights as the minimum-norm solution above, because $\mathbf{G} \circ \mathbf{G}$ carries the same information as the second-moment equations. This package solves the moment equations directly, which is simpler and faster, and adds the second-order extension.

## Declaration of AI Assistance

The Rust core and much of the Python code of tensorderiv were written with substantial assistance from Claude (Anthropic). The method and the reference implementation come from the author's Julia package [DiscreteTensorDerivatives.jl](https://github.com/NikoBiele/DiscreteTensorDerivatives.jl), and the port is validated against it: the test suite compares tensorderiv with the Julia package for the stencil weights, the neighbour lists and all four operators, on scalar, vector and matrix fields in two and three dimensions. The author has reviewed and is responsible for all code.

## License

MIT