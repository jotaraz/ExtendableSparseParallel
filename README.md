# ExtendableSparseParallel

This repository contains a structure which should be used for matrix assembly in multithreaded Finite-Volume- or Finite-Element computations in Julia.
The structure and idea are very similar to Jürgen Fuhrmann's [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl).
This package should finally be dissolved into [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl), [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) and [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl).

## Examples

In [src/examples/example_no_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_no_plot.jl) and [src/examples/example_with_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_with_plot.jl) you can use this package to solve a heat equation on a 2d triangular grid.
Both files contain the `validation(nm; depth=2, nt=3)` function that creates an `nm` grid, e.g. `nm=(300,250)` would be 300 x 250 grid. The `depth` refers to the levels of partitioning (if the grid is partitioned for each thread once, depth=1. If the separator is partitioned again, depth=2...).
**depth should not exceed 2.**
`nt` is the number of threads used.
The validation function will output:
- difference in solution of old (sequential) and new (parallel) algorithm
- timings (of one execution) of old and new algorithms
The validation function will return the assembled matrices.

In [src/examples/example_with_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_with_plot.jl) there also are `solve_parallel(nm, nt; depth=2, do_plot=true)` and `solve_sequential(nm; do_plot=true)` to visualize the results.




