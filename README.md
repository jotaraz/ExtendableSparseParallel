# ExtendableSparseParallel

This repository contains a structure which should be used for matrix assembly in multithreaded Finite-Volume- or Finite-Element computations in Julia.
The structure and idea are very similar to Jürgen Fuhrmann's [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl).
This package should finally be dissolved into [ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl), [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) and [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl).

## Examples

In [src/examples/example_no_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_no_plot.jl) and [src/examples/example_with_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_with_plot.jl) you can use this package to solve a heat equation on a 2d triangular grid.
Both files contain the `validation(nm; depth=2, nt=3)` function that creates an `nm` grid, e.g. `nm=(300,250)` would be 300 x 250 grid. The `depth` refers to the levels of partitioning (if the grid is partitioned for each thread once, depth=1. If the separator is partitioned again, depth=2...). \
**depth should not exceed 2.** \
`nt` is the number of threads used.
The validation function will output:
- difference in solution of old (sequential) and new (parallel) algorithm
- timings (of one execution) of old and new algorithms
The validation function will return the assembled matrices.

In [src/examples/example_with_plot.jl](https://github.com/jotaraz/ExtendableSparseParallel/blob/main/src/examples/example_with_plot.jl) there also are `solve_parallel(nm, nt; depth=2, do_plot=true)` and `solve_sequential(nm; do_plot=true)` to visualize the results.


In general, running the example could look like this:

```
$ julia -t 8
julia> include("/home/.../ExtendableSparseParallel/src/examples/example_with_plot.jl")
julia> validation((1000,1000); nt=8);
[ Info: ("Max diff in solution: ", 3.6570746431152656e-13)
[ Info: Times: 
[ Info: ("Old LNK ____ ", (1.2530530799943553, 0.0008123850000000027))
[ Info: ("Old CSC ____ ", (0.7274713019927594, 0.0005748780000000169))
[ Info: ("New LNK ____ ", (0.825356517, 5.3147e-5, 0.002358855))
[ Info: ("New CSC ____ ", (0.424888863, 3.5282e-5, 0.000755888))
[ Info: ("New LNK para ", (0.147728244, 5.94e-5, 0.002292686))
[ Info: ("New CSC para ", (0.066440017, 4.6364e-5, 0.000724443))
julia> solve_parallel((200,200), 8);
```


