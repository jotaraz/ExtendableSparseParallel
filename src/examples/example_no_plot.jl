# example without plotting

include("../ESMP/ExtendableSparseParallel.jl")
using .ExtendableSparseParallel

using ExtendableGrids, SparseArrays, Printf
using ExtendableSparse
using Base.Threads

using ThreadPinning

pinthreads(:cores)


include("assemble_ESM3.jl")
include("formfactors.jl")



### problem definition:

f(x,y)=sinpi(x)*cospi(y)
β(x,y)=x*(1-x)

δ = 0.05
α = 5e3

### csc comparison

function compare_matrices_light(A, B; name="")
	D = A-B
	l = length(D.nzval)
	
	if l == 0
		@info name*" Matrices are equal"
	else
		md = maximum(abs.(D.nzval))
		@info name*" Matrices differ upto $md"
	end	
end


### run & time example

function validation(nm; depth=2, nt=3)
	grid = ExtendableSparseParallel.getgrid(nm; x0=-1.0, x1=1.0)
	n    = num_nodes(grid) #size(triout.pointlist,2)
	
	# ExtendableSparse
	# LNK Assembly (= unknown structure)
	mat0 = ExtendableSparseMatrix{Float64, Int64}(n, n)
	rhs0 = zeros(n)
	t1   = assemble!(mat0,rhs0,δ,f,α,β,grid)
	
	# flush
	flush!(mat0)
	mat0.cscmatrix.nzval .= 0
	
	# CSC Assembly (= known structure)
	rhs0 = zeros(n)
	t2   = assemble!(mat0,rhs0,δ,f,α,β,grid)
	
	# ExtendableSparseParalle (not parallelized)
	# LNK Assembly (= unknown structure)
	mat1 = ExtendableSparseParallel.ExtendableSparseMatrixParallel{Float64, Int64}(nm, nt, depth; x0=-1.0, x1=1.0)
	rhs1 = zeros(n)
	t3   = assemble_part_ESMP_essential3!(mat1,rhs1,δ,f,α,β,grid)
	
	# flush
	ExtendableSparseParallel.flush!(mat1)
	mat1.cscmatrix.nzval .= 0
	
	# CSC Assembly (= known structure)
	rhs1 = zeros(n)
	t4   = assemble_part_ESMP_essential3!(mat1,rhs1,δ,f,α,β,grid)
	
	# ExtendableSparseParalle (parallelized)
	# LNK Assembly (= unknown structure)
	mat2 = ExtendableSparseParallel.ExtendableSparseMatrixParallel{Float64, Int64}(nm, nt, depth; x0=-1.0, x1=1.0)
	rhs2 = zeros(n)
	t5   = assemble_part_para_ESMP_essential3!(mat2,rhs2,δ,f,α,β,grid)
	
	# flush
	ExtendableSparseParallel.flush!(mat2)
	mat2.cscmatrix.nzval .= 0
	
	# CSC Assembly (= known structure)
	rhs2  = zeros(n)
	t6   = assemble_part_para_ESMP_essential3!(mat2,rhs2,δ,f,α,β,grid)
	
	sol0 = SparseArrays.SparseMatrixCSC(mat0) \ rhs0
	ExtendableSparseParallel.flush!(mat2)
	sol2 = mat2.cscmatrix \ rhs2
	sol2 = reorder(sol2, mat2.rev_new_indices)
	
	
	@info "Max diff in solution: ", maximum(abs.(sol0-sol2))
	
	@info "Times: "
	@info "Old LNK ____ ", t1
	@info "Old CSC ____ ", t2
	@info "New LNK ____ ", t3
	@info "New CSC ____ ", t4
	@info "New LNK para ", t5
	@info "New CSC para ", t6
	
	mat0, mat1, mat2
	
end


function reorder(x, ni)
	y = copy(x)
	for i=1:length(x)
		y[ni[i]] = x[i]
	end
	y

end

