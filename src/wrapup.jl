using SparseArrays
using ExtendableGrids
using ExtendableSparse
using Metis
using Base.Threads
using ThreadPinning
pinthreads(:cores)

include("ESMP/ExtendableSparseParallel.jl")
using .ExtendableSparseParallel


include("CP_LNK/assembly.jl")
include("CP_LNK/conversion.jl")
include("CP_LNK/csc_assembly.jl")
include("CP_LNK/plusequals.jl")

include("ILU/pilu0.jl")
include("ILU/iluz.jl")
include("ILU/ILU_mittal_al-kurdi.jl")
include("ILU/PILU_mittal_al-kurdi.jl")
include("ILU/ilu_solver_wrap.jl")

include("benchmark/benchmark_fcts_da.jl")
include("benchmark/benchmark_fcts_con.jl")
include("benchmark/benchmark_fcts_dc.jl")
include("benchmark/benchmark_fcts_pe.jl")
include("benchmark/benchmark_fcts_ilu.jl")
include("benchmark/benchmark_fcts_sub.jl")


"""
In this function we want to wrap around the benchmarks for 6 tasks done in 3 ways.
Tasks:
- LNK assembly (da / dummy assembly)
- LNK -> CSC conversion (con)
- CSC assembly (dc / dummy csc assembly)
- CSC += new entries (pe / plusequals)
- computing ILU factors (ilu)
- using ILU (forward/backward substitution) (sub)

Ways (for the first 4 tasks):
- sequential
- cheap parallelization
- smart parallelization

Ways (for the last two tasks):
- ILUZero.jl
- modified method of Mittal & Al-Kurdi (sequential)
- modified method of Mittal & Al-Kurdi (parallel)


We use an FEM-like assembly/matrix on an `nm` grid (i.e. nm=(300,70) for 300 x 70 grid or nm=(100,80,40) for a 100 x 80 x 40 grid), run on `nt` threads.
`depth` gives the level of refinement steps (i.e. how often is the separator partitioned again?).
"""
function comparison(nm, nt, depth; num=10, offset=1)
	grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts = preparatory_multi_ps_less_reverse(nm, nt, depth)
	
	csc = spzeros(Float64, Int32, num_nodes(grid), num_nodes(grid))
	lnk = [SuperSparseMatrixLNK{Float64, Int32}(num_nodes(grid), nnts[tid]) for tid=1:nt]
	ESMP = ExtendableSparseMatrixParallel{Float64, Int32}(csc, lnk, grid, nnts, s, onr, cfp, gi, ni, rni, starts, nt, depth)
	
	nn = num_nodes(grid)
	cellnodes = grid[CellNodes]
	
	nc_nt = Int(num_cells(grid)/nt)
	cfpCP = [collect((tid-1)*nc_nt+1:tid*nc_nt) for tid=1:nt]
	
	@info "How many cells per partition:", [length(cfp[(i-1)*nt+1]) for i=1:depth+1]	
	
	
	
	As = bm_da(cellnodes, nn, nnts, s, cfp, nt, depth, gi, ni, cfpCP, num)
	
	C0 = bm_con(As, onr, s, nt, rni, num)
	
	Bs = bm_dc(C0, cellnodes, nn, nt, cfp, nnts, depth, s, 0, ni, offset, cfpCP, grid, num)
	
	C1 = bm_pe(Bs, nt, gi, num)
	
	C01, C02 = C0
	
	ESMP = bm_ESMP(ESMP, num, C02, C1) #C1 statt C01
	
	Is = bm_ilu(ESMP, num)
	bm_sub(Is, num)
	
	@info ""
end


function compare_dyn_sta(nm, nt, depth; offset=1, num=10)
	grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts = preparatory_multi_ps_less_reverse(nm, nt, depth)
	csc = spzeros(Float64, Int32, num_nodes(grid), num_nodes(grid))
	lnk = [SuperSparseMatrixLNK{Float64, Int32}(num_nodes(grid), nnts[tid]) for tid=1:nt]
	
	ESMP = ExtendableSparseMatrixParallel{Float64, Int32}(csc, lnk, grid, nnts, s, onr, cfp, gi, ni, rni, starts, nt, depth)
	for i=1:num
		reset!(ESMP)
		dummy_assembly!(ESMP; offset=0, skew=0.0, known_that_unknown=true, dynamic=true)
		flush!(ESMP; do_dense=true)
		A1 = copy(ESMP.cscmatrix)
		dummy_assembly!(ESMP; offset=1, diagval=3.0, dynamic=true)
		flush!(ESMP)
		A2 = copy(ESMP.cscmatrix)
	
		reset!(ESMP)
		dummy_assembly!(ESMP; offset=0, skew=0.0, known_that_unknown=true, dynamic=false)
		flush!(ESMP; do_dense=true)
		A3 = copy(ESMP.cscmatrix)
		dummy_assembly!(ESMP; offset=1, diagval=3.0, dynamic=false)
		flush!(ESMP)
		A4 = copy(ESMP.cscmatrix)
	
		
		compare_matrices_light(A1, A3)
		compare_matrices_light(A2, A4)
	end
	
end

function bm_ESMP(A, num, C0, C1)
	t1, t2, t3, t4, t5, a1, a2, a3, a4, A0, A1, A = benchmark_da_ESMP(A, num)
	
	
	@warn "ESMP:"
	@info "LNK:", form2(t1), a1*1.0
	@info "flu:", form2(t2), a2*1.0
	@info "CSC:", form2(t3), a3*1.0
	@info "flu:", form2(t4), a4*1.0
	@info "LNK'", form2(t5), 0.0*1.0
	
	@info "0: SEQ-ESMP:"
	compare_matrices_light(A0, C0)
	
	@info "1: SEQ-ESMP:"
	compare_matrices_light(A1, C1)
	A
end
	

	


function bm_da(cn, nn, nnts, s, cfp, nt, depth, gi, ni, cfpCP, num)
	t1, a1, A1 = benchmark_da_LNK_seq(cn, nn, ni, num)
	
	t2, a2, A2 = benchmark_da_LNK_par(cn, nn, cfpCP, nt, ni, num)
	
	#t3, a3, A3 = benchmark_da_RLNK_par(cn, nn, nnts, s, cfp, nt, depth, gi, ni, num)
	
	@warn "Dummy Assembly (LNK):"
	@info "SEQ:", form2(t1), a1*1.0
	@info "CP:", form2(t2), a2*1.0
	#@info "RLNK:", form2(t3), a3*1.0
	
	(A1, A2)
end


function bm_con(As, onr, s, nt, rni, num)
	A1, A2 = As
	t1, a1, C1 = zeros(num), 0, A1
	#t1, a1, C1 = benchmark_con_LNK_seq(A1, num)
	t2, a2, C2 = benchmark_con_LNK_par(A2, num)
	#t3, a3, C3 = benchmark_con_RLNK_par(A3, onr, s, nt, rni, num)
	
	@warn "Conversion:"
	@info "SEQ:", form2(t1), a1*1.0
	@info "CP:", form2(t2), a2*1.0
	#@info "RLNK:", form2(t3), a3*1.0
	
	#@info "SEQ-CP:"
	#compare_matrices_light(C1, C2)
	
	#@info "SEQ-RLNK:"
	#compare_matrices_light(C1, C3)
	
	(C1, C2)
end	
	

function bm_dc(C0, cn, nn, nt, cfp, nnts, depth, s, nr, ni, offset, cfpCP, grid, num)
	C01, C02 = C0
	t1, a1, C1 = benchmark_dc_LNK_seq(C01, cn, grid, ni, offset, num)
	
	t2, a2, C2, A2 = benchmark_dc_LNK_par(C02, cn, cfpCP, grid, ni, nt, offset, num)
	
	#t3, a3, C3, A3 = benchmark_dc_RLNK_par(C02, cn, nn, nt, cfp, nnts, depth, s, nr, ni, offset, num)
	
	@warn "Dummy Assembly (CSC):"
	@info "SEQ:", form2(t1), a1*1.0
	@info "CP:", form2(t2), a2*1.0
	#@info "RLNK:", form2(t3), a3*1.0
	
	(C1, C2, A2) #, C3, A3)
	
end


function bm_pe(Bs, nt, gi, num)
	C1, C2, A2 = Bs
	t1, a1 = zeros(num), 0
	t2, a2, C2 = benchmark_pe_LNK_par(C2, A2, nt, num)
	#t3, a3, C3 = benchmark_pe_RLNK_par(C3, A3, nt, gi, num)
	
	@warn "Plusequals:"
	@info "SEQ:", form2(t1), a1*1.0
	@info "CP:", form2(t2), a2*1.0
	#@info "RLNK:", form2(t3), a3*1.0
	
	#@info "SEQ-CP:"
	#compare_matrices_light(C1, C2)
	
	#@info "SEQ-RLNK:"
	#compare_matrices_light(C1, C3)
	C2
			
end
	
	
function bm_ilu(A, num)
	t1, a1, I1 = benchmark_ilu_ILUZero(A.cscmatrix, num)
	t2, a2, I2 = benchmark_ilu_AM_seq(A, num)
	t3, a3, I3 = benchmark_ilu_AM_par(A, num)
	
	@warn "ILU factorization:"
	@info "ILUZ:", form2(t1), a1*1.0
	@info "MA S:", form2(t2), a2*1.0
	@info "MA P:", form2(t3), a3*1.0
	
	(I1, I2, I3)
end

function bm_sub(Is, num)
	I1, I2, I3 = Is
	b = rand(I3.A.n)
	t1, a1, z1 = benchmark_sub_ILUZero(I1, b, num)
	t2, a2, z2 = benchmark_sub_AM_seq(I2, b, num)
	t3, a3, z3 = benchmark_sub_AM_par(I3, b, num)
	
	@warn "Substitution:"
	@info "ILUZ:", form2(t1), a1*1.0
	@info "MA S:", form2(t2), a2*1.0
	@info "MA P:", form2(t3), a3*1.0
	
	@info "||ILUZ-MA S|| = ", maximum(abs.(z1-z2))
	@info "||ILUZ-MA P|| = ", maximum(abs.(z1-z3))
	
	
end
	

function mean(x)
	sum(x)/length(x)
end

function form2(x)
	y = sort(x)
	[y[1], mean(y), y[Int(ceil(length(y)/2))], y[end]]
end

function form(x)
	[minimum(x), mean(x), maximum(x)]
end
	
function compare_matrices_light(A, B)
	D = A-B
	l = length(D.nzval)
	
	if l == 0
		@info "Matrices are equal"
	else
		md = maximum(abs.(D.nzval))
		@info "Matrices differ upto $md"
	end
	
end
	
	
	

