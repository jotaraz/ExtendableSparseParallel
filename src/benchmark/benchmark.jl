using SparseArrays
using ExtendableSparse
using ExtendableGrids
using Metis
using Base.Threads

nt = nthreads()

include(path*"preparatory.jl")
include(path*"assembly.jl")
include(path*"csc_assembly.jl")
include(path*"conversion.jl")
include(path*"validation.jl")
include(path*"plusequals.jl")

#validate((10*nt, 10*nt+1))

fct_names = [["CP", "Rm1", "Rm2", "Rm1l", "Rm2l"], ["CP", "Rm1", "Rm2", "Rm1_dz", "Rm2_dz", "Rm1l_dz", "Rm2l_dz"], ["CP", "Rm1", "Rm2", "Rm1l", "Rm2l"], ["CP", "Rm1l_3", "Rm1l_u", "Rm2l_3", "Rm2l_u"]]

endings = ["assembly", "conversion", "cscassembly", "plusequals"]

nms_    = []
grid_   = []
gridm1_ = []
nntsm1_ = []
sm1_    = []
nrm1_   = []
cfpm1_  = []
gridm2_ = []
nntsm2_ = []
sm2_    = []
nrm2_   = []
cfpm2_  = []
gridm1l_ = []
nntsm1l_ = []
sm1l_    = []
nrm1l_   = []
cfpm1l_  = []
gridm2l_ = []
nntsm2l_ = []
sm2l_    = []
nrm2l_   = []
cfpm2l_  = []

As_     = []
Ab_     = []
Am1_    = []
Am2_    = []
Cs_     = []
C0_     = []
Cm1_    = []
Cm2_    = []
Am1l_    = []
Am2l_    = []
Cm1l_    = []
Cm2l_    = []
Abm1l_    = []
Abm2l_    = []
C0m1l_    = []
C0m2l_    = []


function array_ci(x)
	if length(x)==0
		@info "length 0 "
	else
		@info "length $(length(x)), max $(maximum(abs.(x))) "
	end
end

"""
`function add_a_grid(nm)`

Add a 2d or 3d grid with (nm)=n,m or (nm)=n,m,l nodes.
Create the preparatory stuff for all assembly functions.
"""
function add_a_grid(nm)
	push!(nms_, nm)
	grid = getgrid(nm)
	push!(grid_, grid)
	push!(As_, da_LNK_cp_sz(grid, nt))
	A1 = copy(As_[end])
	push!(Cs_, CSC_LNKs_s!(A1))
	C1 = copy(Cs_[end])
	C0, Ab = da_csc_LNK_cp!(C1, grid[CellNodes], grid; offset=0)	
	push!(Ab_, Ab)
	push!(C0_, C0)

	grid, nnts, s, nr, cfp = preparatory_multi_ps(nm, nt, 1)
	push!(gridm1_, grid)
	push!(nntsm1_, nnts)
	push!(sm1_, s)
	push!(nrm1_, nr)
	push!(cfpm1_, cfp)
	push!(Am1_, da_RLNK_oc_ps_sz(grid, nnts, s, cfp, nt, 1))
	push!(Cm1_, CSC_RLNK_si_oc_ps_dz(Am1_[end], nr, s, nt, 1))
	
	grid, nnts, s, nr, cfp = preparatory_multi_ps(nm, nt, 2)
	push!(gridm2_, grid)
	push!(nntsm2_, nnts)
	push!(sm2_, s)
	push!(nrm2_, nr)
	push!(cfpm2_, cfp)
	push!(Am2_, da_RLNK_oc_ps_sz(grid, nnts, s, cfp, nt, 2))
	push!(Cm2_, CSC_RLNK_si_oc_ps_dz(Am2_[end], nr, s, nt, 2))
	
	grid, nnts, s, nr, cfp = preparatory_multi_ps_less(nm, nt, 1)
	push!(gridm1l_, grid)
	push!(nntsm1l_, nnts)
	push!(sm1l_, s)
	push!(nrm1l_, nr)
	push!(cfpm1l_, cfp)
	push!(Am1l_, da_RLNK_oc_ps_sz_less(grid[CellNodes], num_nodes(grid), nnts, s, cfp, nt, 1))
	push!(Cm1l_, CSC_RLNK_si_oc_ps_dz_less(Am1l_[end], nr, s, nt))
	C1 = copy(Cm1l_[end])
	C0, Ab = csc_assembly_pe_less_new2!(C1, grid[CellNodes], num_nodes(grid), nt, cfp, nnts, 1, s, nr)
	push!(Abm1l_, Ab)
	push!(C0m1l_, C0)
	
	grid, nnts, s, nr, cfp = preparatory_multi_ps_less(nm, nt, 2)
	push!(gridm2l_, grid)
	push!(nntsm2l_, nnts)
	push!(sm2l_, s)
	push!(nrm2l_, nr)
	push!(cfpm2l_, cfp)
	push!(Am2l_, da_RLNK_oc_ps_sz_less(grid[CellNodes], num_nodes(grid), nnts, s, cfp, nt, 2))
	push!(Cm2l_, CSC_RLNK_si_oc_ps_dz_less(Am2l_[end], nr, s, nt))
	C1 = copy(Cm2l_[end])
	C0, Ab = csc_assembly_pe_less_new2!(C1, grid[CellNodes], num_nodes(grid), nt, cfp, nnts, 2, s, nr)
	push!(Abm2l_, Ab)
	push!(C0m2l_, C0)
	
	
	#check if file exists already, if not, create it
	for (i,ending) in enumerate(endings)
		fn = pre_name*"_"*tts(nm, nt)*"_"*ending*".txt"
		
		io = open(fn, "a")
		close(io)
		
		io = open(fn, "r")
		if length(readlines(io)) == 0
			#@warn "file does not exist yet"
			close(io)
			io = open(fn, "w")
			write(io, "ss 0\n")
			#write(io, array_to_string(fct_names))
			for fct_name in fct_names[i]
				write(io, fct_name*"\n")
			end		
		end
		
		close(io)
	end
		
end


"""
`function strarray_to_string(strings)`

`["a", "b", "c"] -> "a b c"`
"""
function strarray_to_string(strings) 
	s = strings[1]
	for string in strings[2:end]
		s = s*" "*string
	end
	s
end

"""
`function numarray_to_strarray(nums)`

`[1, 3, 4] -> ["1", "3", "4"]`
"""
function numarray_to_strarray(nums)
	strings = v = Vector{String}(undef, length(nums))
	for (i,x) in enumerate(nums)
		strings[i] = "$x"
	end
	strings
end


"""
`function tts(xx, y)`

Turns `(a,b),y -> "2_str(y)_str(a)_str(b)"` or `(a,b,c),y -> "3_str(y)_str(a)_str(b)_str(c)"`.
"""
function tts(xx, y) #tuple to string
	s = "$(length(xx))_$(y)"
	for x in xx
		s = s*"_$x"
	end
	s
end


### Benchmark assembly functions:
###----------------------------------------------------------------------------------------------

"""
`function bm_da_CP(flag, ss; doallocs=false)`

Assembles the flag'th grid using `cheap parallelization`.
Does this `ss` times and returns the recorded times.
The result can be used for comparison.
"""
function bm_da_CP(flag, ss; doallocs=false)
	grid = grid_[flag]
	t  = zeros(ss)
	for j=1:ss
		t[j] = @elapsed da_LNK_cp_sz(grid, nt)
		GC.gc()
	end
	t
end

function al_da_CP(flag)
	grid = grid_[flag]
	@allocated da_LNK_cp_sz(grid, nt)
end



"""
`function bm_da_Rocm1(flag, ss; doallocs=false)`

Assembles the flag'th grid using `RLNK_ocm1` (overlapping columns using the function for arbitrary levels of separator partition, here only one level: the separator is not partitioned further: nt+1 matrices).
Does this `ss` times and returns the recorded times.
"""
function bm_da_Rm1(flag, ss; doallocs=false)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	depth = 1
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed da_RLNK_oc_ps_sz(gridm1, nntsm1, sm1, cfpm1, nt, depth)
		GC.gc()
	end
	t
end

function al_da_Rm1(flag)
	depth = 1
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	@allocated da_RLNK_oc_ps_sz(gridm1, nntsm1, sm1, cfpm1, nt, depth)
end


"""
`function bm_da_Rocm2(flag, ss; doallocs=false)`

Assembles the flag'th grid using `RLNK_ocm2` (overlapping columns using the function for arbitrary levels of separator partition, here two levels: 2*nt+1 matrices).
Does this `ss` times and returns the recorded times.
"""
function bm_da_Rm2(flag, ss; doallocs=false)
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	depth = 2
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed da_RLNK_oc_ps_sz(gridm2, nntsm2, sm2, cfpm2, nt, depth)
		GC.gc()
	end
	t
end	

function al_da_Rm2(flag)
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	depth = 2
	@allocated da_RLNK_oc_ps_sz(gridm2, nntsm2, sm2, cfpm2, nt, depth)
end	


"""
`function bm_da_Rm1l(flag, ss; doallocs=false)`
"""
function bm_da_Rm1l(flag, ss; doallocs=false)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	depth = 1
	cn = gridm1[CellNodes]
	nn = num_nodes(gridm1)
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed da_RLNK_oc_ps_sz_less(cn, nn, nntsm1, sm1, cfpm1, nt, depth)
		GC.gc()
	end
	t
end


function al_da_Rm1l(flag)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	depth = 1
	cn = gridm1[CellNodes]
	nn = num_nodes(gridm1)
	
	@allocated da_RLNK_oc_ps_sz_less(cn, nn, nntsm1, sm1, cfpm1, nt, depth)
end

"""
`function bm_da_Rm2l(flag, ss; doallocs=false)`
"""
function bm_da_Rm2l(flag, ss; doallocs=false)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	depth = 2
	cn = gridm1[CellNodes]
	nn = num_nodes(gridm1)
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed da_RLNK_oc_ps_sz_less(cn, nn, nntsm1, sm1, cfpm1, nt, depth)
		GC.gc()
	end
	t
end


function al_da_Rm2l(flag)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	depth = 2
	cn = gridm1[CellNodes]
	nn = num_nodes(gridm1)
	
	@allocated da_RLNK_oc_ps_sz_less(cn, nn, nntsm1, sm1, cfpm1, nt, depth)
end

### Benchmark conversion functions:
###----------------------------------------------------------------------------------------------

"""
`function bm_c_LNKs_s(ss, grid)`

Benchmarks the conversion of multiple LNK matrices to one CSC matrix
"""
function bm_c_CP(flag, ss)
	grid = grid_[flag]
	
	t = zeros(ss)
	for j=1:ss
		CP = copy(As_[flag])
		t[j] = @elapsed CSC_LNKs_s!(CP)
		GC.gc()
	end
	t
end

function al_c_CP(flag)
	grid = grid_[flag]
	CP = copy(As_[flag])
	@allocated CSC_LNKs_s!(CP)
end

"""
`function bm_c_RLNK_ps1(flag, ss)`

Benchmarks the conversion of an RLNK matrix with depth 1 (i.e. multiple LNK sub-matrices) to a CSC matrix.
"""
function bm_c_Rm1(flag, ss)
	depth = 1
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	Roc = Am1_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps(Roc, nrm1, sm1, nt, depth)
		GC.gc()
	end
	t
end

function al_c_Rm1(flag)
	depth = 1
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	Roc = Am1_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	@allocated CSC_RLNK_si_oc_ps(Roc, nrm1, sm1, nt, depth)
end


"""
`function bm_c_RLNK_ps1(flag, ss)`

Benchmarks the conversion of an RLNK matrix with depth 1 (i.e. multiple LNK sub-matrices) to a CSC matrix.
"""
function bm_c_Rm2(flag, ss)
	depth = 2
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	Roc = Am2_[flag] #da_RLNK_oc_ps(gridm2, nntsm2, sm2, cfpm2, nt, depth)

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps(Roc, nrm2, sm2, nt, depth)
		GC.gc()
	end
	t
end

function al_c_Rm2(flag)
	depth = 2
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	Roc = Am2_[flag] #da_RLNK_oc_ps(gridm2, nntsm2, sm2, cfpm2, nt, depth)
	@allocated CSC_RLNK_si_oc_ps(Roc, nrm2, sm2, nt, depth)
end


"""
`function bm_c_RLNK_ps1_dz(flag, ss)`

Benchmarks the conversion of an RLNK matrix with depth 1 (i.e. multiple LNK sub-matrices) to a CSC matrix.
"""
function bm_c_Rm1_dz(flag, ss)
	depth = 1
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	Roc = Am1_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps_dz(Roc, nrm1, sm1, nt, depth)
		GC.gc()
	end
	t
end

function al_c_Rm1_dz(flag)
	depth = 1
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	Roc = Am1_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	@allocated CSC_RLNK_si_oc_ps_dz(Roc, nrm1, sm1, nt, depth)
end


"""
`function bm_c_RLNK_ps1_dz(flag, ss)`

Benchmarks the conversion of an RLNK matrix with depth 1 (i.e. multiple LNK sub-matrices) to a CSC matrix.
"""
function bm_c_Rm2_dz(flag, ss)
	depth = 2
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	Roc = Am2_[flag] #da_RLNK_oc_ps(gridm2, nntsm2, sm2, cfpm2, nt, depth)

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps_dz(Roc, nrm2, sm2, nt, depth)
		GC.gc()
	end
	t
end

function al_c_Rm2_dz(flag)
	depth = 2
	gridm2, nntsm2, sm2, nrm2, cfpm2 = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	Roc = Am2_[flag] #da_RLNK_oc_ps(gridm2, nntsm2, sm2, cfpm2, nt, depth)
	@allocated CSC_RLNK_si_oc_ps_dz(Roc, nrm2, sm2, nt, depth)
end

"""
less
"""
function bm_c_Rm1l_dz(flag, ss)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	Roc = Am1l_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps_dz_less(Roc, nrm1, sm1, nt)
		GC.gc()
	end
	t
end

function al_c_Rm1l_dz(flag)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	Roc = Am1l_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	

	@allocated CSC_RLNK_si_oc_ps_dz_less(Roc, nrm1, sm1, nt)
end

function bm_c_Rm2l_dz(flag, ss)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	Roc = Am2l_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	

	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_si_oc_ps_dz_less(Roc, nrm1, sm1, nt)
		GC.gc()
	end
	t
end

function al_c_Rm2l_dz(flag)
	gridm1, nntsm1, sm1, nrm1, cfpm1 = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	Roc = Am2l_[flag] #da_RLNK_oc_ps(gridm1, nntsm1, sm1, cfpm1, nt, depth)
	

	@allocated CSC_RLNK_si_oc_ps_dz_less(Roc, nrm1, sm1, nt)
end


###	Benchmark CSC assembly functions
### --------------------------------------------------------------------


### Benchmark CSC assembly

"""
`function bm_ca_CP(flag, ss; offset=1)`

"""
function bm_ca_CP(flag, ss; offset=1)
	grid = grid_[flag]
	A = da_LNK_cp_sz(grid, nt)
	C0 = dropzeros(CSC_LNKs_s!(A))
	cn = grid[CellNodes]
	t = zeros(ss)
	for j=1:ss
		C = copy(C0)
		t[j] = @elapsed da_csc_LNK_cp!(C, cn, grid; offset=offset)
		GC.gc()
	end
	t
end

function al_ca_CP(flag; offset=1)
	grid = grid_[flag]
	cn = grid[CellNodes]
	A = da_LNK_cp_sz(grid, nt)
	C = dropzeros(CSC_LNKs_s!(A))
	@allocated da_csc_LNK_cp!(C, cn, grid; offset=offset)
end

"""
`function bm_ca_Rm1(flag, ss; offset=1)`

"""
function bm_ca_Rm1(flag, ss; offset=1)
	grid, nnts, s, nr, cfp = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	cn = grid[CellNodes]
	A = Am1_[flag]#da_RLNK_oc_ps(grid, nnts, s, cfp, nt, 1)
	C0 = Cm1_[flag]#dropzeros(CSC_RLNK_si_oc_ps(A, nr, s, nt, 1))
	
	t = zeros(ss)
	for j=1:ss
		C = copy(C0)
		t[j] = @elapsed da_csc_RLNK_oc_ps_sz!(C, cn, grid, nnts, s, cfp, nr, nt, 1; offset=offset)
		GC.gc()
	end
	t
end

function al_ca_Rm1(flag; offset=1)
	grid, nnts, s, nr, cfp = gridm1_[flag], nntsm1_[flag], sm1_[flag], nrm1_[flag], cfpm1_[flag]
	cn = grid[CellNodes]
	A = Am1_[flag]#da_RLNK_oc_ps(grid, nnts, s, cfp, nt, 1)
	C0 = copy(Cm1_[flag]) #dropzeros(CSC_RLNK_si_oc_ps(A, nr, s, nt, 1))
	
	@allocated da_csc_RLNK_oc_ps_sz!(C0, cn, grid, nnts, s, cfp, nr, nt, 1; offset=offset)
	
end









"""
`function bm_ca_Rm2(flag, ss; offset=1)`

"""
function bm_ca_Rm2(flag, ss; offset=1)
	grid, nnts, s, nr, cfp = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	cn = grid[CellNodes]
	A = Am2_[flag]
	C0 = Cm2_[flag]
	
	t = zeros(ss)
	for j=1:ss
		C = copy(C0)
		t[j] = @elapsed da_csc_RLNK_oc_ps_sz!(C, cn, grid, nnts, s, cfp, nr, nt, 2; offset=offset)
		GC.gc()
	end
	t
end

function al_ca_Rm2(flag; offset=1)
	grid, nnts, s, nr, cfp = gridm2_[flag], nntsm2_[flag], sm2_[flag], nrm2_[flag], cfpm2_[flag]
	cn = grid[CellNodes]
	A = Am2_[flag]
	C0 = copy(Cm2_[flag])
	
	@allocated da_csc_RLNK_oc_ps_sz!(C0, cn, grid, nnts, s, cfp, nr, nt, 2; offset=offset)
	
end

"""
`function bm_ca_Rm1(flag, ss; offset=1)`

"""
function bm_ca_Rm1l(flag, ss; offset=1)
	grid, nnts, s, nr, cfp = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	A = Am1_[flag]
	C0 = Cm1_[flag]
	
	t = zeros(ss)
	for j=1:ss
		C = copy(C0)
		t[j] = @elapsed csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, 1, s, nr; offset=offset)
		GC.gc()
	end
	t
end

function al_ca_Rm1l(flag; offset=1)
	grid, nnts, s, nr, cfp = gridm1l_[flag], nntsm1l_[flag], sm1l_[flag], nrm1l_[flag], cfpm1l_[flag]
	
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	A = Am1_[flag]
	C0 = copy(Cm1_[flag])
	
	@allocated csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, 1, s, nr; offset=offset)
	
end

"""
`function bm_ca_Rm2(flag, ss; offset=1)`

"""
function bm_ca_Rm2l(flag, ss; offset=1)
	grid, nnts, s, nr, cfp = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	A = Am2_[flag]
	C0 = Cm2_[flag]
	
	t = zeros(ss)
	for j=1:ss
		C = copy(C0)
		t[j] = @elapsed csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, 2, s, nr; offset=offset)
		GC.gc()
	end
	t
end

function al_ca_Rm2l(flag; offset=1)
	grid, nnts, s, nr, cfp = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	A = Am2_[flag]
	C0 = copy(Cm2_[flag])
	
	@allocated csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, 2, s, nr; offset=offset)
	
end

function count_ca_Rm2l(flag; offset=1)
	grid, nnts, s, nr, cfp = gridm2l_[flag], nntsm2l_[flag], sm2l_[flag], nrm2l_[flag], cfpm2l_[flag]
	
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	A = Am2_[flag]
	C0 = Cm2_[flag]
	C1 = copy(C0)
	csc_assembly_pe_less_new2_count!(C1, cn, nn, nt, cfp, nnts, 2, s, nr; offset=offset)
end


###	Benchmark plusequals
### --------------------------------------------------------------------
run
function bm_pe_Rm1l_u(flag, ss)
	As, C, nr, s = Abm1l_[flag], C0m1l_[flag], nrm1l_[flag], sm1l_[flag]
	
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_plusequals_lessu(As, C, nr, s, nt)
		GC.gc()
	end
	t
end

function bm_pe_Rm2l_u(flag, ss)
	As, C, nr, s = Abm2l_[flag], C0m2l_[flag], nrm2l_[flag], sm2l_[flag]
	
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_plusequals_lessu(As, C, nr, s, nt)
		GC.gc()
	end
	t
end

function bm_pe_Rm1l_3(flag, ss)
	As, C, nr, s = Abm1l_[flag], C0m1l_[flag], nrm1l_[flag], sm1l_[flag]
	
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_plusequals_less3(As, C, nr, s, nt)
		GC.gc()
	end
	t
end

function bm_pe_Rm2l_3(flag, ss)
	As, C, nr, s = Abm2l_[flag], C0m2l_[flag], nrm2l_[flag], sm2l_[flag]
	
	t = zeros(ss)
	for j=1:ss
		t[j] = @elapsed CSC_RLNK_plusequals_less3(As, C, nr, s, nt)
		GC.gc()
	end
	t
end

function bm_pe_LNK(flag, ss)
	As, C = Ab_[flag], C0_[flag]
	t = zeros(ss)
	
	for j=1:ss
		t[j] = @elapsed CSC_LNK_plusequals(As, C, nt)
		GC.gc()
	end
	t
end

function al_pe_Rm1l_u(flag)
	As, C, nr, s = Abm1l_[flag], C0m1l_[flag], nrm1l_[flag], sm1l_[flag]
	
	@allocated CSC_RLNK_plusequals_lessu(As, C, nr, s, nt)
end

function al_pe_Rm2l_u(flag)
	As, C, nr, s = Abm2l_[flag], C0m2l_[flag], nrm2l_[flag], sm2l_[flag]
	
	@allocated CSC_RLNK_plusequals_lessu(As, C, nr, s, nt)
end

function al_pe_Rm1l_3(flag)
	As, C, nr, s = Abm1l_[flag], C0m1l_[flag], nrm1l_[flag], sm1l_[flag]
	
	@allocated CSC_RLNK_plusequals_less3(As, C, nr, s, nt)
end

function al_pe_Rm2l_3(flag)
	As, C, nr, s = Abm2l_[flag], C0m2l_[flag], nrm2l_[flag], sm2l_[flag]
	
	@allocated CSC_RLNK_plusequals_less3(As, C, nr, s, nt)
end



function al_pe_LNK(flag)
	As, C = Ab_[flag], C0_[flag]
	
	@allocated CSC_LNK_plusequals(As, C, nt)
end



### compare sequential and parallel plusequals

function test_pe_p(num, flag, nt, nt2; seq=true)
	s = sm2l_[flag]
	nr = nrm2l_[flag]
	As = Am2l_[flag]
	
	C1 = CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
	colptr1 = colptr_RLNK_sequential_less(As, nr, s)
	colptr2 = colptr_RLNK_parallel_less(As, nr, s, nt2)
	
	@info maximum(abs.(C1.colptr-colptr1))
	@info maximum(abs.(C1.colptr-colptr2))
	
	if seq
		ts = zeros(num)
		for i=1:num
			ts[i] = @elapsed colptr_RLNK_sequential_less(As, nr, s)
			GC.gc()
		end
		@info "seq : $(minimum(ts))"
	end
	
	tp = zeros(num)
	for i=1:num
		tp[i] = @elapsed colptr_RLNK_parallel_less(As, nr, s, nt2)
		GC.gc()
	end
	@info "$nt2 : $(minimum(tp))"
	
end

function bm_pe_seq_p1(num, flag::Integer, nt, nt2)
	#gridm2l_, grid)
	#push!(nntsm2l_, nnts)
	
	s = sm1l_[flag]
	nr = nrm1l_[flag]
	As = Am1l_[flag]
	
	C1 = CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
	C2 = CSC_RLNK_si_oc_ps_dz_less_p1(As, nr, s, nt, nt2)
	#@info (C1-C2).nzval
	
	
	array_ci(C1.colptr-C2.colptr)
	array_ci((C1-C2).nzval)
	
	CSC_RLNK_si_oc_ps_dz_less_p1_time(As, nr, s, nt, nt2)
	
	
	#=
	#As = da_RLNK_oc_ps_sz_less(grid[CellNodes], num_nodes(grid), nnts, s, cfp, nt, 2)
	t1 = zeros(num)
	for i=1:num
		t1[i] = @elapsed CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
		GC.gc()
	end
	
	t2 = zeros(num)
	for i=1:num
		t2[i] = @elapsed CSC_RLNK_si_oc_ps_dz_less_p1(As, nr, s, nt, nt)
		GC.gc()
	end
	
	a1 = @allocated CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
	a2 = @allocated CSC_RLNK_si_oc_ps_dz_less_p1(As, nr, s, nt, nt)
	
	@info "time min seq $(minimum(t1)), min par $(minimum(t2)), ratio $(minimum(t1)/minimum(t2))"
	@info "alloc seq $a1, par $a2, ration $(a1/a2)"
	=#
end

function bm_pe_seq_par2(num, flag::Integer, nt)
	#gridm2l_, grid)
	#push!(nntsm2l_, nnts)
	
	s = sm2l_[flag]
	nr = nrm2l_[flag]
	As = Am2l_[flag]
	
	C1 = CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
	C2 = CSC_RLNK_si_oc_ps_dz_less_parallel2(As, nr, s, nt, nt)
	@info (C1-C2).nzval
	#As = da_RLNK_oc_ps_sz_less(grid[CellNodes], num_nodes(grid), nnts, s, cfp, nt, 2)
	t1 = zeros(num)
	for i=1:num
		t1[i] = @elapsed CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
		GC.gc()
	end
	
	t2 = zeros(num)
	for i=1:num
		t2[i] = @elapsed CSC_RLNK_si_oc_ps_dz_less_parallel2(As, nr, s, nt, nt)
		GC.gc()
	end
	
	a1 = @allocated CSC_RLNK_si_oc_ps_dz_less(As, nr, s, nt)
	a2 = @allocated CSC_RLNK_si_oc_ps_dz_less_parallel2(As, nr, s, nt, nt)
	
	@info "time min seq $(minimum(t1)), min par $(minimum(t2)), ratio $(minimum(t1)/minimum(t2))"
	@info "alloc seq $a1, par $a2, ration $(a1/a2)"
	
end

function bm_pe_seq_par2_backup(num, flag::Integer, nt)
	#gridm2l_, grid)
	#push!(nntsm2l_, nnts)
	
	s = sm2l_[flag]
	nr = nrm2l_[flag]
	C0 = C0m2l_[flag]
	Ab0 = Abm2l_[flag]
	
	@info [maximum(abs.(Ab0[i].nzval)) for i=1:nt]
	
	grid = gridm2l_[flag]
	cn = grid[CellNodes]
	
	C0, Ab = da_csc_LNK_cp!(C0, cn, grid; offset=1)
	
	@info [maximum(abs.(Ab[i].nzval)) for i=1:nt]
	@info [Ab[i].nnz for i=1:nt]
	
	
	
	t1 = zeros(num)
	t2 = zeros(num)
	t3 = zeros(num)
	for i=1:num
		t1[i] = @elapsed (Cb = CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt))
		t2[i] = @elapsed (Cs = C0 + Cb)
		t3[i] = @elapsed dropzeros!(Cs)
		GC.gc()
	end
	
	t4 = zeros(num)
	t5 = zeros(num)
	t6 = zeros(num)
	for i=1:num
		t4[i] = @elapsed (Cb = CSC_RLNK_si_oc_ps_dz_less_parallel2(Ab, nr, s, nt, nt))
		t5[i] = @elapsed (Cs = C0 + Cb)
		t6[i] = @elapsed dropzeros!(Cs)
		GC.gc()
	end
	
	a1 = @allocated CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt)
	a2 = @allocated CSC_RLNK_si_oc_ps_dz_less_parallel2(Ab, nr, s, nt, nt)
	
	@info "time min seq $(minimum(t1)), min par $(minimum(t4)), ratio $(minimum(t1)/minimum(t4))"
	@info "time min seq $(minimum(t2)), min par $(minimum(t5)), ratio $(minimum(t2)/minimum(t5))"
	@info "time min seq $(minimum(t3)), min par $(minimum(t6)), ratio $(minimum(t3)/minimum(t6))"
	@info "alloc seq $a1, par $a2, ration $(a1/a2)"
	
end

function time_pe_p2(flag::Integer, nt, nt2)
	s = sm2l_[flag]
	nr = nrm2l_[flag]
	As = Am2l_[flag]
	
	C2 = CSC_RLNK_si_oc_ps_dz_less_parallel2_timed(As, nr, s, nt, nt2)
end


function test_CSC_for_one_RLNK(nm, nt, depth; num=10)
	grid, nnts, s, nr, cfp, gi, gc = preparatory_multi_ps_less_reverse(nm, nt, depth)
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	
	
	
	
	A0 = da_RLNK_oc_ps_sz_less(cn, nn, nnts, s, cfp, nt, depth)
	C0 = CSC_RLNK_si_oc_ps_dz_less(A0, nr, s, nt)
	
	C1 = copy(C0)
	C1, Ab = csc_assembly_pe_less_new2!(C1, cn, nn, nt, cfp, nnts, depth, s, nr; offset=1)

	@info length(C1.rowval), sum([Ab[i].nnz for i=1:nt])
		
	
	
	#Cb = CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt)
	
	if num > 0
		t1 = zeros(num)
		t2 = zeros(num)
		t3 = zeros(num)
		
		for i=1:num
			C1 = copy(C0)
			t3[i] = @elapsed csc_assembly_pe_less_new2!(C1, cn, nn, nt, cfp, nnts, depth, s, nr; offset=1)
			GC.gc()
		end
		
		
		for i=1:num
			t1[i] = @elapsed CSC_RLNK_plusequals_less3p(Ab, C1, s, nt, gi, nn)
			GC.gc()
		end
		
		for i=1:num
			t2[i] = @elapsed CSC_RLNK_plusequals_less3(Ab, C1, nr, s, nt)
			GC.gc()
		end
			
		C2p = CSC_RLNK_plusequals_less3p(Ab, C1, s, nt, gi, nn)
		C2s = CSC_RLNK_plusequals_less3(Ab, C1, nr, s, nt)
		
		@info "diff: ", length((C2p-C2s).nzval)
		
		@info minimum(t1), minimum(t2), minimum(t3)
	elseif num == 0
		oneRLNK_to_CSC_dz_less_time(Ab[1], nn, s, nt, gi, 1)
		oneRLNK_to_CSC_dz_less_time(Ab[2], nn, s, nt, gi, 2)
		
	
	else
		
		
		CSC_RLNK_plusequals_less3p_time(Ab, C1, s, nt, gi, nn)
		
	
	end
	
	
end
	


### is the backup stuff working correctly?
function test_symmetry(nm, nt, depth; num=10)
	grid, nnts, s, nr, cfp = preparatory_multi_ps_less(nm, nt, depth)
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	
	
	A0 = da_RLNK_oc_ps_sz_less(cn, nn, nnts, s, cfp, nt, depth)
	C0 = CSC_RLNK_si_oc_ps_dz_less(A0, nr, s, nt)
	
	C1, Ab = csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, depth, s, nr; offset=1)
	Cb = CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt)
	
	
	
	t1 = zeros(num)
	t2 = zeros(num)
	t3 = zeros(num)
	for i=1:num
		C0 = CSC_RLNK_si_oc_ps_dz_less(A0, nr, s, nt)
		t1[i] = @elapsed (Cb = CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt))
		t2[i] = @elapsed (Cs = C0 + Cb)
		t3[i] = @elapsed dropzeros!(Cs)
		GC.gc()
	end
	
	t4 = zeros(num)
	t5 = zeros(num)
	t6 = zeros(num)
	for i=1:num
		C0 = CSC_RLNK_si_oc_ps_dz_less(A0, nr, s, nt)
		t4[i] = @elapsed (Cb = CSC_RLNK_si_oc_ps_dz_less_parallel2(Ab, nr, s, nt, nt))
		t5[i] = @elapsed (Cs = C0 + Cb)
		t6[i] = @elapsed dropzeros!(Cs)
		GC.gc()
	end
	
	a1 = @allocated CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt)
	a2 = @allocated CSC_RLNK_si_oc_ps_dz_less_parallel2(Ab, nr, s, nt, nt)
	
	@info "time min seq $(minimum(t1)), min par $(minimum(t4)), ratio $(minimum(t1)/minimum(t4))"
	@info "time min seq $(minimum(t2)), min par $(minimum(t5)), ratio $(minimum(t2)/minimum(t5))"
	@info "time min seq $(minimum(t3)), min par $(minimum(t6)), ratio $(minimum(t3)/minimum(t6))"
	@info "alloc seq $a1, par $a2, ration $(a1/a2)"
	
end


function bm_pe_p1(nm, depth, nt, nt2; num=10)
	@info nm
	grid, nnts, s, nr, cfp = preparatory_multi_ps_less(nm, nt, depth)
	cn = grid[CellNodes]
	nn = num_nodes(grid)
	
	
	@info "calc"
	A0 = da_RLNK_oc_ps_sz_less(cn, nn, nnts, s, cfp, nt, depth)
	C0 = CSC_RLNK_si_oc_ps_dz_less(A0, nr, s, nt)
	
	C1, Ab = csc_assembly_pe_less_new2!(C0, cn, nn, nt, cfp, nnts, depth, s, nr; offset=1)
	#Cb = CSC_RLNK_si_oc_ps_dz_less(Ab, nr, s, nt)
	
	C2 = copy(C1)
	Cs = CSC_RLNK_plusequals_lessu!(Ab, C2, nr, s, nt) # dropzeros!(C1 + Cb)
	C2 = copy(C1)
	Cp = CSC_RLNK_plusequals_lessu_p1!(Ab, C2, nr, s, nt, nt2)
	
	array_ci((Cs-Cp).nzval)
	array_ci(Cs.colptr-Cp.colptr)
	
	
	@info "-----------------------------------------------------"

	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	
	
	
	t1 = zeros(num)
	for i=1:num
		C2 = copy(C1)
		t1[i] = @elapsed CSC_RLNK_plusequals_lessu!(Ab, C2, nr, s, nt)
		GC.gc()
	end
	
	t2 = zeros(num)
	for i=1:num
		C2 = copy(C1)
		t2[i] = @elapsed CSC_RLNK_plusequals_lessu_p1!(Ab, C2, nr, s, nt, nt2)
		GC.gc()
	end
	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	C2 = copy(C1)
	CSC_RLNK_plusequals_lessu_p1_time!(Ab, C2, nr, s, nt, nt2)
	
	@info "-----------------------------------------------------"
	
	C2 = copy(C1)
	Cp = CSC_RLNK_plusequals_lessu_p1!(Ab, C2, nr, s, nt, nt2)
	
	array_ci((Cs-Cp).nzval)
	
	@info "-----------------------------------------------------"
	
	
	println("seq: $(minimum(t1)) par: $(minimum(t2)) seq/par: $(minimum(t1)/minimum(t2))")
	
	
	
	
	
end

function get_CSC(n, m, dens)
	C = sprand(n, m, dens)
	for i=1:length(C.nzval)
		if i%2==0
			C.nzval[i] = 0.0
		end
	end
	C
end	

function bm_dz(n, m, dens, nt; num=10)
	C0 = get_CSC(n, m, dens)
	C1 = copy(C0)
	C2 = copy(C0)
	
	Cs = dropzeros!(C0)
	#Cp = paralleldropzeros!(C1, nt)
	
	paralleldropzeros_time!(C2, nt)
	
	#@info (Cs-Cp).nzval
	
	#array_ci((Cs-Cp).nzval)
	#array_ci(Cs.colptr-Cp.colptr)
	
	#@info Cs.colptr[1:100]
	#@info Cp.colptr[1:100]
	#@info "--------------------------------"
	#@info Cs.colptr[end-100:end]
	#@info Cp.colptr[end-100:end]
	
	t1 = zeros(num)
	for i=1:num
		C0 = get_CSC(n, m, dens)
		t1[i] = @elapsed dropzeros!(C0)
		GC.gc()
	end
	
	t2 = zeros(num)
	for i=1:num
		C0 = get_CSC(n, m, dens)
		#t2[i] = @elapsed paralleldropzeros!(C0, nt)
		paralleldropzeros_time!(C0, nt)
		GC.gc()
	end
	
	@info "--------------------------------------------"
	
	#t2 = zeros(num)
	for i=1:num
		C0 = get_CSC(n, m, dens)
		#t2[i] = @elapsed paralleldropzeros!(C0, nt)
		paralleldropzeros_time2!(C0, nt)
		GC.gc()
	end
	
	@info "--------------------------------------------"
	
	
	@info t1
	#@info t2
	@info minimum(t1) #, minimum(t2)
	
end


### Run things:
### ------------------------------------------------------------------------
#["CP", "Rm1l_3", "Rm1l_u", "Rm2l_3", "Rm2l_u"]
bm_fcts = [[bm_da_CP, bm_da_Rm1, bm_da_Rm2, bm_da_Rm1l, bm_da_Rm2l], [bm_c_CP, bm_c_Rm1, bm_c_Rm2, bm_c_Rm1_dz, bm_c_Rm2_dz, bm_c_Rm1l_dz, bm_c_Rm2l_dz], [bm_ca_CP, bm_ca_Rm1, bm_ca_Rm2, bm_ca_Rm1l, bm_ca_Rm2l], [bm_pe_LNK, bm_pe_Rm1l_3, bm_pe_Rm1l_u, bm_pe_Rm2l_3, bm_pe_Rm2l_u]]
al_fcts = [[al_da_CP, al_da_Rm1, al_da_Rm2, al_da_Rm1l, al_da_Rm2l], [al_c_CP, al_c_Rm1, al_c_Rm2, al_c_Rm1_dz, al_c_Rm2_dz, al_c_Rm1l_dz, al_c_Rm2l_dz], [al_ca_CP, al_ca_Rm1, al_ca_Rm2, al_ca_Rm1l, al_ca_Rm2l], [al_pe_LNK, al_pe_Rm1l_3, al_pe_Rm1l_u, al_pe_Rm2l_3, al_pe_Rm2l_u]]

"""
`function run_auto(wo, ss)`

wo: which one should be benchmarked?
ss: sample size
"""
function run_auto(wo, ss)
	for flag=1:length(nms_)
		str_list = []
		fn = pre_name*"_"*tts(nms_[flag], nt)*"_"*endings[wo]*".txt"
		io = open(fn, "r")
	
		for line in readlines(io)
			push!(str_list, line)
			
		end
		
		fcts = bm_fcts[wo] #[bm_ca_CP, bm_ca_Rm1, bm_ca_Rm2]
		
		close(io)
		
		#fn = pre_name*"_"*tts(nms_[flag])*"_assembly.txt"
		io = open(fn, "w")
	
		for (ctr,str) in enumerate(str_list)
			array = split(str, " ")
			if array[1] == "ss"
				total_ss = parse(Int64, array[2])+ss
				write(io, "ss "*"$(total_ss)\n")
			else
				times = fcts[ctr-1](flag, ss)
				array = vcat(array, numarray_to_strarray(times))
				write(io, strarray_to_string(array)*"\n")
				
			end
		end
		
		
		close(io)
	end
	
end


function run_auto_nowrite(;ss=2)
	for wo=1:3
		for flag=1:length(nms_)
			fcts = bm_fcts[wo]
			for f in fcts
				f(flag, ss)
			end
		end
	end	
end



function run_auto_allocs()
	for flag=1:length(nms_)
		str_list = []
		fn = pre_name*"_"*tts(nms_[flag], nt)*"_allocs.txt"
		io = open(fn, "w")
	
		for (i,ending) in enumerate(endings)
			fcts = al_fcts[i]
			write(io, ending*"\n")
			for (j,f) in enumerate(fcts)
				f(flag)
				write(io, fct_names[i][j]*" $(f(flag))\n")
			end
		end
		
		
		close(io)
	end
	
end

