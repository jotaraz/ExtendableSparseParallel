module ExtendableSparseParallel

using Metis
using Base.Threads
using ExtendableGrids
include("supersparse.jl")
include("preparatory.jl")
#include("prep_time.jl")

mutable struct ExtendableSparseMatrixParallel{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Linked list structure holding data of extension
    """
    lnkmatrices::Vector{SuperSparseMatrixLNK{Tv, Ti}}

	grid::ExtendableGrid

	nnts::Vector{Ti}
    
    sortednodesperthread::Matrix{Ti}
    
    old_noderegions::Matrix{Ti}
    
    cellsforpart::Vector{Vector{Ti}}
    
    globalindices::Vector{Vector{Ti}}
    
    new_indices::Vector{Ti}
    
    rev_new_indices::Vector{Ti}
    
    start::Vector{Ti}
    
    nt::Ti
    
    depth::Ti
    
end



function ExtendableSparseMatrixParallel{Tv, Ti}(nm, nt, depth) where {Tv, Ti <: Integer}
	grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts = preparatory_multi_ps_less_reverse(nm, nt, depth)
	csc = spzeros(Tv, Ti, num_nodes(grid), num_nodes(grid))
	lnk = [SuperSparseMatrixLNK{Float64, Int32}(num_nodes(grid), nnts[tid]) for tid=1:nt]
	ExtendableSparseMatrixParallel{Tv, Ti}(csc, lnk, grid, nnts, s, onr, cfp, gi, ni, rni, starts, nt, depth)
end



function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, tid, v; known_that_unknown=true) where {Tv, Ti <: Integer}
	if known_that_unknown
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
		return
	end
	
	if updatentryCSC2!(A.cscmatrix, i, j, v)
	else
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
	end
end

function reset!(A::ExtendableSparseMatrixParallel)
	A.cscmatrix = spzeros(Float64, Int32, num_nodes(A.grid), num_nodes(A.grid))
	A.lnkmatrices = [SuperSparseMatrixLNK{Float64, Int32}(num_nodes(A.grid), A.nnts[tid]) for tid=1:A.nt]
end

function nnz_flush(ext::ExtendableSparseMatrixParallel)
    flush!(ext)
    return nnz(ext.cscmatrix)
end

function nnz_noflush(ext::ExtendableSparseMatrixParallel)
    return nnz(ext.cscmatrix), sum([ext.lnkmatrices[i].nnz for i=1:ext.nt])
end
	

function Base.show(io::IO, ::MIME"text/plain", ext::ExtendableSparseMatrixParallel)
    #flush!(ext)
    xnnzCSC, xnnzLNK = nnz_noflush(ext)
    m, n = size(ext)
    print(io,
          m,
          "×",
          n,
          " ",
          typeof(ext),
          " with ",
          xnnzCSC,
          " stored ",
          xnnzCSC == 1 ? "entry" : "entries",
          " in CSC and ",
          xnnzLNK,
          " stored ",
          xnnzLNK == 1 ? "entry" : "entries",
          " in LNK. CSC:")

    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    if !(m == 0 || n == 0 || xnnzCSC == 0)
        print(io, ":\n")
        Base.print_array(IOContext(io), ext.cscmatrix)
    end
end

Base.size(A::ExtendableSparseMatrixParallel) = (A.cscmatrix.m, A.cscmatrix.n)


include("struct_assembly.jl")
include("struct_assembly_dyn.jl")
include("struct_flush.jl")


export ExtendableSparseMatrixParallel, SuperSparseMatrixLNK
export addtoentry!, reset!, dummy_assembly!, preparatory_multi_ps_less_reverse, fr


end

