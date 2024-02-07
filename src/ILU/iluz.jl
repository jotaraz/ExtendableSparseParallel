using LinearAlgebra
using SparseArrays
using ILUZero


function total_ILUZ(cscmatrix::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv}) where {Tv, Ti <: Integer}
	x = copy(b)
	#ilu = ILUZero.ILU0Precon(cscmatrix, Ti)
	#ilu = ILUZero.ilu0!(ilu, cscmatrix)
	ilu = ilu0(cscmatrix)
	
	#@info typeof(ilu)
	
	ILUZero.ldiv!(x, ilu, b)
end


function create_diag_csc(Ti, n; Tv=Float64)
	colptr = Vector{Ti}(undef, n+1)
	rowval = Vector{Ti}(undef, n)
	nzval  = Vector{Tv}(undef, n)
	
	for i=1:n
		colptr[i] = Ti(i)
		rowval[i] = Ti(i)
		nzval[i] = Tv(1.0/i)
	end
	colptr[n+1] = Ti(n+1)
	
	SparseArrays.SparseMatrixCSC(Ti(n), Ti(n), colptr, rowval, nzval)
end
	
	

function ILUZero.ILU0Precon(A::SparseMatrixCSC{T,N}, b_type = T) where {T <: Any,N <: Integer}
    m, n = size(A)
    #@info "yes"

    # Determine number of elements in lower/upper
    lnz = 0
    unz = 0
    @inbounds for i = 1:n
        for j = A.colptr[i]:A.colptr[i + 1] - 1
            if A.rowval[j] > i
                lnz += 1
            else
                unz += 1
            end
        end
    end

    # Preallocate variables
    l_colptr = zeros(N, n + 1)
    u_colptr = zeros(N, n + 1)
    l_nzval = zeros(T, lnz)
    u_nzval = zeros(T, unz)
    l_rowval = zeros(N, lnz)
    u_rowval = zeros(N, unz)
    l_map = Vector{N}(undef, lnz)
    u_map = Vector{N}(undef, unz)
    wrk = zeros(b_type, n)
    l_colptr[1] = 1
    u_colptr[1] = 1

    # Map elements of A to lower and upper triangles, fill out colptr, and fill out rowval
    lit = 1
    uit = 1
    @inbounds for i = 1:n
        l_colptr[i + 1] = l_colptr[i]
        u_colptr[i + 1] = u_colptr[i]
        for j = A.colptr[i]:A.colptr[i + 1] - 1
            if A.rowval[j] > i
                l_colptr[i + 1] += 1
                l_rowval[lit] = A.rowval[j]
                l_map[lit] = j
                lit += 1
            else
                u_colptr[i + 1] += 1
                u_rowval[uit] = A.rowval[j]
                u_map[uit] = j
                uit += 1
            end
        end
    end
	#@info "almost"
    return ILU0Precon(N(m), N(n), l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map, wrk)
end
