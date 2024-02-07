#include(path*"preparatory.jl")

"""
`function partitionbetter(x, n)`

Similar to Base.Iterators.partition(x, m) but with a different notation and it outputs a vector of vectors. It always outputs n vectors. The first always have the same length and the last one can have a different length.
Example:
partitionbetter(collect(1:25), 3) returns 
[[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16], [17,18,19,20,21,22,23,24,25]]
"""
function partitionbetter(x, n)
	baselength = Int(round(length(x)/n))
	residuallength = length(x) - (n-1)*baselength
	lengths = ones(Int32, n)*baselength
	lengths[end] = residuallength
	Ty = typeof(x[1])
	ret = [Vector{Ty}(undef, l) for l in lengths] #[zeros(typeof(x[1]), l) 
	for i=1:n
		ret[i] = view(x, (i-1)*baselength+1:(i-1)*baselength+lengths[i])
	end
	ret	
end

function partitionbetter2(k, n)
	baselength = Int(round(k/n))
	residuallength = k - (n-1)*baselength
	lengths = ones(Int32, n)*baselength
	lengths[end] = residuallength
	ret = [collect((i-1)*baselength+1:(i-1)*baselength+lengths[i]) for i=1:n] #Vector{Ty}(undef, l) for l in lengths] #[zeros(typeof(x[1]), l) 
	#for i=1:n
	#	ret[i] = view(x, (i-1)*baselength+1:(i-1)*baselength+lengths[i])
	#end
	ret	
end

function partitionbetter2p(x, n)
	baselength = Int(round(length(x)/n))
	residuallength = length(x) - (n-1)*baselength
	lengths = ones(Int32, n)*baselength
	lengths[end] = residuallength
	Ty = typeof(x[1])
	ret = [Vector{Ty}(undef, l) for l in lengths] #[zeros(typeof(x[1]), l) 
	@threads :static for i=1:n
		ret[i] = collect((i-1)*baselength+1:(i-1)*baselength+lengths[i])
	end
	ret	
end




"""
`function CSC_LNK_si(lnk::SparseMatrixLNK{Tv, Ti}) where {Tv, Ti <: Integer}`

Turns the LNK matrix `lnk` into a CSC matrix.
Stolen from `https://github.com/j-fu/ExtendableSparse.jl/blob/master/src/matrix/sparsematrixlnk.jl`, `https://github.com/j-fu/ExtendableSparse.jl/blob/master/src/matrix/sparsematrixlnk.jl`.. 
Only modification is, there is no CSC matrix on which the LNK matrix added, this is a speed up.
"""
function CSC_LNK_si(lnk::SparseMatrixLNK{Tv, Ti}) where {Tv, Ti <: Integer}
    csc = spzeros(lnk.m, lnk.n)
    # overallocate arrays in order to avoid
    # presumably slower push!
    xnnz = nnz(lnk)
    colptr = Vector{Ti}(undef, csc.n + 1)
    rowval = Vector{Ti}(undef, xnnz)
    nzval = Vector{Tv}(undef, xnnz)

	l_lnk_col = 0
        
    inz = 1 # counts the nonzero entries in the new matrix
	for j = 1:(csc.n)
		jc = 0
        # Copy extension entries into col and sort them
        k = j
        colptr[j] = inz
        while k > 0
            if lnk.rowval[k] > 0
                l_lnk_col += 1
				rowval[l_lnk_col] = lnk.rowval[k]
				nzval[l_lnk_col]  = lnk.nzval[k]
				
                #col[l_lnk_col] = ColEntry(lnk.rowval[k], lnk.nzval[k])

				for jcc=1:jc
					if rowval[l_lnk_col-jcc] > rowval[l_lnk_col-jcc+1]
						tmp_r = rowval[l_lnk_col-jcc+1]
						tmp_v = nzval[l_lnk_col-jcc+1]
						rowval[l_lnk_col-jcc+1] = rowval[l_lnk_col-jcc]
						nzval[l_lnk_col-jcc+1]  = nzval[l_lnk_col-jcc]
						rowval[l_lnk_col-jcc] = tmp_r
						nzval[l_lnk_col-jcc]  = tmp_v
					else
						break
					end
						
				end
				inz += 1
				jc += 1
            end
            k = lnk.colptr[k]
        end
    end
    colptr[csc.n + 1] = inz
    # Julia 1.7 wants this correct
    resize!(rowval, inz - 1)
    resize!(nzval, inz - 1)
    SparseArrays.SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)
end


"""
`function CSC_LNKs_s!(As::Vector{SparseMatrixLNK{Tv, Ti}}) where {Tv, Ti <: Integer}`

Creates a CSC matrix that contains the sum of the LNK matrices that are entered as elements of `As`.
In other words:
`As = [A1, A2, A3, A4]; A1 += A2; A1 += A2; A1 += A3; A1 += A4; return CSC(A1)`

"""
function CSC_LNKs_s!(As::Vector{SparseMatrixLNK{Tv, Ti}}) where {Tv, Ti <: Integer}
	A = CSC_LNK_si(As[1])
	for i=2:length(As)
		A += CSC_LNK_si(As[i])
	end
	A
end

function CSC_LNKs_s_dz!(As::Vector{SparseMatrixLNK{Tv, Ti}}) where {Tv, Ti <: Integer}
	A = CSC_LNK_si(As[1])
	for i=2:length(As)
		A += CSC_LNK_si(As[i])
	end
	dropzeros!(A)
end

