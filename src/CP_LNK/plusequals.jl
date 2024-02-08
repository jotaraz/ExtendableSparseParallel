"""
`function add_LNKs!(A,B)`

A += B for LNK matrices.
"""
function add_LNKs!(A,B) #A += B
	n = B.n #number of columns #size(B)[2]
	val = B.nzval
	col = B.colptr
	row = B.rowval

	j = 1
	id = 1
	ctr = 0
	while j <= n
		if row[id] > 0
			A[row[id], j] += val[id]
		end

		if col[id] == 0
			j += 1
			id = j
		else
			id = col[id]
		end
		if j > n
			break
		end
	end
end

function CSC_LNK_plusequals!(
	As::Vector{SparseMatrixLNK{Tv, Ti}}, 
	C::SparseArrays.SparseMatrixCSC{Tv, Ti}, nt) where {Tv, Ti <: Integer}

	for tid=2:nt
		add_LNKs!(As[1], As[tid])
	end
	dropzeros!(C + As[1])
	
end



