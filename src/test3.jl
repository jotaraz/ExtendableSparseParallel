function test3!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	# if its not known that all values are unknown, the csc matrix is used
	# thus it has to be cleared before
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	
	for level=1:depth
		@threads for tid=1:nt
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					_addnz(A, ninode, ninode, tid, diagval)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						_addnz(A, ninode, njnode, tid, v)
						_addnz(A, njnode, ninode, tid, v+dv)
					end
				end
			end
		end
	end

	level = depth+1
	@threads for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				_addnz(A, ninode, ninode, tid, diagval)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					_addnz(A, ninode, njnode, tid, v)
					_addnz(A, njnode, ninode, tid, v+dv)
				end
			end
		end
	end

end
     
function _addnz(matrix::ExtendableSparseMatrixParallel, i, j, tid, v::Tv) where {Tv}
    if isnan(v)
        error("trying to assemble NaN, i:", i, ", j: ", j, "v: ", v, "fac: ", 1)
    end
    if v != zero(Tv)
        ExtendableSparseParallel.rawupdateindex!(matrix, +, v, i, j, tid)
    end
end
