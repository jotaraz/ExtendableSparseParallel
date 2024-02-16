function test1!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
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
		for tid=1:nt
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end

	level = depth+1
	for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end

end
