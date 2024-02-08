"""
`function fr(x)`

fr stands for fake random and it is supposed to imitate a random number generator, but reproducible.
It just returns `(x%10)-1`. Used in matrix assembly to have different nonzero entries in the original and the 'new' CSC matrix.
"""
function fr(x)
	((x%10_000)-1)/10_000
end	


"""
`function entryexists2(CSC, i, j)`

Find out if CSC already has an nonzero entry at i,j without any allocations
"""
function entryexists2(CSC, i, j) #find out if CSC already has an nonzero entry at i,j
	#vals = 
	#ids = CSC.colptr[j]:(CSC.colptr[j+1]-1)
	i in view(CSC.rowval, CSC.colptr[j]:(CSC.colptr[j+1]-1))
end


function updatentryCSC2!(CSC::SparseArrays.SparseMatrixCSC{Tv, Ti}, i::Integer, j::Integer, v) where {Tv, Ti <: Integer}
	p1 = CSC.colptr[j]
	p2 = CSC.colptr[j+1]-1

	searchk = searchsortedfirst(view(CSC.rowval, p1:p2), i) + p1 - 1
	
	if (searchk <= p2) && (CSC.rowval[searchk] == i)
		CSC.nzval[searchk] += v
		return true
	else
		return false
	end
end

function dummy_assembly!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false, extra_fct=true, dynamic=true) where {Tv, Ti <: Integer}
	
	if known_that_unknown && extra_fct
		cn = A.grid[CellNodes]
		nn = num_nodes(A.grid)
		nnts = A.nnts
		s = A.sortednodesperthread
		cfp = A.cellsforpart
		nt = A.nt
		depth = A.depth
		gi = A.globalindices
		ni = A.new_indices
		if dynamic
			A.lnkmatrices = da_RLNK_oc_ps_sz_less_reordered_super_spawn(cn, nn, nnts, s, cfp, nt, depth, gi, ni, Tv, Ti; offset=offset, diagval=diagval, symm=symm, skew=skew)
		else
			A.lnkmatrices = da_RLNK_oc_ps_sz_less_reordered_super(cn, nn, nnts, s, cfp, nt, depth, gi, ni; offset=offset, diagval=diagval, symm=symm, skew=skew)
		end
	else
		if dynamic
			dummy_assembly_do_spawn!(A::ExtendableSparseMatrixParallel; offset=offset, diagval=diagval, symm=symm, skew=skew, known_that_unknown=known_that_unknown)
		else
			dummy_assembly_do!(A::ExtendableSparseMatrixParallel; offset=offset, diagval=diagval, symm=symm, skew=skew, known_that_unknown=known_that_unknown)
		end
	end

end



function da_RLNK_oc_ps_sz_less_reordered_super_spawn(cellnodes, nn, nnts, s, cellsforpart, nt, depth, gi, ni, Tv, Ti; offset=0, diagval=5.0, symm=0.5, skew=0.25)
	K = size(cellnodes)[1]
	As = [SuperSparseMatrixLNK{Tv, Ti}(nn, nnts[tid]) for tid=1:nt]

	for level=1:depth
		@threads for tid=1:nt
			for icell in cellsforpart[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					As[tid][ninode, s[tid,ninode]] += diagval
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew				
						As[tid][ninode, s[tid,njnode]] += v
						As[tid][njnode, s[tid,ninode]] += v+dv
					end
				end
			end
		end
	end

	for icell in cellsforpart[depth*nt+1] 
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i]
			ninode = ni[inode]
			As[1][ninode, s[1,ninode]] += diagval
			for j=i+1:K
				jnode = tmp[j]
				njnode = ni[jnode]
				v = fr(inode+jnode+offset)*symm
				dv = fr(inode+jnode+offset+1)*skew
				As[1][ninode, s[1,njnode]] += v
				As[1][njnode, s[1,ninode]] += v+dv
			end
		end
	end
	
	
 	As
end



function dummy_assembly_do_spawn!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
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
	@threads for tid=1:1
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
