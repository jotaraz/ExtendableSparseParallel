#=
function dummy_assembly!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false, extra_fct=true) where {Tv, Ti <: Integer}
	
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
		A.lnkmatrices = da_RLNK_oc_ps_sz_less_reordered_super(cn, nn, nnts, s, cfp, nt, depth, gi, ni; offset=offset, diagval=diagval, symm=symm, skew=skew)
	else
		dummy_assembly_do!(A::ExtendableSparseMatrixParallel; offset=offset, diagval=diagval, symm=symm, skew=skew, known_that_unknown=known_that_unknown)
	end

end
=#

function dummy_assembly_do!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
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
		@threads :static for tid=1:nt
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
	@threads :static for tid=1:1
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

	#=
	for icell in cfp[depth*nt+1] 
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i]
			ninode = ni[inode]
			addtoentry!(A, ninode, ninode, 1, diagval; known_that_unknown=known_that_unknown)					
			for j=i+1:K
				jnode = tmp[j]
				njnode = ni[jnode]
				v = fr(inode+jnode+offset)*symm
				dv = fr(inode+jnode+offset+1)*skew
				
				addtoentry!(A, ninode, njnode, 1, v; known_that_unknown=known_that_unknown)
				addtoentry!(A, njnode, ninode, 1, v+dv; known_that_unknown=known_that_unknown)
			end
		end
	end
	=#
	#flush!(A)

end


function dummy_assembly_time!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	time = zeros(3)
	time[1] = @elapsed begin
		cellnodes = A.grid[CellNodes]
		K = size(cellnodes)[1]
		nt = A.nt
		depth = A.depth
		ni = A.new_indices
		cfp = A.cellsforpart
		
		# if its not known that all values are unknown, the csc matrix is used
		# thus it has to be cleared before
		if !known_that_unknown
			A.cscmatrix.nzval .= 0
		end
	end
	
	
	
	
	
	time[2] = @elapsed for level=1:depth
		@threads :static for tid=1:nt
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
	
	level=depth+1
	time[3] = @elapsed @threads :static for tid=1:1
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


	#=
	time[3] = @elapsed for icell in cfp[depth*nt+1] 
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i]
			ninode = ni[inode]
			addtoentry!(A, ninode, ninode, 1, diagval; known_that_unknown=known_that_unknown)					
			for j=i+1:K
				jnode = tmp[j]
				njnode = ni[jnode]
				v = fr(inode+jnode+offset)*symm
				dv = fr(inode+jnode+offset+1)*skew
				
				addtoentry!(A, ninode, njnode, 1, v; known_that_unknown=known_that_unknown)
				addtoentry!(A, njnode, ninode, 1, v+dv; known_that_unknown=known_that_unknown)
			end
		end
	end
	=#
	
	
	
	#flush!(A)
	@info time

end

function dummy_assembly_time_altsepa!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	time = zeros(3)
	time[1] = @elapsed begin
		cellnodes = A.grid[CellNodes]
		K = size(cellnodes)[1]
		nt = A.nt
		depth = A.depth
		ni = A.new_indices
		cfp = A.cellsforpart
		s = A.sortednodesperthread
		
		# if its not known that all values are unknown, the csc matrix is used
		# thus it has to be cleared before
		if !known_that_unknown
			A.cscmatrix.nzval .= 0
		end
	end
	
	
	
	
	
	time[2] = @elapsed for level=1:depth
		@threads :static for tid=1:nt
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

	#csc = A.cscmatrix
	#lnk = A.lnkmatrices[1]

	level = depth+1
	time[3] = @elapsed @threads :static for tid=1:1
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

	#=
	time[3] = @elapsed for icell in cfp[depth*nt+1] 
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i]
			ninode = ni[inode]
			v = diagval
			if updatentryCSC2!(csc, ninode, ninode, v)
			else
				lnk[ninode, s[1, ninode]] += v
			end
			#addtoentry!(A, ninode, ninode, 1, diagval; known_that_unknown=known_that_unknown)					
			for j=i+1:K
				jnode = tmp[j]
				njnode = ni[jnode]
				v = fr(inode+jnode+offset)*symm
				dv = fr(inode+jnode+offset+1)*skew
				
				#addtoentry!(A, ninode, njnode, 1, v; known_that_unknown=known_that_unknown)
				#addtoentry!(A, njnode, ninode, 1, v+dv; known_that_unknown=known_that_unknown)
				if updatentryCSC2!(csc, ninode, njnode, v)
				else
					lnk[ninode, s[1, njnode]] += v
				end
				if updatentryCSC2!(csc, njnode, ninode, v+dv)
				else
					lnk[njnode, s[1, ninode]] += v+dv
				end
			end
		end
	end
	=#
	
	
	#flush!(A)
	@info time

end





function da_RLNK_oc_ps_sz_less_reordered_super(cellnodes, nn, nnts, s, cellsforpart, nt, depth, gi, ni, Tv, Ti; offset=0, diagval=5.0, symm=0.5, skew=0.25)
	K = size(cellnodes)[1]
	As = [SuperSparseMatrixLNK{Tv, Ti}(nn, nnts[tid]) for tid=1:nt]

	for level=1:depth
		@threads :static for tid=1:nt
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

