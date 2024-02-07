
"""
`function da_LNK_cp_sz_reordered(grid::ExtendableGrid, nt::Integer)`

Dummy assembly (da) of an LNK using 'using cheap parallelization' with some zeros (since the off-diagonal entries have the sign -1, 0 or 1).
"""
function da_LNK_cp_sz_reordered(cellnodes, nn, cfp, nt::Integer, ni)
	K = size(cellnodes)[1]
	#nc_nt = Int(num_cells(grid)/nt)
	#cfp = [collect((tid-1)*nc_nt+1:tid*nc_nt) for tid=1:nt]
	As = [SparseMatrixLNK{Float64, Int32}(nn, nn) for tid=1:nt]
	
	@threads :static for tid=1:nt
		for icell in cfp[tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i] #grid[CellNodes][i,icell]
				ninode = ni[inode]
				As[tid][ninode, ninode] += 5.0
				for j=i+1:K
					jnode = tmp[j] #grid[CellNodes][j,icell]
					njnode = ni[jnode]
					v = fr(inode+jnode)*0.5	
					As[tid][ninode, njnode] += v #(-0.5)
					As[tid][njnode, ninode] += v #(-0.5)
				end
			end
		end
	end

	As
end

function da_LNK_sz_reordered(cellnodes, nn, ni)
	K, nc = size(cellnodes)	
	A = ExtendableSparseMatrix{Float64, Int32}(nn, nn)
	
	for icell=1:nc
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i] #grid[CellNodes][i,icell]
			ninode = ni[inode]
			A[ninode, ninode] += 5.0
			for j=i+1:K
				jnode = tmp[j] #grid[CellNodes][j,icell]
				njnode = ni[jnode]
				v = fr(inode+jnode)*0.5	
				A[ninode, njnode] += v #(-0.5)
				A[njnode, ninode] += v #(-0.5)
			end
		end
	end
	
	A
end

































