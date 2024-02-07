#include(path*"assembly.jl")


"""
`function entryexists2(CSC, i, j)`

Find out if CSC already has an nonzero entry at i,j without any allocations
"""
function entryexists2(CSC, i, j) #find out if CSC already has an nonzero entry at i,j
	#vals = 
	#ids = CSC.colptr[j]:(CSC.colptr[j+1]-1)
	i in view(CSC.rowval, CSC.colptr[j]:(CSC.colptr[j+1]-1))
end


function updatentryCSC2!(CSC::SparseArrays.SparseMatrixCSC{Float64, Int32}, i::Integer, j::Integer, v::Float64)
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





"""
`function da_csc_LNK_cp_reordered!(C, grid; offset=0)`

Dummy assembly in the CSC matrix cheaply parallelized.
THIS IS UNSAFE (can lead to wrong results) AND ONLY DONE FOR TESTING.
Different threads concurrently write into the same matrix entry.
"""
function da_csc_LNK_cp_reordered!(C, cellnodes, cfp, grid, ni, nt; offset=0)
	A_backup = [SparseMatrixLNK{Float64, Int32}(num_nodes(grid), num_nodes(grid)) for i=1:nt]
	#nc_nt = Int(num_cells(grid)/nt)
	#cfp = [collect((tid-1)*nc_nt+1:tid*nc_nt) for tid=1:nt]
	K = size(cellnodes)[1]
	
	C.nzval .= 0
	
	@threads :static for tid=1:nt
		for icell in cfp[tid] 
			tmp = view(cellnodes, :, icell)
			#for (i,inode) in enumerate(grid[CellNodes][:,icell])
			for i=1:K
				inode = tmp[i] #grid[CellNodes][i,icell]
				ninode = ni[inode]
				v = 3.0
			
				if updatentryCSC2!(C, ninode, ninode, v)
				else
					A_backup[tid][ninode,ninode] += v
				end
				
				#for jnode in grid[CellNodes][i+1:end,icell]
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*0.5	
					dv = fr(inode+jnode+offset+1)*0.25	
					if updatentryCSC2!(C, ninode, njnode, v)
					else
						A_backup[tid][ninode,njnode] += v
					end
					if updatentryCSC2!(C, njnode, ninode, v+dv)
					else
						A_backup[tid][njnode,ninode] += v+dv
					end
					
				end
				
				
			end
		end
	end

	
	C, A_backup
	
end


function da_csc_LNK_seq_reordered!(C::ExtendableSparseMatrix, cellnodes, grid, ni; offset=0)
	K, nc = size(cellnodes)
	
	C.cscmatrix.nzval .= 0
	
	for icell=1:nc
		tmp = view(cellnodes, :, icell)
		for i=1:K
			inode = tmp[i] #grid[CellNodes][i,icell]
			ninode = ni[inode]
			v = 3.0
			C[ninode,ninode] += v
			
			for j=i+1:K
				jnode = tmp[j]
				njnode = ni[jnode]
				v = fr(inode+jnode+offset)*0.5	
				dv = fr(inode+jnode+offset+1)*0.25	
				C[ninode,njnode] += v
				C[njnode,ninode] += v+dv
			end
		end
	end

	
	C	
end




#=

"""
`function fill_dummy_zeros!(RLNKs)`

Fills the first row of the matrices with zeros, if this is not done there might be problems with CSC conversion.
`RLNKs` is a vector of LNK matrices of reduced size.
"""
function fill_dummy_zeros!(RLNKs)
	@threads :static for tid=1:length(RLNKs) #depth*nt+1
		for j=1:RLNKs[tid].n
			RLNKs[tid][1,j] = 1.0
			RLNKs[tid][1,j] -= 1.0
		end
	end
end
=#




