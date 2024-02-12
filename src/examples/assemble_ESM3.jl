function assemble_part_ESMP_essential3!(matrix, # System matrix
		rhs,
		δ, # heat conduction coefficient
		f::TF, # Source/sink function
		α, # boundary transfer coefficient
		β::TB, # boundary condition function
		grid) where {TF, TB}
		
	trianglelist = grid[CellNodes]
	pointlist    = grid[Coordinates]
	segmentlist  = grid[BEdgeNodes]
	
	ntri   = num_cells(grid) #size(trianglelist,2)
	nbface = num_bedges(grid)
	
	num_nodes_per_cell=3;
	num_edges_per_cell=3;
	num_nodes_per_bface=2
	
	local_edgenodes=[ 2 3; 3 1; 1 2]'
	e, ω, γ = compute_formfactors(grid)
	ni = matrix.new_indices
	nt = matrix.nt
	depth = matrix.depth
	cfp = matrix.cellsforpart
	
	
	t_tri = @elapsed for level=1:depth
		for tid=1:nt
			for itri in cfp[(level-1)*nt+tid]
				for k_local=1:num_nodes_per_cell
					k_global=trianglelist[k_local,itri]
					x=pointlist[1,k_global]
					y=pointlist[2,k_global]
					rhs[ni[k_global]]+=f(x,y)*ω[k_local]
				end
					
				for iedge=1:num_edges_per_cell
					k_global=trianglelist[local_edgenodes[1,iedge],itri]
					l_global=trianglelist[local_edgenodes[2,iedge],itri]
					val = δ*e[iedge, itri]
					ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, -, val, ni[l_global], ni[k_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, -, val, ni[k_global], ni[l_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, +, val, ni[l_global], ni[l_global], tid)
				end
			end
		end
	end
	t_sepa = @elapsed for itri in cfp[(depth*nt+1)]
		for k_local=1:num_nodes_per_cell
			k_global=trianglelist[k_local,itri]
			x=pointlist[1,k_global]
			y=pointlist[2,k_global]
			rhs[ni[k_global]]+=f(x,y)*ω[k_local]
		end
	
		for iedge=1:num_edges_per_cell
			k_global=trianglelist[local_edgenodes[1,iedge],itri]
			l_global=trianglelist[local_edgenodes[2,iedge],itri]
			val = δ*e[iedge, itri]
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, -, val, ni[l_global], ni[k_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, -, val, ni[k_global], ni[l_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[l_global], ni[l_global], 1)
		end
	end
		
	t_bfa = @elapsed for ibface=1:nbface
		bfacefactors!(γ,ibface, pointlist, segmentlist)
		for k_local=1:num_nodes_per_bface
			k_global=segmentlist[k_local,ibface]
			val = α*γ[k_local]
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global])
			x=pointlist[1,k_global]
	        y=pointlist[2,k_global]
			rhs[ni[k_global]]+=β(x,y)*γ[k_local]
	    end
	end
		
    t_tri, t_sepa, t_bfa
	
end

function assemble_part_para_ESMP_essential3!(matrix, # System matrix
		rhs, # right hand side
		δ, # heat conduction coefficient
		f::TF, # Source/sink function
		α, # boundary transfer coefficient
		β::TB, # boundary condition function
		grid) where {TF, TB}
		
	trianglelist = grid[CellNodes]
	pointlist    = grid[Coordinates]
	segmentlist  = grid[BEdgeNodes]
	
	ntri   = num_cells(grid) #size(trianglelist,2)
	nbface = num_bedges(grid)
	
	num_nodes_per_cell=3;
	num_edges_per_cell=3;
	num_nodes_per_bface=2
	local_edgenodes=[ 2 3; 3 1; 1 2]'
	e, ω, γ = compute_formfactors(grid)
	ni = matrix.new_indices
	nt = matrix.nt
	depth = matrix.depth
	cfp = matrix.cellsforpart
	
	
	t_tri = @elapsed for level=1:depth
		@threads :static for tid=1:nt
			for itri in cfp[(level-1)*nt+tid]
				for k_local=1:num_nodes_per_cell
					k_global=trianglelist[k_local,itri]
					x=pointlist[1,k_global]
					y=pointlist[2,k_global]
					rhs[ni[k_global]]+=f(x,y)*ω[k_local]
				end
			
				for iedge=1:num_edges_per_cell
					k_global=trianglelist[local_edgenodes[1,iedge],itri]
					l_global=trianglelist[local_edgenodes[2,iedge],itri]
					val = δ*e[iedge, itri]
					ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, -, val, ni[l_global], ni[k_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, -, val, ni[k_global], ni[l_global], tid)
					ExtendableSparseParallel.updateindex!(matrix, +, val, ni[l_global], ni[l_global], tid)
				end
			end
		end
	end
		
	t_sepa = @elapsed for itri in cfp[(depth*nt+1)]
		for k_local=1:num_nodes_per_cell
			k_global=trianglelist[k_local,itri]
			x=pointlist[1,k_global]
			y=pointlist[2,k_global]
			rhs[ni[k_global]]+=f(x,y)*ω[k_local]
		end
	
		for iedge=1:num_edges_per_cell
			k_global=trianglelist[local_edgenodes[1,iedge],itri]
			l_global=trianglelist[local_edgenodes[2,iedge],itri]
			val = δ*e[iedge, itri]
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, -, val, ni[l_global], ni[k_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, -, val, ni[k_global], ni[l_global], 1)
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[l_global], ni[l_global], 1)
		end
	end
	
	t_bfa = @elapsed for ibface=1:nbface
		bfacefactors!(γ,ibface, pointlist, segmentlist)
		for k_local=1:num_nodes_per_bface
			k_global=segmentlist[k_local,ibface]
			val = α*γ[k_local]
			ExtendableSparseParallel.updateindex!(matrix, +, val, ni[k_global], ni[k_global])
			x=pointlist[1,k_global]
	        y=pointlist[2,k_global]
			rhs[ni[k_global]]+=β(x,y)*γ[k_local]
	    end
	end
		
    t_tri, t_sepa, t_bfa
end



function assemble!(matrix, # System matrix
		rhs, # Right hand side vector
		δ, # heat conduction coefficient
		f::TF, # Source/sink function
		α, # boundary transfer coefficient
		β::TB, # boundary condition function
		grid) where {TF, TB}
	
	trianglelist = grid[CellNodes]
	pointlist    = grid[Coordinates]
	segmentlist  = grid[BEdgeNodes]
		
	num_nodes_per_cell=3;
	num_edges_per_cell=3;
	num_nodes_per_bface=2
	ntri   = num_cells(grid) #size(trianglelist,2)
	nbface = num_bedges(grid) #size(segmentlist,2)
	# Local edge-node connectivity
	local_edgenodes=[ 2 3; 3 1; 1 2]'
	# Storage for form factors
	e=zeros(num_nodes_per_cell)
	ω=zeros(num_edges_per_cell)
	γ=zeros(num_nodes_per_bface)
	# Initialize right hand side to zero
	rhs.=0.0
	# Loop over all triangles
	t_tri = 0.0
	t_bfa = 0.0
	for itri=1:ntri
		trifactors!(ω,e,itri,pointlist,trianglelist)
		t_tri += @elapsed begin
			# Assemble nodal contributions to right hand side
			for k_local=1:num_nodes_per_cell
				k_global=trianglelist[k_local,itri]
				x=pointlist[1,k_global]
				y=pointlist[2,k_global]
				rhs[k_global]+=f(x,y)*ω[k_local]
			end
			# Assemble edge contributions to matrix
			for iedge=1:num_edges_per_cell
				k_global=trianglelist[local_edgenodes[1,iedge],itri]
				l_global=trianglelist[local_edgenodes[2,iedge],itri]
				matrix[k_global,k_global]+=δ*e[iedge]
				matrix[l_global,k_global]-=δ*e[iedge]
				matrix[k_global,l_global]-=δ*e[iedge]
				matrix[l_global,l_global]+=δ*e[iedge]
			end
		end
	end
	# Assemble boundary conditions
	for ibface=1:nbface
		bfacefactors!(γ,ibface, pointlist, segmentlist)
		t_bfa += @elapsed for k_local=1:num_nodes_per_bface
			k_global=segmentlist[k_local,ibface]
			matrix[k_global,k_global]+=α*γ[k_local]
			x=pointlist[1,k_global]
            y=pointlist[2,k_global]
			rhs[k_global]+=β(x,y)*γ[k_local]
        end
    end
    t_tri, t_bfa
end










