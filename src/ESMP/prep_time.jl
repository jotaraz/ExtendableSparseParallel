function preparatory_multi_ps_less_reverse_time(nm, nt, depth)
	time = zeros(4)
	time[1] = @elapsed (grid = getgrid(nm))
	
	time[2] = @elapsed ((allcells, start) = grid_to_graph_ps_multi_time!(grid, nt, depth))
	
	time[3] = @elapsed ((nnts, s, onr, gi, gc, ni, rni, starts) = get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse(
		grid[CellRegions], allcells, start, num_nodes(grid), Int32, nt
	))
	
	time[4] = @elapsed (cfp = bettercellsforpart(grid[CellRegions], depth*nt+1))
	
	@info time
	return grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts
end

function preparatory_multi_ps_less_reverse_par_nopush_time(nm, nt, depth)
	time = zeros(4)
	time[1] = @elapsed (grid = getgrid(nm))
	
	time[2] = @elapsed ((allcells, start) = grid_to_graph_ps_multi_par_time!(grid, nt, depth))
	
	time[3] = @elapsed ((nnts, s, onr, gi, gc, ni, rni, starts) = get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse_time(
		grid[CellRegions], allcells, start, num_nodes(grid), Int32, nt
	))
	
	time[4] = @elapsed (cfp = bettercellsforpart(grid[CellRegions], depth*nt+1))
	
	@info time
	return grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts
end

function preparatory_multi_ps_less_reverse_par_time(nm, nt, depth)
	time = zeros(4)
	time[1] = @elapsed (grid = getgrid(nm))
	
	time[2] = @elapsed ((allcells, start) = grid_to_graph_ps_multi_par_time!(grid, nt, depth))
	
	time[3] = @elapsed ((nnts, s, onr, gi, gc, ni, rni, starts) = get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse(
		grid[CellRegions], allcells, start, num_nodes(grid), Int32, nt
	))
	
	time[4] = @elapsed (cfp = bettercellsforpart(grid[CellRegions], depth*nt+1))
	
	@info time
	return grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts
end


function grid_to_graph_ps_multi_time!(grid, nt, depth)
	time = zeros(12)
	time[1] = @elapsed (A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid)))
	time[2] = @elapsed (number_cells_per_node = zeros(Int64, num_nodes(grid)))
	
	time[3] = @elapsed for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			number_cells_per_node[node_id] += 1
		end
	end
		
	time[4] = @elapsed begin
		allcells = zeros(Int64, sum(number_cells_per_node))
		start = ones(Int64, num_nodes(grid)+1)
		start[2:end] += cumsum(number_cells_per_node)
		number_cells_per_node .= 0
	end
	time[5] = @elapsed for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	time[6] = @elapsed for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		for (i,id1) in enumerate(cells)
			for id2 in cells[i+1:end]
				A[id1,id2] = 1
				A[id2,id1] = 1
			end
		end	
	end
	
	time[7] = @elapsed (ACSC = SparseArrays.SparseMatrixCSC(A))
	
	
	time[8] = @elapsed (partition = Metis.partition(ACSC, nt))
	time[9] = @elapsed (cellregs  = copy(partition))
	
	ctr_sepanodes = 0
	time[10] = @elapsed for j=1:num_cells(grid)
		rows = ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
		if minimum(partition[rows]) != maximum(partition[rows])
			cellregs[j] = nt+1
			ctr_sepanodes += 1
		end
	end
	RART = ACSC
	time[11] = @elapsed for level=1:depth-1
		RART, ctr_sepanodes = separate!(cellregs, num_cells(grid), RART, nt, level, ctr_sepanodes)
	end

			
	time[12] = @elapsed (grid[CellRegions] = cellregs)
	
	@info time
	
	return allcells, start
end


function grid_to_graph_ps_multi_par_time!(grid, nt, depth)
	time = zeros(12)
	time[1] = @elapsed (As = [ExtendableSparseMatrix{Int64, Int64}(num_cells(grid), num_cells(grid)) for tid=1:nt])
		#A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid)))
	time[2] = @elapsed (number_cells_per_node = zeros(Int64, num_nodes(grid)))
	
	cn = grid[CellNodes]
	
	time[3] = @elapsed for j=1:num_cells(grid)
		tmp = view(cn, :, j)
		for node_id in tmp
			number_cells_per_node[node_id] += 1
		end
	end
		
	time[4] = @elapsed begin
		allcells = zeros(Int64, sum(number_cells_per_node))
		start = ones(Int64, num_nodes(grid)+1)
		start[2:end] += cumsum(number_cells_per_node)
		number_cells_per_node .= 0
	end
	time[5] = @elapsed for j=1:num_cells(grid)
		tmp = view(cn, :, j)
		for node_id in tmp
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	node_range = get_starts(num_nodes(grid), nt)
	time[6] = @elapsed Threads.@threads :static for tid=1:nt
		for j in node_range[tid]:node_range[tid+1]-1
			cells = @view allcells[start[j]:start[j+1]-1]
			l = length(cells)
			for (i,id1) in enumerate(cells)
				ce = view(cells, i+1:l)
				for id2 in ce
					As[tid][id1,id2] = 1
					As[tid][id2,id1] = 1
				end
			end	
		end
		flush!(As[tid])
	end
	
	time[7] = @elapsed (ACSC = add_all_par!(As).cscmatrix)
		
	#SparseArrays.SparseMatrixCSC(A))
	
	
	time[8] = @elapsed (partition = Metis.partition(ACSC, nt))
	time[9] = @elapsed (cellregs  = copy(partition))
	
	ctr_sepanodes_a = zeros(Int64, nt)
	
	cell_range = get_starts(num_cells(grid), nt)
	time[10] = @elapsed Threads.@threads :static for tid=1:nt
		for j in cell_range[tid]:cell_range[tid+1]-1
			rows = @view ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
			if minimum(partition[rows]) != maximum(partition[rows])
				cellregs[j] = nt+1
				ctr_sepanodes_a[tid] += 1
			end
		end
	end
	
	ctr_sepanodes = sum(ctr_sepanodes_a)
			
	#=
	time[10] = @elapsed for j=1:num_cells(grid)
		rows = ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
		if minimum(partition[rows]) != maximum(partition[rows])
			cellregs[j] = nt+1
			ctr_sepanodes += 1
		end
	end
	=#
	RART = ACSC
	time[11] = @elapsed for level=1:depth-1
		RART, ctr_sepanodes = separate!(cellregs, num_cells(grid), RART, nt, level, ctr_sepanodes)
	end

			
	time[12] = @elapsed (grid[CellRegions] = cellregs)
	
	@info time
	
	return allcells, start
end



function assembly(nm)
	t0 = @elapsed begin
		grid = getgrid(nm)
		number_cells_per_node = zeros(Int64, num_nodes(grid))
		A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
		cn = grid[CellNodes]
		for j=1:num_cells(grid)
			tmp = view(cn, :, j)
			for node_id in tmp
				number_cells_per_node[node_id] += 1
			end
		end
		allcells = zeros(Int64, sum(number_cells_per_node))
		start = ones(Int64, num_nodes(grid)+1)
		start[2:end] += cumsum(number_cells_per_node)
		number_cells_per_node .= 0
		for j=1:num_cells(grid)
			tmp = view(cn, :, j)
			for node_id in tmp
				allcells[start[node_id] + number_cells_per_node[node_id]] = j
				number_cells_per_node[node_id] += 1
			end
		end
	end
	
	t = @elapsed for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		l = length(cells)
		for (i,id1) in enumerate(cells)
			ce = view(cells, i+1:l)
			for id2 in ce
				#for (i,id1) in enumerate(cells)
				#for id2 in cells[i+1:end]
				A[id1,id2] = 1
				A[id2,id1] = 1
			end
		end	
	end
	
	#=
	a = @allocated for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		l = length(cells)
		for (i,id1) in enumerate(cells)
			ce = view(cells, i+1:l)
			for id2 in ce #cells[i+1:end]
				a = 0
				#A[id1,id2] = 1
				#A[id2,id1] = 1
			end
		end	
	end
	=#
	
	t0, t #, a

end


function get_starts(n, nt)
	ret = ones(Int64, nt+1)
	ret[end] = n+1
	for i=nt:-1:2
		ret[i] = ret[i+1] - Int(round(ret[i+1]/i)) #Int(round(n/nt))-1
	end 
	ret
end


function assembly_par(nm, nt)
	t0 = @elapsed begin
		grid = getgrid(nm)
		number_cells_per_node = zeros(Int64, num_nodes(grid))
		cn = grid[CellNodes]
		As = [ExtendableSparseMatrix{Int64, Int64}(num_cells(grid), num_cells(grid)) for tid=1:nt]
		
		#[SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid)) for tid=1:nt]
		for j=1:num_cells(grid)
			tmp = view(cn, :, j)
			for node_id in tmp
				number_cells_per_node[node_id] += 1
			end
		end
		allcells = zeros(Int64, sum(number_cells_per_node))
		start = ones(Int64, num_nodes(grid)+1)
		start[2:end] += cumsum(number_cells_per_node)
		number_cells_per_node .= 0
		for j=1:num_cells(grid)
			tmp = view(cn, :, j)
			for node_id in tmp
				allcells[start[node_id] + number_cells_per_node[node_id]] = j
				number_cells_per_node[node_id] += 1
			end
		end
		
		node_range = get_starts(num_nodes(grid), nt)
	end
	
	done = zeros(Int64, nt)
	
	t = @elapsed Threads.@threads :static for tid=1:nt
		for j in node_range[tid]:node_range[tid+1]-1
			cells = @view allcells[start[j]:start[j+1]-1]
			l = length(cells)
			for (i,id1) in enumerate(cells)
				ce = view(cells, i+1:l)
				for id2 in ce #cells[i+1:end]
					As[tid][id1,id2] = 1
					As[tid][id2,id1] = 1
				end
			end	
		end
		flush!(As[tid])
		#done[tid] = 1
		#if (tid%2 == 0) && (done[tid-1] == 1)
		#	As[tid-1] += As[tid]
		#end
		#...
	end
	
	
	t2 = @elapsed add_all_par!(As)
	t3 = @elapsed (C = As[1].cscmatrix)
	t0, t, t2, t3

end

function add_all_par!(As)
	nt = length(As)
	depth = Int(floor(log2(nt)))
	ende = nt
	for level=1:depth
		
		@threads :static for tid=1:2^(depth-level)
			#@info "$level, $tid"
			start = tid+2^(depth-level)
			while start <= ende
				As[tid] += As[start]
				start += 2^(depth-level)
			end
		end
		ende = 2^(depth-level)
	end
	As[1]

end


function add_all_seq!(As)
	nt = length(As)
	for tid=2:nt
		As[1] += As[tid]
	end
	As[1]
end

function gen(n, m, r)
	[sprand(n, n, r) for i=1:m]
end

function compare(n, m, r)
	
	As = gen(n, m, r)
		
	A1 = add_all_seq!(copy(As))
	A2 = add_all_par!(copy(As))
	diff = (A1-A2).nzval
	if length(diff) > 0
		@info maximum(abs.(diff))
	else
		@info 0
	end
		
	
	for i=1:10
		As = gen(n, m, r)
		
		t1 = @elapsed (add_all_seq!(copy(As)))
		t2 = @elapsed (add_all_par!(copy(As)))
	
		@info t1, t2
		GC.gc()
	end
	
end


function assembly2(nm)
	grid = getgrid(nm)
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	t = @elapsed for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		
		for id1 in cells
			for id2 in cells
				if id1 != id2
					A[id1, id2] = 1
				end
			end
		end
	end
	t

end


function assembly3(nm)
	grid = getgrid(nm)
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	t = @elapsed for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		
		for id1 in cells
			for id2 in cells
				if id1 != id2
					a = 0 #A[id1, id2] = 1
				end
			end
		end
	end
	t

end


function assembly4(nm)
	grid = getgrid(nm)
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	cellnodes = grid[CellNodes]
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	#t = @elapsed 
	t = zeros(4)
	for cell_id=1:num_cells(grid)
		neighbor_cells = []
		
		t[1] += @elapsed for node_id in cellnodes[:,cell_id]
			neighbor_cells = vcat(neighbor_cells, allcells[start[node_id]:start[node_id+1]-1])
		end
		
		t[2] += @elapsed (neighbor_cells = unique(sort!(neighbor_cells)))
		
		tmp = []
		
		for (ctr,cell_j) in enumerate(neighbor_cells)
			t[3] += @elapsed if (cell_j != cell_id)
				A[cell_id, cell_j] = 1.0
				#push!(tmp, cell_j)
			end
		end
		
		
	end
	
	t

end

function assembly42(nm)
	grid = getgrid(nm)
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	cellnodes = grid[CellNodes]
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	#t = @elapsed 
	t = zeros(4)
	for cell_id=1:num_cells(grid)
		neighbor_cells = []
		
		t[1] += @elapsed for node_id in cellnodes[:,cell_id]
			neighbor_cells = vcat(neighbor_cells, allcells[start[node_id]:start[node_id+1]-1])
		end
		
		t[2] += @elapsed (neighbor_cells = unique(sort!(neighbor_cells)))
		
		t[3] += @elapsed for (ctr,cell_j) in enumerate(neighbor_cells)
			if (cell_j != cell_id)
				A[cell_id, cell_j] = 1.0
			end
		end
	end
	
	t

end


function assembly5(nm)
	grid = getgrid(nm)
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	cellnodes = grid[CellNodes]
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	#t = @elapsed 
	t = zeros(4)
	for cell_id=1:num_cells(grid)
		neighbor_cells = []
		
		t[1] += @elapsed for node_id in cellnodes[:,cell_id]
			neighbor_cells = vcat(neighbor_cells, allcells[start[node_id]:start[node_id+1]-1])
		end
		
		#t[2] += @elapsed (neighbor_cells = unique(sort!(neighbor_cells)))
		
		tmp = []
		
		for (ctr,cell_j) in enumerate(neighbor_cells)
			#t[2] += @elapsed (vi = view(neighbor_cells, 1:ctr-1))
			#t[3] += @elapsed (had = (cell_j in tmp))
			#t[4] += @elapsed if !had && (cell_j != cell_id)
			t[4] += @elapsed if !(cell_j in tmp) && (cell_j != cell_id)
				A[cell_id, cell_j] = 1.0
				push!(tmp, cell_j)
			end
		end
		
		
	end
	
	t

end

function assembly6(nm)
	
	t = zeros(7)
	t[1] = @elapsed (grid = getgrid(nm))

	t[2] = @elapsed begin		
		number_cells_per_node = zeros(Int64, num_nodes(grid))
		A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
		cellnodes = grid[CellNodes]
		for j=1:num_cells(grid)
			for node_id in cellnodes[:,j]
				number_cells_per_node[node_id] += 1
			end
		end
		allcells = zeros(Int64, sum(number_cells_per_node))
		start = ones(Int64, num_nodes(grid)+1)
		start[2:end] += cumsum(number_cells_per_node)
		number_cells_per_node .= 0
		for j=1:num_cells(grid)
			for node_id in cellnodes[:,j]
				allcells[start[node_id] + number_cells_per_node[node_id]] = j
				number_cells_per_node[node_id] += 1
			end
		end
	end

	#t = @elapsed 
	for cell_id=1:num_cells(grid)
		neighbor_cells = []
		
		t[3] += @elapsed for node_id in cellnodes[:,cell_id]
			neighbor_cells = vcat(neighbor_cells, allcells[start[node_id]:start[node_id+1]-1])
		end
		
		t[4] += @elapsed (neighbor_cells = sort!(neighbor_cells))
		
		last = 0
		
		#tmp = []
		
		for (ctr,cell_j) in enumerate(neighbor_cells)
			#t[2] += @elapsed (vi = view(neighbor_cells, 1:ctr-1))
			#t[3] += @elapsed (had = (cell_j in tmp))
			#t[4] += @elapsed if !had && (cell_j != cell_id)
			t[5] += @elapsed (newv = (last < cell_j) && (cell_j != cell_id))
			if newv
				t[6] += @elapsed (A[cell_id, cell_j] = 1.0)
			end
			last = cell_j
		end
		
		
	end
	
	t

end



function assembly7(nm)
	
	grid = getgrid(nm)

	number_cells_per_node = zeros(Int64, num_nodes(grid))
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	cellnodes = grid[CellNodes]
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in cellnodes[:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	t = zeros(4)
	for cell_id=1:num_cells(grid)
		neighbor_cells = []
		
		t[1] += @elapsed for node_id in cellnodes[:,cell_id]
			neighbor_cells = vcat(neighbor_cells, allcells[start[node_id]:start[node_id+1]-1])
		end
		
		t[2] += @elapsed (neighbor_cells = sort!(neighbor_cells))
		
		last = 0
		
		t[3] += @elapsed for (ctr,cell_j) in enumerate(neighbor_cells)
			if (last < cell_j) && (cell_j != cell_id)
				(A[cell_id, cell_j] = 1.0)
			end
			last = cell_j
		end
		
		last = 0
		t[4] += @elapsed for (ctr,cell_j) in enumerate(neighbor_cells)
			if (last < cell_j) && (cell_j != cell_id)
				a = 0 #(A[cell_id, cell_j] = 1.0)
			end
			last = cell_j
		end
		
		
	end
	
	t

end
