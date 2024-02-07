# normal
function benchmark_dc_LNK_seq(C0, cellnodes, grid, ni, offset, num)
	time = zeros(num)
	C1 = copy(C0)
	C1 = da_csc_LNK_seq_reordered!(C1, cellnodes, grid, ni; offset=offset)
	
	for i=1:num
		C = copy(C0)
		time[i] = @elapsed da_csc_LNK_seq_reordered!(C, cellnodes, grid, ni; offset=offset)
		GC.gc()
	end
	
	C = copy(C0)
	all = @allocated da_csc_LNK_seq_reordered!(C, cellnodes, grid, ni; offset=offset)

	time, all, C1
end



# cheap parallelization / this is inherently unsafe and just done to get an idea how fast a parallelized process could be
function benchmark_dc_LNK_par(C0, cellnodes, cfp, grid, ni, nt, offset, num)
	time = zeros(num)
	C1 = copy(C0)
	C1, A = da_csc_LNK_cp_reordered!(C1, cellnodes, cfp, grid, ni, nt; offset=offset)
	
	for i=1:num
		C = copy(C0)
		time[i] = @elapsed da_csc_LNK_cp_reordered!(C, cellnodes, cfp, grid, ni, nt; offset=offset)
		GC.gc()
	end
	
	C = copy(C0)
	all = @allocated da_csc_LNK_cp_reordered!(C, cellnodes, cfp, grid, ni, nt; offset=offset)

	time, all, C1, A
end




# RLNKs...
function benchmark_dc_RLNK_par(C0, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni, offset, num)
	time = zeros(num)
	C1 = copy(C0)
	C1, A = csc_assembly_pe_less_new2_reordered_super!(C1, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni; offset=offset)
	
	@warn "Timing CSC/RLNK"
	
	C = copy(C0)
	csc_assembly_pe_less_new2_reordered_super_time!(C, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni; offset=offset)
	
	C = copy(C0)
	csc_assembly_pe_less_new2_reordered_super_time!(C, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni; offset=offset)
	
	for i=1:num
		C = copy(C0)
		time[i] = @elapsed (csc_assembly_pe_less_new2_reordered_super!(C, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni; offset=offset))
		GC.gc()
	end
	
	C = copy(C0)
	all = @allocated (csc_assembly_pe_less_new2_reordered_super!(C, cellnodes, nn, nt, cfp, nnts, depth, s, nr, ni; offset=offset))
	
	time, all, C1, A
end	
	



