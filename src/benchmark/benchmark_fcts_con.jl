# normal
function benchmark_con_LNK_seq(A, num)
	time = zeros(num)
	C = SparseMatrixCSC(A)
	
	for i=1:num
		time[i] = @elapsed SparseMatrixCSC(A)
		GC.gc()
	end
	
	all = @allocated SparseMatrixCSC(A)
	
	time, all, C
end

# cheap parallel
function benchmark_con_LNK_par(As0, num)
	nt = length(As0)
	time = zeros(num)
	As = [copy(As0[i]) for i=1:nt]
	C = CSC_LNKs_s_dz!(As)
	
	for i=1:num
		As = [copy(As0[i]) for i=1:nt]
		time[i] = @elapsed CSC_LNKs_s_dz!(As)
		GC.gc()
	end
	
	As = [copy(As0[i]) for i=1:nt]
	all = @allocated CSC_LNKs_s_dz!(As)
	
	time, all, C
end

# RLNK
function benchmark_con_RLNK_par(As, onr, s, nt, rni, num)
	time = zeros(num)
	C = CSC_RLNK_si_oc_ps_dz_less_reordered(As, onr, s, nt, rni)
	
	for i=1:num
		time[i] = @elapsed CSC_RLNK_si_oc_ps_dz_less_reordered(As, onr, s, nt, rni)
		GC.gc()
	end
	
	all = @allocated CSC_RLNK_si_oc_ps_dz_less_reordered(As, onr, s, nt, rni)
	
	time, all, C
end




