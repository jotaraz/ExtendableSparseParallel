# normal
function benchmark_pe_LNK_seq(num)
	zeros(num), 0
end


# cheap parallel
function benchmark_pe_LNK_par(C0, As0, nt, num)
	time = zeros(num)
	As = [copy(As0[i]) for i=1:nt]
	C1 = copy(C0)
	C1 = CSC_LNK_plusequals!(As, C1, nt)
	
	for i=1:num
		As = [copy(As0[i]) for i=1:nt]
		C = copy(C0)
		time[i] = @elapsed CSC_LNK_plusequals!(As, C, nt)
		GC.gc()
	end
	
	As = [copy(As0[i]) for i=1:nt]
	C = copy(C0)
	all = @allocated CSC_LNK_plusequals!(As, C, nt)
	
	time, all, C1
end


# RLNK
function benchmark_pe_RLNK_par(C0, As0, nt, gi, num)
	time = zeros(num)
	As = copy(As0) #[copy(As0[i]) for i=1:nt]
	#A2 = copy(As0)
	#@info typeof(As), typeof(As0), typeof(A2)
	C1 = copy(C0)
	C1 = CSC_RLNK_plusequals_less3_reordered_super!(As, C1, nt, gi)
	
	
	for i=1:num
		As = copy(As0) #[copy(As0[i]) for i=1:nt]
		C = copy(C0)
		time[i] = @elapsed CSC_RLNK_plusequals_less3_reordered_super!(As, C, nt, gi)
		GC.gc()
	end
	
	As = copy(As0) #for i=1:nt]
	C = copy(C0)
	all = @allocated CSC_RLNK_plusequals_less3_reordered_super!(As, C, nt, gi)
	
	time, all, C1
end

