# normal
function benchmark_da_LNK_seq(cn, nn, ni, num, Tv, Ti)
	time = zeros(num)
	A = ExtendableSparseMatrix{Tv, Ti}(nn, nn)
	A = da_LNK_sz_reordered!(A, cn, nn, ni)
	
	for i=1:num
		A1 = ExtendableSparseMatrix{Tv, Ti}(nn, nn)
		time[i] = @elapsed (da_LNK_sz_reordered!(A1, cn, nn, ni))
		GC.gc()
	end
	
	A1 = ExtendableSparseMatrix{Tv, Ti}(nn, nn)
	all = @allocated (da_LNK_sz_reordered!(A1, cn, nn, ni))
	
	time, all, A
end


# cheap parallelization / just to get an idea how fast a parallelized process could be
function benchmark_da_LNK_par(cn, nn, cfp, nt, ni, num, Tv, Ti)
	time = zeros(num)
	As = As = [SparseMatrixLNK{Tv, Ti}(nn, nn) for tid=1:nt]
	As = da_LNK_cp_sz_reordered!(As, cn, nn, cfp, nt, ni)
	
	for i=1:num
		As1 = [SparseMatrixLNK{Tv, Ti}(nn, nn) for tid=1:nt]
		time[i] = @elapsed (da_LNK_cp_sz_reordered!(As1, cn, nn, cfp, nt, ni))
		GC.gc()
	end
	
	As1 = [SparseMatrixLNK{Tv, Ti}(nn, nn) for tid=1:nt]
	all = @allocated (da_LNK_cp_sz_reordered!(As1, cn, nn, cfp, nt, ni))
	
	time, all, As

end


# ExtendableSparseMatrixParallel struct
function benchmark_da_ESMP(A, num; parallel=true)
	time1 = zeros(num)
	time2 = zeros(num)
	time3 = zeros(num)
	time4 = zeros(num)
	time5 = zeros(num)
	
	
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true, parallel)
	ExtendableSparseParallel.flush!(A; do_dense=true)
	A1 = copy(A.cscmatrix)
	dummy_assembly!(A; offset=1, diagval=3.0, parallel)
	ExtendableSparseParallel.flush!(A)
	A2 = copy(A.cscmatrix)
	
	#reset!(A)
	#dummy_assembly_time!(A; offset=0, skew=0.0, known_that_unknown=true)
	#reset!(A)
	#dummy_assembly_time!(A; offset=0, skew=0.0, known_that_unknown=true)
	#=
	@warn "Timing ESMP"
	reset!(A)
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true)
	flush!(A; do_dense=true)
	dummy_assembly_time!(A; offset=1, diagval=3.0)
	
	reset!(A)
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true)
	flush!(A; do_dense=true)
	dummy_assembly_time!(A; offset=1, diagval=3.0)
	
	@warn "Timing ESMP AltSepa"
	reset!(A)
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true)
	flush!(A; do_dense=true)
	dummy_assembly_time_altsepa!(A; offset=1, diagval=3.0)
	
	reset!(A)
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true)
	flush!(A; do_dense=true)
	dummy_assembly_time_altsepa!(A; offset=1, diagval=3.0)
	=#
	
	
	for i=1:num
		reset!(A)
		time1[i] = @elapsed dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true, parallel)
		time2[i] = @elapsed ExtendableSparseParallel.flush!(A; do_dense=true)
		GC.gc()
		time3[i] = @elapsed dummy_assembly!(A; offset=1, diagval=3.0, parallel)
		time4[i] = @elapsed ExtendableSparseParallel.flush!(A)
		GC.gc()
	end
	
	for i=1:num
		reset!(A)
		time5[i] = @elapsed dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true, extra_fct=false, parallel)
		GC.gc()
	end
	

	reset!(A)
	all1 = @allocated dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true, parallel)
	all2 = @allocated ExtendableSparseParallel.flush!(A; do_dense=true)
	all3 = @allocated dummy_assembly!(A; offset=1, diagval=3.0, parallel)
	all4 = @allocated ExtendableSparseParallel.flush!(A)
	
	time1, time2, time3, time4, time5, all1, all2, all3, all4, A1, A2, A
end




function compute_static(A, num)
	dummy_assembly!(A; offset=0, skew=0.0, known_that_unknown=true)
	ExtendableSparseParallel.flush!(A; do_dense=true)
	A1 = copy(A.cscmatrix)
	dummy_assembly!(A; offset=1, diagval=3.0)
	ExtendableSparseParallel.flush!(A)
	A2 = copy(A.cscmatrix)
	
	A1, A2
end




function compute_spawn(A, num)
	dummy_assembly_spawn!(A; offset=0, skew=0.0, known_that_unknown=true)
	ExtendableSparseParallel.flush!(A; do_dense=true)
	A1 = copy(A.cscmatrix)
	dummy_assembly_spawn!(A; offset=1, diagval=3.0)
	ExtendableSparseParallel.flush!(A)
	A2 = copy(A.cscmatrix)
	
	A1, A2
end













































