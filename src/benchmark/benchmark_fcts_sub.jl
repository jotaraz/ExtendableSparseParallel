# ILUZero.jl
function benchmark_sub_ILUZero(ILU, b, num)
	time = zeros(num)
	n = length(b)
	z = ILUZero_solve(ILU, b)
	
	#=
	@warn "Timing forw ILUZ"
	y = rand(n)
	ILUZero_solve_time(ILU, y)
	y = rand(n)
	ILUZero_solve_time(ILU, y)
	y = rand(n)
	ILUZero_solve_time(ILU, y)
	@warn "------------"
	=#
	
	for i=1:num
		y = rand(n)
		time[i] = @elapsed ILUZero_solve(ILU, y)
		GC.gc()
	end
	
	all = @allocated ILUZero_solve(ILU, b)
	
	time, all, z
end


# Mittal, Al-Kurdi sequential
function benchmark_sub_AM_seq(ILU::ILUPrecon, b, num)
	time = zeros(num)
	n = length(b)
	z = MA_solve_seq_old(ILU, b)
	
	#=
	@warn "Timing forward subst"
	y = rand(n)
	MA_solve_seq_old_time(ILU, y)
	y = rand(n)
	MA_solve_seq_old_time(ILU, y)
	y = rand(n)
	MA_solve_seq_old_time(ILU, y)
	@warn "---------"
	=#
	
	for i=1:num
		y = rand(n)
		time[i] = @elapsed MA_solve_seq_old(ILU, y)
		GC.gc()
	end
	
	all = @allocated MA_solve_seq_old(ILU, b)
	
	time, all, z
end


# Mittal, Al-Kurdi parallel
function benchmark_sub_AM_par(PILU::PILUPrecon, b, num)
	time = zeros(num)
	n = length(b)
	z = MA_solve_par_old(PILU, b)
	
	
	for i=1:num
		y = rand(n)
		time[i] = @elapsed MA_solve_par_old(PILU, y)
		GC.gc()
	end
	
	all = @allocated MA_solve_par_old(PILU, b)
	
	time, all, z
end
		

#=
# Mittal, Al-Kurdi sequential
function benchmark_sub_AM_seq(ILU, b, num, CSC, diag)
	time = zeros(num)
	n = length(b)
	z = MA_solve_seq(ILU, CSC, diag, b)
	
	
	for i=1:num
		y = rand(n)
		time[i] = @elapsed MA_solve_seq(ILU, CSC, diag, y)
		GC.gc()
	end
	
	all = @allocated MA_solve_seq(ILU, CSC, diag, b)
	
	time, all, z
end


# Mittal, Al-Kurdi parallel
function benchmark_sub_AM_par(ILU, b, num, CSC, diag, starts, nt, depth)
	time = zeros(num)
	n = length(b)
	z = MA_solve_par(ILU, CSC, diag, b, starts, nt, depth)
	
	
	for i=1:num
		y = rand(n)
		time[i] = @elapsed MA_solve_par(ILU, CSC, diag, y, starts, nt, depth)
		GC.gc()
	end
	
	all = @allocated MA_solve_par(ILU, CSC, diag, b, starts, nt, depth)
	
	time, all, z
end
=#

