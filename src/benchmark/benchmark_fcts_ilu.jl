# ILUZero.jl
function benchmark_ilu_ILUZero(A, num)
	time = zeros(num)
	ILU = ilu0(A)
	
	for i=1:num
		time[i] = @elapsed ilu0(A)
		GC.gc()
	end
	
	all = @allocated ilu0(A)
	
	time, all, ILU
end


# Mittal, Al-Kurdi sequential
function benchmark_ilu_AM_seq(A, num)
	time = zeros(num)
	ILU = create_ILU(A)
	
	for i=1:num
		time[i] = @elapsed create_ILU(A)
		GC.gc()
	end
	
	all = @allocated create_ILU(A)
	
	time, all, ILU
end


# Mittal, Al-Kurdi parallel
function benchmark_ilu_AM_par(A, num)
	time = zeros(num)
	nn = num_nodes(A.grid)
	point = use_vector_par(nn, A.nt, Int32)
	ILU = create_PILU(A, point)
	
	
	for i=1:num
		time[i] = @elapsed begin
			point = use_vector_par(nn, A.nt, Int32)
			create_PILU(A, point)
		end
		GC.gc()
	end
	
	all = @allocated begin
		point = use_vector_par(nn, A.nt, Int32)
		create_PILU(A, point)
	end
	
	time, all, ILU
end
		


