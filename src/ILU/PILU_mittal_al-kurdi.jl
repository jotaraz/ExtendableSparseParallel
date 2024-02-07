using SparseArrays

mutable struct PILUPrecon

	nzval::AbstractVector
	diag::AbstractVector
	A::AbstractMatrix
	start::AbstractVector
	nt::Integer
	depth::Integer

end


function use_vector_par(n, nt, Ti)
	point = [Vector{Ti}(undef, n) for tid=1:nt]
	@threads :static for tid=1:nt
		point[tid] = zeros(Ti, n)
	end
	point
end

function zeros_seq(n, nt, Ti)
	[zeros(Ti, n) for tid=1:nt]
end

function create_PILU(A::ExtendableSparseMatrixParallel{Tv,Ti}, point) where {Tv, Ti <:Integer}
	start = A.start
	nt = A.nt
	depth = A.depth
	
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
	nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
	n = A.cscmatrix.n # number of columns
	diag  = Vector{Ti}(undef, n)
	
	# find diagonal entries
	#
	@threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.cscmatrix.nzval[v]
				if rowval[v] == j
					diag[j] = v
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	#point =  use_vector_par(n, nt, Ti) #[zeros(Ti, n) for tid=1:nt] #Vector{Ti}(undef, n)
	for level=1:depth
		@threads :static for tid=1:nt
			for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	
	PILUPrecon(nzval, diag, A.cscmatrix, start, nt, depth)
end

function create_PILU_time2(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <:Integer}
	times = zeros(4+depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					#break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	times[3] = @elapsed (point =  use_vector_par(n, nt, Ti))  
	#(point = [zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		times[3+level] = @elapsed @threads :static for tid=1:nt
			for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	#@info times
	
	@warn "Quick:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4]
	@info "lev2:", times[5]
	@info "sepa:", times[6]
	
	PILUPrecon(nzval, diag, A, start, nt, depth)
end

function create_PILU_time2_copy(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <:Integer}
	times = zeros(4+depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				#nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	times[3] = @elapsed (point =  use_vector_par(n, nt, Ti))
	#(point = [zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		times[3+level] = @elapsed @threads :static for tid=1:nt
			for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	#@info times
	
	@warn "Quick:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4]
	@info "lev2:", times[5]
	@info "sepa:", times[6]
	
	PILUPrecon(nzval, diag, A, start, nt, depth)
end

function create_PILU_time1_copy(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <:Integer}
	times = zeros(4+nt*depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				#nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	times[3] = @elapsed (point =  use_vector_par(n, nt, Ti))
	#(point = [zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		for tid=1:nt
			times[3+(level-1)*nt+tid] = @elapsed for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+nt*depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	@warn "Detail:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4:3+nt]
	@info "lev2:", times[3+nt+1:3+2*nt]
	@info "sepa:", times[4+2*nt]
	PILUPrecon(nzval, diag, A, start, nt, depth)
end


function create_PILU_time1_pg(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth, point) where {Tv, Ti <:Integer}
	times = zeros(4+nt*depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					#break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	#times[3] = @elapsed (point =  use_vector_par(n, nt, Ti))
	#(point = [zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		for tid=1:nt
			times[3+(level-1)*nt+tid] = @elapsed for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+nt*depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	@warn "Detail:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4:3+nt]
	@info "lev2:", times[3+nt+1:3+2*nt]
	@info "sepa:", times[4+2*nt]
	PILUPrecon(nzval, diag, A, start, nt, depth)
end

function create_PILU_time1(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <:Integer}
	times = zeros(4+nt*depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					#break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	times[3] = @elapsed (point =  use_vector_par(n, nt, Ti))
	#(point = [zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		for tid=1:nt
			times[3+(level-1)*nt+tid] = @elapsed for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+nt*depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end
	@warn "Detail:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4:3+nt]
	@info "lev2:", times[3+nt+1:3+2*nt]
	@info "sepa:", times[4+2*nt]
	PILUPrecon(nzval, diag, A, start, nt, depth)
end

function create_PILU_time1_seqpoint(A::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <:Integer}
	times = zeros(4+nt*depth)
	times[1] = @elapsed begin
		#ILU = copy(A)
		colptr = A.colptr
		rowval = A.rowval
		nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
		n = A.n # number of columns
		diag  = Vector{Ti}(undef, n)
	end	
	# find diagonal entries
	#
	
	times[2] = @elapsed @threads :static for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.nzval[v]
				if rowval[v] == j
					diag[j] = v
					#break
				end
				#elseif rowval[v] 
			end
		end
	end
	
	# compute L and U
	
	times[3] = @elapsed (point = zeros(Ti, (nt,n))) #[zeros(Ti, n) for tid=1:nt]) #Vector{Ti}(undef, n)
	for level=1:depth
		for tid=1:nt
			times[3+(level-1)*nt+tid] = @elapsed for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid,rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid,rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid,rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	times[4+nt*depth] = @elapsed for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1,rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1,rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1,rowval[v]] = zero(Ti)
		end
	end
	@warn "Detail:"
	@info "Pre :", times[1]
	@info "diag:", times[2]
	@info "poin:", times[3]
	@info "lev1:", times[4:3+nt]
	@info "lev2:", times[3+nt+1:3+2*nt]
	@info "sepa:", times[4+2*nt]
	PILUPrecon(nzval, diag, A, start, nt, depth)
end



function forward_subst!(y, v, pilu::PILUPrecon)
	nzval = pilu.nzval
	diag  = pilu.diag
	start = pilu.start
	nt = pilu.nt
	depth = pilu.depth	
	n = pilu.A.n
	colptr = pilu.A.colptr
	rowval = pilu.A.rowval
	
	y .= 0
	
	for level=1:depth
		@threads :static for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				y[j] += v[j]
				for v=diag[j]+1:colptr[j+1]-1
					y[rowval[v]] -= nzval[v]*y[j]
				end
			end
		end
	end
	
	@inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end

function forward_subst_old!(y, v, nzval, diag, start, nt, depth, A)
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	
	y .= 0
	
	for level=1:depth
		@threads :static for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				y[j] += v[j]
				for v=diag[j]+1:colptr[j+1]-1
					y[rowval[v]] -= nzval[v]*y[j]
				end
			end
		end
	end
	
	@inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end

function forward_subst_test!(y, v, pilu::PILUPrecon)
	nzval = pilu.nzval
	diag  = pilu.diag
	start = pilu.start
	nt = pilu.nt
	depth = pilu.depth	
	n = pilu.A.n
	colptr = pilu.A.colptr
	rowval = pilu.A.rowval
	
	y .= 0
	
	for level=1:depth
		arrays = [[] for tid=1:nt]
		#@threads :static 
		for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				y[j] += v[j]
				push!(arrays[tid], j)
				for v=diag[j]+1:colptr[j+1]-1
					y[rowval[v]] -= nzval[v]*y[j]
					push!(arrays[tid], rowval[v])
				end
			end
		end
		@info "level $level"
		check_disjoint(arrays)
	end
	
	@inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end



"""
Solving Ux = y for the upper-triangular matrix U `encoded` in ILU.
"""
function backward_subst!(x, y, pilu::PILUPrecon)
	nzval = pilu.nzval
	diag  = pilu.diag
	start = pilu.start
	nt = pilu.nt
	depth = pilu.depth	
	n = pilu.A.n
	colptr = pilu.A.colptr
	rowval = pilu.A.rowval
	wrk = copy(y)
	
	
	@inbounds for j=start[depth*nt+2]-1:-1:start[depth*nt+1]
		x[j] = wrk[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	
	for level=depth:-1:1
		@threads :static for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid+1]-1:-1:start[(level-1)*nt+tid]
				x[j] = wrk[j] / nzval[diag[j]] 
				for i=colptr[j]:diag[j]-1
					wrk[rowval[i]] -= nzval[i]*x[j]
				end
			end
		end
	end

end

function backward_subst_old!(x, y, nzval, diag, start, nt, depth, A)
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	#wrk = copy(y)
	
	
	@inbounds for j=start[depth*nt+2]-1:-1:start[depth*nt+1]
		x[j] = y[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			y[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	
	for level=depth:-1:1
		@threads :static for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid+1]-1:-1:start[(level-1)*nt+tid]
				x[j] = y[j] / nzval[diag[j]] 
				for i=colptr[j]:diag[j]-1
					y[rowval[i]] -= nzval[i]*x[j]
				end
			end
		end
	end

end

function backward_subst_test!(x, y, pilu::PILUPrecon)
	nzval = pilu.nzval
	diag  = pilu.diag
	start = pilu.start
	nt = pilu.nt
	depth = pilu.depth	
	n = pilu.A.n
	colptr = pilu.A.colptr
	rowval = pilu.A.rowval
	wrk = copy(y)
	
	
	for level=1:depth
		@info "level $level"
		arrays = [[] for tid=1:nt]
		#@threads :static 
		for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				x[j] = wrk[j] / nzval[diag[j]] 
				push!(arrays[tid], j)
				for i=colptr[j]:diag[j]-1
					wrk[rowval[i]] -= nzval[i]*x[j] 
					push!(arrays[tid], rowval[i])
				end
			end
		end
		@info "level $level"
		check_disjoint(arrays)
		
	end
	
	
	@inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		x[j] = wrk[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
end
