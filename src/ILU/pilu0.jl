using SparseArrays
using ExtendableSparse
using LinearAlgebra




# https://github.com/j-fu/ExtendableSparse.jl/blob/master/src/factorizations/ilu0.jl
# NO MODIFICATION OF OFF-DIAGONAL ENTRIES!


function total_ILU0p(cscmatrix::SparseMatrixCSC{Tv,Ti}, start, nt, depth, b) where {Tv, Ti <: Integer}
	ilup = ExtendableSparse.ilu0(cscmatrix, start, nt, depth)
	LinearAlgebra.ldiv!(copy(b), ilup, b, start, nt, depth) 	
end

function total_ILU0(cscmatrix::SparseMatrixCSC{Tv,Ti}, b) where {Tv, Ti <: Integer}
	ilup = ExtendableSparse.ilu0(cscmatrix)
	LinearAlgebra.ldiv!(copy(b), ilup, b) 	
end

function decide(i)
	if i == 1
		return ""
	else
		return 0.0
	end
end

function nice(x, y)
	n = length(x)
	mat = ["" for i=1:2, j=1:n]
	for j=1:n
		mat[1,j] = x[j]
		mat[2,j] = "$(y[j])"
	end
	display(mat)
end

function check_disjoint(arrays)
	total = arrays[1]
	for tid=2:length(arrays)
		@info "tid = $tid"
		for a in arrays[tid]
			if (a in total)
				js = findall(x->x==a, total)
				@info "$a from $tid in total: $(js)"
				return
			end
		end
		total = vcat(total, arrays[tid])
	end
	@info "The arrays are disjoint!"
end

"""
x = [x1, x2, x3] (elements are vectors again)
y = [y1, y2, y3] (elements are vectors again)

Are elements of x[i] in y[j] for j != i?
NOT SYMMETRIC!
"""
function check_disjoint2(x, y)
	nt = length(x)
	for i=1:nt
		for j=1:nt
			if i!=j
				for xx in x[i]
					if xx in y[j]
						@info "$xx from $i in $j"
						return
					end
				end
			end
		end
		@info "x[$i] not in y"
	end
end


function ExtendableSparse.ilu0(cscmatrix::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <: Integer}
    colptr = cscmatrix.colptr
    rowval = cscmatrix.rowval
    nzval = cscmatrix.nzval
    n = cscmatrix.n
    
    
    
    
    # Find main diagonal index and
    # copy main diagonal values
    idiag=Vector{Ti}(undef,n)
    
    for level=1:depth
    	@threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:start[(level-1)*nt + tid+1]-1
    			@inbounds for k = colptr[j]:(colptr[j + 1] - 1)
				    i = rowval[k]
				    if i == j
				        idiag[j] = k
				        break
				    end
				end
			end
		end
    end
    
    part = nt*depth+1
    @inbounds for j=start[part]:start[part+1]-1
    	@inbounds for k = colptr[j]:(colptr[j + 1] - 1)
		    i = rowval[k]
		    if i == j
		        idiag[j] = k
		        break
		    end
		end
	end
    
    xdiag=Vector{Tv}(undef,n)
    for level=1:depth
    	@threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:start[(level-1)*nt + tid+1]-1
    			#@inbounds for j = 1:n
				xdiag[j] = one(Tv) / nzval[idiag[j]]
				@inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    for l = colptr[i]:(colptr[i + 1] - 1)
				        if rowval[l] == j
				            xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
				            break
				        end
				    end
				end
			end
		end
	end
	
	
	part = nt*depth+1
	@inbounds for j=start[part]:start[part+1]-1
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l = colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
	
    ExtendableSparse._ILU0Preconditioner(cscmatrix,xdiag,idiag)
    #cscmatrix
end


function ilu0_time(cscmatrix::SparseMatrixCSC{Tv,Ti}) where{Tv,Ti}
	times = zeros(4)
	names = ["" for i=1:4]
	
	names[1] = "Pre"
	times[1] = @elapsed begin
		colptr = cscmatrix.colptr
		rowval = cscmatrix.rowval
		nzval = cscmatrix.nzval
		n = cscmatrix.n
		
		# Find main diagonal index and
		# copy main diagonal values
		idiag=Vector{Ti}(undef,n)
		xdiag=Vector{Tv}(undef,n)
	end
	
	names[2] = "L1"
    times[2] = @elapsed @inbounds for j = 1:n
        @inbounds for k = colptr[j]:(colptr[j + 1] - 1)
            i = rowval[k]
            if i == j
                idiag[j] = k
                break
            end
        end
    end
    
    names[3] = "L2"
    times[3] = @elapsed @inbounds for j = 1:n
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l = colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
    
    names[4] = "create"
    times[4] = @elapsed (ilu = ExtendableSparse._ILU0Preconditioner(cscmatrix,xdiag,idiag))
    
    nice(names, times)
    
    ilu
end


function ilu0_time(cscmatrix::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where {Tv, Ti <: Integer}
    times = zeros(4+2*depth)
    names = ["Pre" for i=1:4+2*depth]
    times[1] = @elapsed begin
		colptr = cscmatrix.colptr
		rowval = cscmatrix.rowval
		nzval = cscmatrix.nzval
		n = cscmatrix.n
		
		# Find main diagonal index and
		# copy main diagonal values
		idiag=Vector{Ti}(undef,n)
		xdiag=Vector{Tv}(undef,n)
    
	end
	    
    for level=1:depth
    	names[1+level] = "L1, l$level"
    	times[1+level] = @elapsed @threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:start[(level-1)*nt + tid+1]-1
    			@inbounds for k = colptr[j]:(colptr[j + 1] - 1)
				    i = rowval[k]
				    if i == j
				        idiag[j] = k
				        break
				    end
				end
			end
		end
    end
    
    part = nt*depth+1
    names[2+depth] = "L1, sepa"
    times[2+depth] = @elapsed @inbounds for j=start[part]:start[part+1]-1
    	@inbounds for k = colptr[j]:(colptr[j + 1] - 1)
		    i = rowval[k]
		    if i == j
		        idiag[j] = k
		        break
		    end
		end
	end
    
    for level=1:depth
    	names[2+depth+level] = "L2, l$level"
    	times[2+depth+level] = @elapsed @threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:start[(level-1)*nt + tid+1]-1
    			#@inbounds for j = 1:n
				xdiag[j] = one(Tv) / nzval[idiag[j]]
				@inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    for l = colptr[i]:(colptr[i + 1] - 1)
				        if rowval[l] == j
				            xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
				            break
				        end
				    end
				end
			end
		end
	end
	
	
	part = nt*depth+1
    names[3+2*depth] = "L2, sepa"
	times[3+2*depth] = @elapsed @inbounds for j=start[part]:start[part+1]-1
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l = colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
	
	names[4+2*depth] = "create"
    times[4+2*depth] = @elapsed (ilu = ExtendableSparse._ILU0Preconditioner(cscmatrix,xdiag,idiag))
    nice(names, times)
    ilu
end


function ilu0_test(cscmatrix::SparseMatrixCSC{Tv,Ti}, start, nt, depth) where{Tv,Ti}
    colptr = cscmatrix.colptr
    rowval = cscmatrix.rowval
    nzval = cscmatrix.nzval
    n = cscmatrix.n
    
    test_diag(cscmatrix)
    
    @info start
    
    #@info [length(start[i]) for i=1:length(start)]
    
    # Find main diagonal index and
    # copy main diagonal values
    idiag=zeros(Ti, n) #Vector{Ti}(undef,n)
    
    for level=1:depth
    	arrays = [Vector{Int64}(undef, n) for tid=1:nt]
    	ctrs = zeros(Int64, nt)
    	@threads :static for tid=1:nt
    		#ctr = 1
    		#ctrs[tid] = (level-1)*nt + tid
    		
    		#@info tid, ctrs[tid], ((level-1)*nt + tid)
    		
    		@info "$tid: $(start[(level-1)*nt + tid]) : $(start[(level-1)*nt + tid+1]-1)"
    		
    		#@inbounds 
    		
    		for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			found_diag = false
    			#@inbounds 
    			for k = colptr[j]:(colptr[j + 1] - 1)
				    i = rowval[k]
				    if i == j
				        idiag[j] = k
				        found_diag = true
				        #push!(arrays[tid], j)
				        ctrs[tid] += 1
				        arrays[tid][ctrs[tid]] = j
				        break
				    end
				end
				if !found_diag
					@info "Can't find diag for $j"
				end
			end
			
		end
		
		@info ctrs
		@info [start[(level-1)*nt+tid+1]-start[(level-1)*nt+tid] for tid=1:nt]
		
		
		
		#@info sum(ctrs), length(collect(start[(level-1)*nt+1]:start[level*nt+1]-1))
		va = vcat([arrays[i][1:ctrs[i]] for i=1:nt]...)
		if va != collect(start[(level-1)*nt+1]:start[level*nt+1]-1)
			@info "not equal"
			@info "va:", va
			@info "..:", length(collect(start[(level-1)*nt+1]:start[level*nt+1]-1))
		else
			@info "equal!"
		end
    end
    
    part = nt*depth+1
    @inbounds for j=start[part]:start[part+1]-1
    	@inbounds for k = colptr[j]:(colptr[j + 1] - 1)
		    i = rowval[k]
		    if i == j
		        idiag[j] = k
		        break
		    end
		end
	end
    
    
    if zero(Ti) in idiag
	    @info "0 in diag"
	end
    
    xdiag=Vector{Tv}(undef,n)
    for level=1:depth
    	arrays = [[] for tid=1:nt]
    	@threads :static for tid=1:nt
    		#part = (level-1)*nt + tid
    		@info "level = $level, tid = $tid"
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			#@inbounds for j = 1:n
    			xdiag[j] = one(Tv) / nzval[idiag[j]]
				@inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    for l = colptr[i]:(colptr[i + 1] - 1)
				        if rowval[l] == j
				            xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
				            push!(arrays[tid], i)
				            break
				        end
				    end
				end
			end
		end
		check_disjoint(unique.(arrays))
	end
	
	
	part = nt*depth+1
	@inbounds for j=start[part]:start[part+1]-1
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l = colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
	
    ExtendableSparse._ILU0Preconditioner(cscmatrix,xdiag,idiag)
    #cscmatrix
end



function LinearAlgebra.ldiv!(u, p::ExtendableSparse._ILU0Preconditioner{Tv,Ti}, v, start, nt, depth) where {Tv,Ti}
    colptr = p.cscmatrix.colptr
    rowval = p.cscmatrix.rowval
    nzval = p.cscmatrix.nzval
    n = p.cscmatrix.n
    idiag = p.idiag
    xdiag = p.xdiag
    
    for level=1:depth
    	@threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			u[j] = xdiag[j] * v[j]
    		end
    	end
    end
    
    @inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		u[j] = xdiag[j] * v[j]
	end
    
    #@info maximum(abs.(u))
    
    @inbounds for j=(start[depth*nt+2]-1):-1:start[depth*nt+1]
		for k = (idiag[j] + 1):(colptr[j + 1] - 1)
		    i = rowval[k]
		    u[i] -= xdiag[i] * nzval[k] * u[j]
		end
	end
    
    for level=depth:-1:1
    	@threads :static for tid=1:nt
    		@inbounds for j=(start[(level-1)*nt + tid+1]-1):-1:start[(level-1)*nt + tid]
    			for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
	
	
	
	
    #@info maximum(abs.(u))
    	
    	
    for level=1:depth
    	@threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			for k = colptr[j]:(idiag[j] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
    
    
    
    @inbounds for j=start[depth*nt+1]:(start[depth*nt+2]-1)
        for k = colptr[j]:(idiag[j] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    u
end




function ldiv_time!(u, p::ExtendableSparse._ILU0Preconditioner{Tv,Ti}, v, start, nt, depth) where {Tv,Ti}
	times = zeros(4+3*depth)
	names = ["" for i=1:4+3*depth]
	names[1] = "Pre"
	times[1] = @elapsed begin
		colptr = p.cscmatrix.colptr
		rowval = p.cscmatrix.rowval
		nzval = p.cscmatrix.nzval
		n = p.cscmatrix.n
		idiag = p.idiag
		xdiag = p.xdiag
	end
    
    
    for level=1:depth
    	names[1+level] = "L1, l$level"
    	times[1+level] = @elapsed @threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			u[j] = xdiag[j] * v[j]
    		end
    	end
    end
    
    names[2+depth] = "L1, sepa"
    times[2+depth] = @elapsed @inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		u[j] = xdiag[j] * v[j]
	end
    
    #@info maximum(abs.(u))
    
    
    names[3+depth] = "L2, sepa"
    times[3+depth] = @elapsed @inbounds for j=(start[depth*nt+2]-1):-1:start[depth*nt+1]
		for k = (idiag[j] + 1):(colptr[j + 1] - 1)
		    i = rowval[k]
		    u[i] -= xdiag[i] * nzval[k] * u[j]
		end
	end
    
    for level=depth:-1:1
    	names[3+depth+level] = "L2, l$level"
    	times[3+depth+level] = @elapsed @threads :static for tid=1:nt
    		@inbounds for j=(start[(level-1)*nt + tid+1]-1):-1:start[(level-1)*nt + tid]
    			for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
	
	
	
	
    #@info maximum(abs.(u))
    	
    	
    for level=1:depth
    	names[3+2*depth+level] = "L3, l$level"
    	times[3+2*depth+level] = @elapsed @threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			for k = colptr[j]:(idiag[j] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
    
    
    names[4+3*depth] = "L3, sepa"
    times[4+3*depth] = @elapsed @inbounds for j=start[depth*nt+1]:(start[depth*nt+2]-1)
        for k = colptr[j]:(idiag[j] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    
	nice(names, times)
    
    
    u
end


function ldiv_test!(u, p::ExtendableSparse._ILU0Preconditioner{Tv,Ti}, v, start, nt, depth) where {Tv,Ti}
    colptr = p.cscmatrix.colptr
    rowval = p.cscmatrix.rowval
    nzval = p.cscmatrix.nzval
    n = p.cscmatrix.n
    idiag = p.idiag
    xdiag = p.xdiag
    
    for level=1:depth
    	#@threads :static 
    	for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			u[j] = xdiag[j] * v[j]
    		end
    	end
    end
    
    @inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		u[j] = xdiag[j] * v[j]
	end
    
    @info maximum(abs.(u))
    
    for level=1:depth
    	#@threads :static 
    	for tid=1:nt
    		@inbounds for j=(start[(level-1)*nt + tid+1]-1):-1:start[(level-1)*nt + tid]
    			for k = (idiag[j] + 1):(colptr[j + 1] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
	
	@inbounds for j=(start[depth*nt+2]-1):-1:start[depth*nt+1]
		for k = (idiag[j] + 1):(colptr[j + 1] - 1)
		    i = rowval[k]
		    u[i] -= xdiag[i] * nzval[k] * u[j]
		end
	end
	
	
    @info maximum(abs.(u))
    	
    	
    for level=1:depth
    	@threads :static for tid=1:nt
    		@inbounds for j=start[(level-1)*nt + tid]:(start[(level-1)*nt + tid+1]-1)
    			for k = colptr[j]:(idiag[j] - 1)
				    i = rowval[k]
				    u[i] -= xdiag[i] * nzval[k] * u[j]
				end
			end
		end
	end
    
    
    
    @inbounds for j=start[depth*nt+1]:(start[depth*nt+2]-1)
        for k = colptr[j]:(idiag[j] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    u
end

function ldiv_time!(u,p::ExtendableSparse._ILU0Preconditioner{Tv,Ti},v) where {Tv,Ti}
    times = zeros(4)
    names = ["Pre", "L1", "L2", "L3"]
    
    times[1] = @elapsed begin
		colptr = p.cscmatrix.colptr
		rowval = p.cscmatrix.rowval
		nzval = p.cscmatrix.nzval
		n = p.cscmatrix.n
		idiag = p.idiag
		xdiag = p.xdiag
	end
    
    times[2] = @elapsed for j = 1:n
        u[j] = xdiag[j] * v[j]
    end
    
    times[3] = @elapsed for j = n:-1:1
        for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    
    times[4] = @elapsed for j = 1:n
        for k = colptr[j]:(idiag[j] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    
    nice(names, times)
    
    u
end


#=

function get_LU_struct(A::SparseArrays.SparseMatrixCSC{Tv, Ti}, starts::Vector{Ti}) where {Tv, Ti <: Integer}

	
	
end


function ILU0(A::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}










end


function locate(i, j, starts)
	for k=1:length(starts)-1
		if (starts[k] < i && i <= starts[k+1]) || (starts[k] < j && j <= starts[k+1])
			return k
		end
	end
end



function ILUZero.ILU0Precon(A::SparseMatrixCSC{T,N}, starts::Vector{N}, nt, b_type = T) where {T <: Any,N <: Integer}
	@warn "Wrong starts? / not equivalent to the one in prep???"
	depth = Int((length(starts)-2)/nt)
    m, n = size(A)

    # Determine number of elements in lower/upper
    # lnz = 0
    # unz = 0
    
    lit0 = zeros(N, depth*nt+1)
    uit0 = zeros(N, depth*nt+1)
    
    lrow0 = [Vector{N}(undef, n) for i=1:nt*depth+1]
    
    
    urow0 = [Vector{N}(undef, n) for i=1:nt*depth+1]
    urow1 = [Vector{N}(undef, n) for i=1:nt*depth+1]
    
    @inbounds for i = 1:n #columns
    	for tid=1:nt*depth+1
    		lrow0[tid][i] = 0
    		urow0[tid][i] = 0
    	end
        for j = A.colptr[i]:A.colptr[i + 1] - 1 #rows
            if A.rowval[j] > i
		            tid = locate(i, j, starts) 
		            lit0[tid] += 1
		            
		            if lrow0[tid0][i] == 0
		            	lrow0[tid][i] = j
		            end
		            
		            
		        else
		            uit0[locate(i, j, starts)] += 1
		            
		            if urow0[tid0][i] == 0
		            	urow0[tid][i] = j
		            end
		            urow1[tid][i] = j
		            
		            #unz += 1
		        end
        end
    end

	lnz = sum(lit0)
    unz = sum(uit0)
    
    l_starts = vcat([0], cumsum(lit0)).+1
    u_starts = vcat([0], cumsum(uit0)).+1
    
    

    # Preallocate variables
    l_colptr = zeros(N, n + 1)
    u_colptr = zeros(N, n + 1)
    l_nzval = zeros(T, lnz)
    u_nzval = zeros(T, unz)
    l_rowval = zeros(Int64, lnz)
    u_rowval = zeros(Int64, unz)
    l_map = Vector{N}(undef, lnz)
    u_map = Vector{N}(undef, unz)
    wrk = zeros(b_type, n)
    l_colptr[1] = 1
    u_colptr[1] = 1

    #
    for level=1:depth
    	@threads :static for tid=1:nt
    		pid = (level-1)*nt+tid
    		i0 = starts[pid]+1
    		lit = l_starts[pid]
    		uit = u_starts[pid]
    		for i=i0:n
    			for j=urow0[pid]:urow1[pid]
    				# do u things
    				u_colptr[i + 1] += 1
		            u_rowval[uit] = A.rowval[j]
		            u_map[uit] = j
		            uit += 1
    			end
    			
    			for j=lrow0[pid]:A.colptr[i+1]-1
    				# do l things
    				l_colptr[i + 1] += 1
		            l_rowval[lit] = A.rowval[j]
		            l_map[lit] = j
		            lit += 1
    			end
    		
    		end
    		
    	end
    end
    
    # separator
    pid = depth*nt+1
    i0 = starts[pid]+1
	lit = l_starts[pid]
	uit = u_starts[pid]
	for i=i0:n
		for j=urow0[pid]:urow1[pid]
			# do u things
			u_colptr[i + 1] += 1
            u_rowval[uit] = A.rowval[j]
            u_map[uit] = j
            uit += 1
		end
		
		for j=lrow0[pid]:A.colptr[i+1]-1
			# do l things
			l_colptr[i + 1] += 1
            l_rowval[lit] = A.rowval[j]
            l_map[lit] = j
            lit += 1
		end
	
	end
    
    
    
    # Map elements of A to lower and upper triangles, fill out colptr, and fill out rowval
	#=  
    lit = 1
    uit = 1
    @inbounds for i = 1:n
        l_colptr[i + 1] = l_colptr[i]
        u_colptr[i + 1] = u_colptr[i]
        for j = A.colptr[i]:A.colptr[i + 1] - 1
            if A.rowval[j] > i
                l_colptr[i + 1] += 1
                l_rowval[lit] = A.rowval[j]
                l_map[lit] = j
                lit += 1
            else
                u_colptr[i + 1] += 1
                u_rowval[uit] = A.rowval[j]
                u_map[uit] = j
                uit += 1
            end
        end
    end
    =#

    return ILU0Precon(m, n, l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map, wrk)
end


function ilu0!(LU::ILU0Precon{T,N}, A::SparseMatrixCSC{T,N}) where {T <: Any,N <: Integer}
    m = LU.m
    n = LU.n
    l_colptr = LU.l_colptr
    l_rowval = LU.l_rowval
    l_nzval = LU.l_nzval
    u_colptr = LU.u_colptr
    u_rowval = LU.u_rowval
    u_nzval = LU.u_nzval
    l_map = LU.l_map
    u_map = LU.u_map

    # Redundant data or better speed... speed is chosen, but this might be changed.
    # This shouldn't be inbounded either.
    for i = 1:length(l_map)
        l_nzval[i] = A.nzval[l_map[i]]
    end
    for i = 1:length(u_map)
        u_nzval[i] = A.nzval[u_map[i]]
    end


    @inbounds for i = 1:m - 1
        m_inv = inv(u_nzval[u_colptr[i + 1] - 1])
        for j = l_colptr[i]:l_colptr[i + 1] - 1
            l_nzval[j] = l_nzval[j] * m_inv
        end
        for j = u_colptr[i + 1]:u_colptr[i + 2] - 2
            multiplier = u_nzval[j]
            qn = j + 1
            rn = l_colptr[i + 1]
            pn = l_colptr[u_rowval[j]]
            while pn < l_colptr[u_rowval[j] + 1] && l_rowval[pn] <= i + 1
                while qn < u_colptr[i + 2] && u_rowval[qn] < l_rowval[pn]
                    qn += 1
                end
                if qn < u_colptr[i + 2] && l_rowval[pn] == u_rowval[qn]
                    u_nzval[qn] -= l_nzval[pn] * multiplier
                end
                pn += 1
            end
            while pn < l_colptr[u_rowval[j] + 1]
                while rn < l_colptr[i + 2] && l_rowval[rn] < l_rowval[pn]
                    rn += 1
                end
                if rn < l_colptr[i + 2] && l_rowval[pn] == l_rowval[rn]
                    l_nzval[rn] -= l_nzval[pn] * multiplier
                end
                pn += 1
            end
        end
    end
    return
end


=#

