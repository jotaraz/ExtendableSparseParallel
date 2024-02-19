using SparseArrays
using ExtendableGrids
using ExtendableSparse
using Metis
using Base.Threads
using ThreadPinning
pinthreads(:cores)

include("ESMP/ExtendableSparseParallel.jl")
using .ExtendableSparseParallel




function validate(; nm=(100,100), nt=3, depth=2, Ti=Int64, Tv=Float64)
	A = ExtendableSparseMatrixParallel{Tv, Ti}(nm, nt, depth)
	B = ExtendableSparseMatrixParallel{Tv, Ti}(nm, nt, depth)
	
	full_assembly_seq!(A)
	ExtendableSparseParallel.flush!(A)
	
	
	full_assembly_double_threaded!(B)
	ExtendableSparseParallel.flush!(B)
	
	(A.cscmatrix-B.cscmatrix).nzval
end


function compare(; num=10, nm=(100,100), nt=3, depth=2, Ti=Int64, Tv=Float64)
	A = ExtendableSparseMatrixParallel{Tv, Ti}(nm, nt, depth)
	
	sepa_assembly_seq!(A)
	ExtendableSparseParallel.flush!(A)
	
	t_seq = zeros(num)
	t_thr = zeros(num)
	a_seq = 0
	a_thr = 0
	
	for i=1:num
		t_seq[i] = @elapsed sepa_assembly_seq!(A)
	end
	
	a_seq = @allocated sepa_assembly_seq!(A)	
	
	for i=1:num
		t_thr[i] = @elapsed sepa_assembly_threaded!(A)
	end
	
	a_thr = @allocated sepa_assembly_threaded!(A)	
	
	@info "Time   Sequent.: ", minimum(t_seq)
	@info "Time   @threads: ", minimum(t_thr)
	@info "Allocs Sequent.: ", a_seq
	@info "Allocs @threads: ", a_thr
end

##########################################################################################
### full assembly
##########################################################################################

function full_assembly_seq!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	
	for level=1:depth
		for tid=1:nt
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end
	level = depth+1
	for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
end



function full_assembly_double_threaded!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	
	for level=1:depth
		@threads for tid=1:nt
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end
	level = depth+1
	@threads for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
end

##########################################################################################
### separator assembly
##########################################################################################

function sepa_assembly_seq!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K::Ti = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
end


function sepa_assembly_threaded!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K::Ti = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	@threads for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
end

##########################################################################################
### counting allocations of threaded variant in the function 
##########################################################################################


function full_assembly_double_threaded_outside!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	
	alloc = @allocated for level=1:depth
		@threads for tid=1:nt
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end
	level = depth+1
	alloc += @allocated @threads for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
	alloc
end


function full_assembly_double_threaded_inside!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	
	alloc = zeros(Int64, nt)
	for level=1:depth
		@threads for tid=1:nt
			alloc[tid] += @allocated for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end
	level = depth+1
	@threads for tid=1:1
		alloc[1] += @allocated for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
	sum(alloc)
end

##########################################################################################
### counting allocations of entry adding in separator
##########################################################################################

function sepa_assembly_seq_count!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K::Ti = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	all = 0
	
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				all += @allocated addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					all += @allocated addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					all += @allocated addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
	@info all
end

function sepa_assembly_threaded_count!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K::Ti = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	all = 0
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	@threads for tid=1:1
		for icell in cfp[(level-1)*nt + tid] 
			tmp = view(cellnodes, :, icell)
			for i=1:K
				inode = tmp[i]
				ninode = ni[inode]
				
				all += @allocated addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
				for j=i+1:K
					jnode = tmp[j]
					njnode = ni[jnode]
					v = fr(inode+jnode+offset)*symm
					dv = fr(inode+jnode+offset+1)*skew
					
					all += @allocated addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
					all += @allocated addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
				end
			end
		end
	end
	@info all
end
 

function sepa_assembly_spawn_count!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	K = size(cellnodes)[1]
	
	if !known_that_unknown
		A.cscmatrix.nzval .= 0
	end
	
	all = 0
	nt = A.nt
	depth = A.depth
	ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	@sync begin
		tid=1
		@spawn begin
			for icell in cfp[(level-1)*nt + tid] 
				tmp = view(cellnodes, :, icell)
				for i=1:K
					inode = tmp[i]
					ninode = ni[inode]
					
					all += @allocated addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
					for j=i+1:K
						jnode = tmp[j]
						njnode = ni[jnode]
						v = fr(inode+jnode+offset)*symm
						dv = fr(inode+jnode+offset+1)*skew
						
						all += @allocated addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
						all += @allocated addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
					end
				end
			end
		end
	end
	@info all
end



function multiloop(n, K)
	x = 0

	for idcell=1:n #in cfp[(level-1)*nt + tid]   # =1:1000
		for i in 1:K
			x += 1
		
		end
	end	
end

function ml2!(A::ExtendableSparseMatrixParallel{Tv, Ti}, m; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	
	
	#K = size(cellnodes, 1)
	#using 
	#K::Ti = size(cellnodes, 1)
	#leads to 0 allocations, instead of new allocations for each idcell loop
	
	#if !known_that_unknown
	#	A.cscmatrix.nzval .= 0
	#end
	
	all = 0
	nt = A.nt
	depth = A.depth
	#ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	tid=1
	
	#a = @allocated (
	#it = 1:K#)
	#@info a
	#n = length(cfp[(level-1)*nt + tid])
	
	i = 0
	#@threads 
	
	#for tid=1:1
		for idcell=1:m #in cfp[(level-1)*nt + tid]   # =1:1000
			all += 1
			for i in 1:K
			
			end
		end
	#end
	
	# if K is not known at beginning, then in the inner loop a new iterator has to be 
	
end




function sepa_assembly_inbounds_count!(A::ExtendableSparseMatrixParallel{Tv, Ti}; offset=0, diagval=5.0, symm=0.5, skew=0.25, known_that_unknown=false) where {Tv, Ti <: Integer}
	cellnodes = A.grid[CellNodes]
	#a = @allocated (
	K = size(cellnodes, 1)
	#)
	#@info a
	#K = 3
	
	#if !known_that_unknown
	#	A.cscmatrix.nzval .= 0
	#end
	
	all = 0
	nt = A.nt
	depth = A.depth
	#ni = A.new_indices
	cfp = A.cellsforpart
	
	level = depth+1
	tid=1
	
	#a = @allocated (
	#it = 1:K#)
	#@info a
	n = length(cfp[(level-1)*nt + tid])
	
	i = 0
	#@threads 
	
	#for tid=1:1
		for idcell=1:3 #in cfp[(level-1)*nt + tid]   # =1:1000
			all += 1
			for i in 1:K
			
			end
		end
	#end
	
	# if K is not known at beginning, then in the inner loop a new iterator has to be 
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	#@inbounds 
	
	#
	#icell = cfp[(level-1)*nt + tid][idcell]
		
	
	
	
	#=for icell in 1:K #cfp[(level-1)*nt + tid] 
		#tmp = view(cellnodes, :, icell)
		for i=1:K
			#inode = tmp[i]
			#ninode = ni[inode]
					
			#all += @allocated addtoentry!(A, ninode, ninode, tid, diagval; known_that_unknown=known_that_unknown)
			for j=i+1:K
				#jnode = tmp[j]
				#njnode = ni[jnode]
				#v = fr(inode+jnode+offset)*symm
				#dv = fr(inode+jnode+offset+1)*skew
						
				#all += @allocated addtoentry!(A, ninode, njnode, tid, v; known_that_unknown=known_that_unknown)
				#all += @allocated addtoentry!(A, njnode, ninode, tid, v+dv; known_that_unknown=known_that_unknown)
			end
		end
	end
	=#
	#@info all
end

