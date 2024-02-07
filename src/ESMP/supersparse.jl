
using SparseArrays
using ExtendableSparse

mutable struct SuperSparseMatrixLNK{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    """
    Number of rows
    """
    m::Ti

    """
    Number of columns
    """
    n::Ti

    """
    Number of nonzeros
    """
    nnz::Ti

    """
    Length of arrays
    """
    nentries::Ti

    """
    Linked list of column entries. Initial length is n,
    it grows with each new entry.

    colptr[index] contains the next
    index in the list or zero, in the later case terminating the list which
    starts at index 1<=j<=n for each column j.
    """
    colptr::Vector{Ti}

    """
    Row numbers. For each index it contains the zero (initial state)
    or the row numbers corresponding to the column entry list in colptr.

    Initial length is n,
    it grows with each new entry.
    """
    rowval::Vector{Ti}

    """
    Nonzero entry values correspondin to each pair
    (colptr[index],rowval[index])

    Initial length is n,  it grows with each new entry.
    """
    nzval::Vector{Tv}
    
    
    collnk::Vector{Ti}
    
    colctr::Ti
end


function SparseArrays.SparseMatrixCSC(A::SuperSparseMatrixLNK{Tv, Ti})::SparseArrays.SparseMatrixCSC where {Tv, Ti <: Integer}
	SparseArrays.SparseMatrixCSC(SparseMatrixLNK{Tv, Ti}(A.m, A.n, A.nnz, A.nentries, A.colptr, A.rowval, A.nzval))

end

function SuperSparseMatrixLNK{Tv, Ti}(m, n) where {Tv, Ti <: Integer}
    SuperSparseMatrixLNK{Tv, Ti}(m, n, 0, n, zeros(Ti, n), zeros(Ti, n), zeros(Tv, n), zeros(Ti, n), 0)
end


function findindex(lnk::SuperSparseMatrixLNK, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end

    k = j
    k0 = j
    while k > 0
        if lnk.rowval[k] == i
            return k, 0
        end
        k0 = k
        k = lnk.colptr[k]
    end
    return 0, k0
end

"""
Return tuple containing size of the matrix.
"""
Base.size(lnk::SuperSparseMatrixLNK) = (lnk.m, lnk.n)

"""    
Return value stored for entry or zero if not found
"""
function Base.getindex(lnk::SuperSparseMatrixLNK{Tv, Ti}, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k == 0
        return zero(Tv)
    else
        return lnk.nzval[k]
    end
end

function addentry!(lnk::SuperSparseMatrixLNK, i, j, k, k0)
    # increase number of entries
    lnk.nentries += 1
    if length(lnk.nzval) < lnk.nentries
        newsize = Int(ceil(5.0 * lnk.nentries / 4.0))
        resize!(lnk.nzval, newsize)
        resize!(lnk.rowval, newsize)
        resize!(lnk.colptr, newsize)
    end

    # Append entry if not found
    lnk.rowval[lnk.nentries] = i

    # Shift the end of the list
    lnk.colptr[lnk.nentries] = 0
    lnk.colptr[k0] = lnk.nentries

    # Update number of nonzero entries
    lnk.nnz += 1
    return lnk.nentries
end

"""    
Update value of existing entry, otherwise extend matrix if v is nonzero.
"""
function Base.setindex!(lnk::SuperSparseMatrixLNK, v, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end

    # Set the first  column entry if it was not yet set.
    if lnk.rowval[j] == 0 && !iszero(v)
    	lnk.colctr             += 1
    	lnk.collnk[lnk.colctr] = j
        lnk.rowval[j]          = i
        lnk.nzval[j]           = v
        lnk.nnz                += 1
        return lnk
    end

    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = v
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk, i, j, k, k0)
        lnk.nzval[k] = v
    end
    return lnk
end

"""
Update element of the matrix  with operation `op`. 
It assumes that `op(0,0)==0`. If `v` is zero, no new 
entry is created.
"""
function updateindex!(lnk::SuperSparseMatrixLNK{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    # Set the first  column entry if it was not yet set.
    if lnk.rowval[j] == 0 && !iszero(v)
        lnk.colctr             += 1
    	lnk.collnk[lnk.colctr] = j
        lnk.rowval[j]          = i
        lnk.nzval[j]           = op(lnk.nzval[j], v)
        lnk.nnz                += 1
        return lnk
    end
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = op(lnk.nzval[k], v)
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk, i, j, k, k0)
        lnk.nzval[k] = op(zero(Tv), v)
    end
    lnk
end

mutable struct ColEntry{Tv, Ti <: Integer}
    rowval::Ti
    nzval::Tv
end

# Comparison method for sorting
Base.isless(x::ColEntry, y::ColEntry) = (x.rowval < y.rowval)

function get_column!(col::Vector{ColEntry{Tv, Ti}}, lnk::SuperSparseMatrixLNK{Tv, Ti}, j::Ti)::Ti where {Tv, Ti <: Integer}
	k = j
	ctr = zero(Ti)
	while k>0
		if abs(lnk.nzval[k]) > 0
			ctr += 1
			col[ctr] = ColEntry(lnk.rowval[k], lnk.nzval[k])
		end
		k = lnk.colptr[k]
	end
	sort!(col, 1, ctr, Base.QuickSort, Base.Forward)
	ctr
end


function remove_doubles!(col, coll)
	#input_ctr = 1
	last = 1
	for j=2:coll
		if col[j].rowval == col[last].rowval
			col[last].nzval += col[j].nzval
		else
			last += 1
			if last != j
				col[last] = col[j]
			end
		end
	end	
	last
end

function get_column!(col::Vector{ColEntry{Tv, Ti}}, lnks::Vector{SuperSparseMatrixLNK{Tv, Ti}}, js, tids, length)::Ti where {Tv, Ti <: Integer}
	ctr = zero(Ti)
	for i=1:length
		tid = tids[i]
		k   = js[i] 
		#for (tid,j) in zip(tids, js) #j0:j1
		#tid = tids[j]
		#k = j
		while k>0
			if abs(lnks[tid].nzval[k]) > 0
				ctr += 1
				col[ctr] = ColEntry(lnks[tid].rowval[k], lnks[tid].nzval[k])
			end
			k = lnks[tid].colptr[k]
		end
	end
	
	sort!(col, 1, ctr, Base.QuickSort, Base.Forward)
	ctr = remove_doubles!(col, ctr)
	#print_col(col, ctr)
	ctr

end

function merge_into!(rowval::Vector{Ti}, nzval::Vector{Tv}, C::SparseArrays.SparseMatrixCSC{Tv, Ti}, col::Vector{ColEntry{Tv, Ti}}, J::Ti, coll::Ti, ptr1::Ti) where {Tv, Ti <: Integer}
	j_min = 1
	numshifts = 0
	j_last = 0
	last_row = 0
	
	#@warn "MERGING $J"
	
	#rowval0 = copy(C.rowval[C.colptr[J]:C.colptr[J+1]-1])
	#endptr = C.colptr[J+1]
	
	for (di,i) in enumerate(C.colptr[J]:C.colptr[J+1]-1)
		for j=j_min:coll
			#if col[j].rowval == last_row
			#	#@info "!! col j rowval == last row"
			#end
			if col[j].rowval < C.rowval[i] #ptr1+di+numshifts] #i+numshifts]
				if col[j].rowval == last_row
					#@info "$(ptr1+di+numshifts) : backwards EQUALITY: "
					nzval[ptr1+di+numshifts] += col[j].nzval
				else
					#@info "$(ptr1+di+numshifts) : Insert from col: j=$j"
					#shift_e!(C.rowval, C.nzval, 1, i+numshifts, C.colptr[end]-1)
					rowval[ptr1+di+numshifts] = col[j].rowval
					nzval[ptr1+di+numshifts]  = col[j].nzval
					numshifts += 1
					#endptr += 1
				end
				j_last = j
			elseif col[j].rowval > C.rowval[i] #if col[j].rowval  
				#@info "$(ptr1+di+numshifts) : Insert from C: i=$i"
				rowval[ptr1+di+numshifts] = C.rowval[i]
				nzval[ptr1+di+numshifts]  = C.nzval[i]
				j_min = j
				break
			else
				#@info "$(ptr1+di+numshifts) : normal EQUALITY: i=$i, j=$j"
				rowval[ptr1+di+numshifts] = C.rowval[i]
				nzval[ptr1+di+numshifts]  = C.nzval[i]+col[j].nzval
				#numshifts += 1
				j_min = j+1
				j_last = j
				
				if j == coll
					#@info "$(ptr1+di+numshifts+1) → $(ptr1+numshifts+(C.colptr[J+1]-C.colptr[J]))"
					rowval[ptr1+di+numshifts+1:ptr1+numshifts+(C.colptr[J+1]-C.colptr[J])] = view(C.rowval, i+1:C.colptr[J+1]-1) #C.rowval[i:C.colptr[J+1]-1]
					nzval[ptr1+di+numshifts+1:ptr1+numshifts+(C.colptr[J+1]-C.colptr[J])]  = view(C.nzval, i+1:C.colptr[J+1]-1) #C.nzval[i:C.colptr[J+1]-1]
					
					#@info "FINISH"
					return numshifts
				end
				
				break
			end
			
			if j == coll
				#@info "$(ptr1+di+numshifts) → $(ptr1+numshifts+(C.colptr[J+1]-C.colptr[J]))"
				rowval[ptr1+di+numshifts:ptr1+numshifts+(C.colptr[J+1]-C.colptr[J])] = view(C.rowval, i:C.colptr[J+1]-1) #C.rowval[i:C.colptr[J+1]-1]
				nzval[ptr1+di+numshifts:ptr1+numshifts+(C.colptr[J+1]-C.colptr[J])]  = view(C.nzval, i:C.colptr[J+1]-1) #C.nzval[i:C.colptr[J+1]-1]
				
				#@info "FINISH"
				return numshifts
			end
			
			last_row = col[j].rowval
		end
	end
	endptr = ptr1 + numshifts + (C.colptr[J+1]-C.colptr[J])
	last_row = 0
	numshifts_old = numshifts
	numshifts = 0
	#start_ptr = endptr - 1 #C.colptr[J+1]-1
	if j_last > 0
		last_row = col[j_last].rowval
	end
	
	if j_last != coll
		for j=j_last+1:coll
			if col[j].rowval != last_row
				numshifts += 1
				#shift_e!(C.rowval, C.nzval, 1, start_ptr+numshifts, C.colptr[end]-1)
				#for k=start_ptr+numshifts:
				#@info "$(endptr+numshifts) : after..."
				rowval[endptr+numshifts] = col[j].rowval
				nzval[endptr+numshifts]  = col[j].nzval
				last_row                    = rowval[endptr+numshifts]
				#colptr[J+1:end]             .+= 1
			else
				nzval[endptr+numshifts] += col[j].nzval
			end
		end
	end
	
	return numshifts + numshifts_old

end


function print_col(col, coll)
	v = zeros((2, coll))
	for j=1:coll
		v[1,j] = col[j].rowval
		v[2,j] = col[j].nzval
	end
	@info v
end

function plus(lnk::SparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC) where {Tv, Ti <: Integer}
	if lnk.nnz == 0
		return csc
	elseif length(csc.rowval) == 0
		return SparseMatrixCSC(lnk)
	else
		return lnk + csc
	end
end

function plus(lnk::SuperSparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC) where {Tv, Ti <: Integer}
	gi = collect(1:csc.n)
	
	
	supersparsecolumns = gi[lnk.collnk[1:lnk.colctr]]
	sortedcolumnids    = sortperm(supersparsecolumns)
	sortedcolumns      = supersparsecolumns[sortedcolumnids]
	#sortedcolumns      = vcat([1], sortedcolumns)
	sortedcolumns      = vcat(sortedcolumns, [csc.n+1])
	
	col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:csc.m]
	
	#@info sortedcolumnids 
	
	nnz_sum   = length(csc.rowval) + lnk.nnz
	colptr    = Vector{Ti}(undef, csc.n+1)
	rowval    = Vector{Ti}(undef, nnz_sum)
	nzval     = Vector{Tv}(undef, nnz_sum)
	colptr[1] = one(Ti)
	
	#first part: columns between 1 and first column of lnk
	
	colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
	rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
	nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
	
	numshifts = 0
	
	for J=1:length(sortedcolumns)-1
		#@info ">>>>>>> $J <<<<<<<<<<<<<<<"
		# insert new added column here / dummy
		i = sortedcolumns[J]
		coll = get_column!(col, lnk, i)
		#print_col(col, coll)
		
		nns       = merge_into!(rowval, nzval, csc, col, i, coll, colptr[i]-1)
		
		numshifts += nns
		#j = colptr[i] #sortedcolumns[J]] 
		#rowval[j] = J
		#nzval[j]  = J
		# insertion end
		
		#colptr[i+1] = colptr[i] + csc.colptr[i+1]-csc.colptr[i] + numshifts
		
		#a = i+1
		#b = sortedcolumns[J+1]
		#@info a, b
		
		
		#colptr[i+1:sortedcolumns[J+1]] = (csc.colptr[i+1:sortedcolumns[J+1]]-csc.colptr[i:sortedcolumns[J+1]-1]).+(colptr[i] + nns)
		
		colptr[i+1:sortedcolumns[J+1]] = csc.colptr[i+1:sortedcolumns[J+1]].+(-csc.colptr[i]+colptr[i] + nns)
		
		
		rowval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1] = view(csc.rowval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
		nzval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1]  = view(csc.nzval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
		
		
		#=
		
		@info csc.colptr[a:b]
		
		colptr[a:b] = csc.colptr[a:b].+numshifts
		
		#colptr[i+2:sortedcolumns[J+1]] = csc.colptr[i+2:sortedcolumns[J+1]].+numshifts
		@info i, J, colptr[i+2], colptr[sortedcolumns[J+1]], csc.colptr[i+2], csc.colptr[sortedcolumns[J+1]]
		@info i, J, colptr[a], colptr[b], csc.colptr[a], csc.colptr[b]
		rowval[colptr[i+2]:colptr[sortedcolumns[J+1]]] = view(csc.rowval, csc.colptr[i+2]:csc.colptr[sortedcolumns[J+1]])
		nzval[colptr[i+2]:colptr[sortedcolumns[J+1]]]  = view(csc.nzval, csc.colptr[i+2]:csc.colptr[sortedcolumns[J+1]])
		#rowval[colptrsortedcolumns[J+1]]
		=#
	end
	
	#@info colptr
	
	resize!(rowval, length(csc.rowval)+numshifts)
	resize!(nzval, length(csc.rowval)+numshifts)
	
	
	SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)

		

end


function plus_remap(lnks::Vector{SuperSparseMatrixLNK{Tv, Ti}}, csc::SparseArrays.SparseMatrixCSC, gi::Vector{Vector{Ti}}) where {Tv, Ti <: Integer}
	nt = length(lnks)
	
	lnkscols           = vcat([lnks[i].collnk[1:lnks[i].colctr] for i=1:nt]...)
	supersparsecolumns = vcat([gi[i][lnks[i].collnk[1:lnks[i].colctr]] for i=1:nt]...)
	num_cols = sum([lnks[i].colctr for i=1:nt])
	tids     = Vector{Ti}(undef, num_cols)
	ctr = 0
	for i=1:nt
		for j=1:lnks[i].colctr
			ctr += 1
			tids[ctr] = i
		end
	end


	sortedcolumnids    = sortperm(supersparsecolumns)
	sortedcolumns      = supersparsecolumns[sortedcolumnids]
	sortedcolumns      = vcat(sortedcolumns, [Ti(csc.n+1)])
	
	coll = sum([lnks[i].nnz for i=1:nt])
	nnz_sum   = length(csc.rowval) + coll
	colptr    = Vector{Ti}(undef, csc.n+1)
	rowval    = Vector{Ti}(undef, nnz_sum)
	nzval     = Vector{Tv}(undef, nnz_sum)
	colptr[1] = one(Ti)
	
	if csc.m < coll
		coll = csc.m
	end
	
	col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:coll]
	numshifts = 0
	
	colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
	rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
	nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
	
	J = 1
	i0 = 0
	#lj_last = []
	#tid_last = []
	lj_last  = Vector{Ti}(undef, nt)
	tid_last = Vector{Ti}(undef, nt)
	ctr_last = 1
	gj_last = 0
	for J=1:length(sortedcolumns)-1
		gj_now  = sortedcolumns[J]
		gj_next = sortedcolumns[J+1]
			
		lj_last[ctr_last]  = lnkscols[sortedcolumnids[J]]
		tid_last[ctr_last] = tids[sortedcolumnids[J]]
		
		if gj_now != gj_next
			#@info typeof(lnks)
			# do stuff from gj_last to gj_now / from last_lj to J
			#@info lj_last, tid_last
			coll = get_column!(col, lnks, lj_last, tid_last, ctr_last)
			
			nns       = merge_into!(rowval, nzval, csc, col, gj_now, coll, colptr[gj_now]-one(Ti))
			numshifts += nns
			
			
			colptr[gj_now+1:sortedcolumns[J+1]] = csc.colptr[gj_now+1:sortedcolumns[J+1]].+(-csc.colptr[gj_now]+colptr[gj_now] + nns)
			
			rowval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1] = view(csc.rowval, csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1)
			nzval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1]  = view(csc.nzval, csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1)
			
			#rowval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1] = csc.rowval[csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1]
			#nzval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1]  = csc.nzval[csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1]
			
			
			#for k=csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1
			#	k2 = k+(-csc.colptr[gj_now]+colptr[gj_now] + nns)
			#	rowval[k2] = csc.rowval[k]
			#	nzval[k2]  = csc.nzval[k]
			#end	
			
			gj_last  = gj_now
			ctr_last = 0 #tids[sortedcolumnids[J]]]
		end
		
		ctr_last += 1
		
		
	end
		
	
	resize!(rowval, length(csc.rowval)+numshifts)
	resize!(nzval, length(csc.rowval)+numshifts)
	
	
	SparseArrays.SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)
	

	#for ...
	#	take many columns together if necessary in `get_column`
	#end



end


function plus_remap_time(lnks::Vector{SuperSparseMatrixLNK{Tv, Ti}}, csc::SparseArrays.SparseMatrixCSC, gi::Vector{Vector{Ti}}) where {Tv, Ti <: Integer}
	times = zeros(8)
	insertion_ctr = 0
	times[1] = @elapsed begin
		nt = length(lnks)
		lnkscols           = vcat([lnks[i].collnk[1:lnks[i].colctr] for i=1:nt]...)
		supersparsecolumns = vcat([gi[i][lnks[i].collnk[1:lnks[i].colctr]] for i=1:nt]...)
		num_cols = sum([lnks[i].colctr for i=1:nt])
		tids     = Vector{Ti}(undef, num_cols)
		ctr = 0
		for i=1:nt
			for j=1:lnks[i].colctr
				ctr += 1
				tids[ctr] = i
			end
		end


		sortedcolumnids    = sortperm(supersparsecolumns)
		sortedcolumns      = supersparsecolumns[sortedcolumnids]
		sortedcolumns      = vcat(sortedcolumns, [Ti(csc.n+1)])
		
		coll = sum([lnks[i].nnz for i=1:nt])
		nnz_sum   = length(csc.rowval) + coll
		colptr    = Vector{Ti}(undef, csc.n+1)
		rowval    = Vector{Ti}(undef, nnz_sum)
		nzval     = Vector{Tv}(undef, nnz_sum)
		colptr[1] = one(Ti)
		
		if csc.m < coll
			coll = csc.m
		end
		
		col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:coll]
		numshifts = 0
		
		colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
		rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
		nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
		
		J = 1
		i0 = 0
		lj_last  = Vector{Ti}(undef, nt)
		tid_last = Vector{Ti}(undef, nt)
		ctr_last = 1
		
		#lj_last = []
		#tid_last = []
		gj_last = 0
	end
	
	for J=1:length(sortedcolumns)-1
		times[2] += @elapsed begin
			gj_now  = sortedcolumns[J]
			gj_next = sortedcolumns[J+1]
			
			#@info "add : lj = $(lnkscols[sortedcolumnids[J]]), tid = $(tids[sortedcolumnids[J]])"
			lj_last[ctr_last]  = lnkscols[sortedcolumnids[J]]
			tid_last[ctr_last] = tids[sortedcolumnids[J]]
		end		
		
		if gj_now != gj_next
			times[3] += @elapsed (coll = get_column!(col, lnks, lj_last, tid_last, ctr_last))
			
			times[4] += @elapsed (nns = merge_into!(rowval, nzval, csc, col, gj_now, coll, colptr[gj_now]-one(Ti)))
		
			numshifts += nns
			
			
			times[5] += @elapsed (colptr[gj_now+1:sortedcolumns[J+1]] = csc.colptr[gj_now+1:sortedcolumns[J+1]].+(-csc.colptr[gj_now]+colptr[gj_now] + nns))
			
			insertion_ctr += 1
			times[6] += @elapsed begin
				rowval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1] = view(csc.rowval, csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1)
				nzval[colptr[gj_now+1]:colptr[sortedcolumns[J+1]]-1]  = view(csc.nzval, csc.colptr[gj_now+1]:csc.colptr[sortedcolumns[J+1]]-1)
			end		
			
			times[7] = @elapsed begin
				gj_last  = gj_now
				#@info "RESET"
				ctr_last = 0
				#lj_last  = [] #lnkscols[sortedcolumnids[J]]]
				#tid_last = [] #tids[sortedcolumnids[J]]]
			end
		end
		
		ctr_last += 1
		
		
		
	end
		
	
	times[8] = @elapsed begin
		resize!(rowval, length(csc.rowval)+numshifts)
		resize!(nzval, length(csc.rowval)+numshifts)
		
		D = SparseArrays.SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)
	end
	@info sum(times), times
	D, insertion_ctr
end


function plus_remap(lnk::SuperSparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC, gi::Vector{Ti}) where {Tv, Ti <: Integer}

	#@info lnk.collnk[1:lnk.colctr]
	
	
	supersparsecolumns = gi[lnk.collnk[1:lnk.colctr]]
	sortedcolumnids    = sortperm(supersparsecolumns)
	sortedcolumns      = supersparsecolumns[sortedcolumnids]
	#sortedcolumns      = vcat([1], sortedcolumns)
	#@info typeof(supersparsecolumns), typeof(sortedcolumns)
	
	sortedcolumns      = vcat(sortedcolumns, [Ti(csc.n+1)])
	
	#@info typeof(supersparsecolumns), typeof(sortedcolumns)
	
	
	#@info supersparsecolumns
	#@info sortedcolumns
	#@info lnk.collnk[1:length(sortedcolumns)-1]
	#@info lnk.collnk[sortedcolumnids[1:length(sortedcolumns)-1]]
	
	col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:csc.m]
	
	#@info sortedcolumnids 
	
	nnz_sum   = length(csc.rowval) + lnk.nnz
	colptr    = Vector{Ti}(undef, csc.n+1)
	rowval    = Vector{Ti}(undef, nnz_sum)
	nzval     = Vector{Tv}(undef, nnz_sum)
	colptr[1] = one(Ti)
	
	#first part: columns between 1 and first column of lnk
	
	colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
	rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
	nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
	
	numshifts = 0
	
	for J=1:length(sortedcolumns)-1
		i = sortedcolumns[J]
		
		coll = get_column!(col, lnk, lnk.collnk[sortedcolumnids[J]] )
		#@info typeof(i), typeof(coll), typeof(colptr), typeof(colptr[i]), typeof(colptr[i]-1)
		nns       = merge_into!(rowval, nzval, csc, col, i, coll, colptr[i]-one(Ti))
		numshifts += nns
		
		
		colptr[i+1:sortedcolumns[J+1]] = csc.colptr[i+1:sortedcolumns[J+1]].+(-csc.colptr[i]+colptr[i] + nns)
		rowval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1] = view(csc.rowval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
		nzval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1]  = view(csc.nzval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
		
		#=
		for k=csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1
			k2 = k+(-csc.colptr[i]+colptr[i] + nns)
			rowval[k2] = csc.rowval[k]
			nzval[k2]  = csc.nzval[k]
		end
		=#
	end
	
	
	resize!(rowval, length(csc.rowval)+numshifts)
	resize!(nzval, length(csc.rowval)+numshifts)
	
	
	SparseArrays.SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)

end




function plus_loop(lnk::SuperSparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC) where {Tv, Ti <: Integer}
	gi = collect(1:csc.n)
	
	supersparsecolumns = gi[lnk.collnk[1:lnk.colctr]]
	sortedcolumnids    = sortperm(supersparsecolumns)
	sortedcolumns      = supersparsecolumns[sortedcolumnids]
	#sortedcolumns      = vcat([1], sortedcolumns)
	sortedcolumns      = vcat(sortedcolumns, [csc.n+1])
	
	col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:csc.m]
	
	#@info sortedcolumnids 
	
	nnz_sum   = length(csc.rowval) + lnk.nnz
	colptr    = Vector{Ti}(undef, csc.n+1)
	rowval    = Vector{Ti}(undef, nnz_sum)
	nzval     = Vector{Tv}(undef, nnz_sum)
	colptr[1] = one(Ti)
	
	#first part: columns between 1 and first column of lnk
	
	colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
	rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
	nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
	
	numshifts = 0
	
	for J=1:length(sortedcolumns)-1
		i = sortedcolumns[J]
		coll = get_column!(col, lnk, i)
		
		nns       = merge_into!(rowval, nzval, csc, col, i, coll, colptr[i]-1)
		numshifts += nns
		
		colptr[i+1:sortedcolumns[J+1]] = csc.colptr[i+1:sortedcolumns[J+1]].+(-csc.colptr[i]+colptr[i] + nns)
		
		
		for k=csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1
			k2 = k+(-csc.colptr[i]+colptr[i] + nns)
			rowval[k2] = csc.rowval[k]
			nzval[k2]  = csc.nzval[k]
		end
		
		
	end
	
	#@info colptr
	
	resize!(rowval, length(csc.rowval)+numshifts)
	resize!(nzval, length(csc.rowval)+numshifts)
	
	
	SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)

		

end


function plus_loop_time(lnk::SuperSparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
	t = zeros(8)
	
	t[1] = @elapsed begin
		gi = collect(1:csc.n)
		supersparsecolumns = gi[lnk.collnk[1:lnk.colctr]]
		sortedcolumnids    = sortperm(supersparsecolumns)
		sortedcolumns      = supersparsecolumns[sortedcolumnids]
		sortedcolumns      = vcat(sortedcolumns, [csc.n+1])
		
		col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:csc.m]
		
		nnz_sum   = length(csc.rowval) + lnk.nnz
		colptr    = Vector{Ti}(undef, csc.n+1)
		rowval    = Vector{Ti}(undef, nnz_sum)
		nzval     = Vector{Tv}(undef, nnz_sum)
		colptr[1] = one(Ti)
		
		colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
		rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
		nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
		
		numshifts = 0
	end
	
	for J=1:length(sortedcolumns)-1
		t[2] += @elapsed (i = sortedcolumns[J])
		t[3] += @elapsed (coll = get_column!(col, lnk, i))
		
		t[4] += @elapsed (nns = merge_into!(rowval, nzval, csc, col, i, coll, colptr[i]-1))
		t[5] += @elapsed (numshifts += nns)
		
		t[6] += @elapsed (colptr[i+1:sortedcolumns[J+1]] = csc.colptr[i+1:sortedcolumns[J+1]].+(-csc.colptr[i]+colptr[i] + nns))
		
		t[7] += @elapsed for k=csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1
			k2 = k+(-csc.colptr[i]+colptr[i] + nns)
			rowval[k2] = csc.rowval[k]
			nzval[k2]  = csc.nzval[k]
		end
	end
	
	t[8] = @elapsed begin
		resize!(rowval, length(csc.rowval)+numshifts)
		resize!(nzval, length(csc.rowval)+numshifts)
		SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)
	end
	t
end

function plus_time(lnk::SuperSparseMatrixLNK{Tv, Ti}, csc::SparseArrays.SparseMatrixCSC) where {Tv, Ti <: Integer}
	t = zeros(8)
	
	t[1] = @elapsed begin
		gi = collect(1:csc.n)
		supersparsecolumns = gi[lnk.collnk[1:lnk.colctr]]
		sortedcolumnids    = sortperm(supersparsecolumns)
		sortedcolumns      = supersparsecolumns[sortedcolumnids]
		sortedcolumns      = vcat(sortedcolumns, [csc.n+1])
		
		col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i=1:csc.m]
		
		nnz_sum   = length(csc.rowval) + lnk.nnz
		colptr    = Vector{Ti}(undef, csc.n+1)
		rowval    = Vector{Ti}(undef, nnz_sum)
		nzval     = Vector{Tv}(undef, nnz_sum)
		colptr[1] = one(Ti)
		
		colptr[1:sortedcolumns[1]] = view(csc.colptr, 1:sortedcolumns[1])
		rowval[1:csc.colptr[sortedcolumns[1]]-1] = view(csc.rowval, 1:csc.colptr[sortedcolumns[1]]-1)
		nzval[1:csc.colptr[sortedcolumns[1]]-1]  = view(csc.nzval, 1:csc.colptr[sortedcolumns[1]]-1)
		
		numshifts = 0
	end
	
	for J=1:length(sortedcolumns)-1
		t[2] += @elapsed (i = sortedcolumns[J])
		t[3] += @elapsed (coll = get_column!(col, lnk, i))
		
		t[4] += @elapsed (nns = merge_into!(rowval, nzval, csc, col, i, coll, colptr[i]-1))
		t[5] += @elapsed (numshifts += nns)
		
		t[6] += @elapsed (colptr[i+1:sortedcolumns[J+1]] = csc.colptr[i+1:sortedcolumns[J+1]].+(-csc.colptr[i]+colptr[i] + nns))
		
		t[7] += @elapsed begin
			#rowval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1] = csc.rowval[csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1]
			#nzval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1]  = csc.nzval[csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1]
			
			rowval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1] = view(csc.rowval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
			nzval[colptr[i+1]:colptr[sortedcolumns[J+1]]-1]  = view(csc.nzval, csc.colptr[i+1]:csc.colptr[sortedcolumns[J+1]]-1)
						
		end
	end
	
	t[8] = @elapsed begin
		resize!(rowval, length(csc.rowval)+numshifts)
		resize!(nzval, length(csc.rowval)+numshifts)
		SparseMatrixCSC(csc.m, csc.n, colptr, rowval, nzval)
	end
	t
end


function test_plus()
	#C = sprand(10, 10, 0.5)
	C = SparseMatrixLNK{Float64, Int64}(10,10)
	C[2,1] = -1.0
	C[7,1] = -1.0
	C[9,1] = -1.0
	
	C[3,1] = -1.0
	
	
	C[2,2] = -1.0
	C[7,2] = -1.0
	C[9,2] = -1.0
	
	C[1,4] = -1.0
	C[2,4] = -1.0
	C[3,4] = -1.0
	
	C[1,7] = -1.0
	C[2,7] = -1.0
	C[3,7] = -1.0
	
	C[2,9] = -1.0
	C[5,9] = -1.0
	C[8,9] = -1.0
	
	C = SparseMatrixCSC(C)
	
	A = SuperSparseMatrixLNK{Float64, Int64}(10,10)
	A[1,1] = 1.0
	A[3,1] = 1.0
	A[5,1] = 1.0
	
	A[5,3] = 1.0
	
	A[9,4] = 1.0
	A[4,4] = 1.0
	
	A[8,8] = 1.0
	A[10,10] = 1.0
	
	gi = collect(1:10)
	
	#@info C
	#@info A
	
	AC = SparseArrays.SparseMatrixCSC(A)
	
	S = (plus_remap(A, C, gi))
	
	count_doubles_CSC(C)
	
	count_doubles_CSC(AC)
	
	count_doubles_CSC(S)
	
	@info "S's zeros: ", count_zeros(S.rowval)
	
	
	C, AC, S #dropzeros!(SparseMatrixCSC(A)), S
end


function test_plus()
	#C = sprand(10, 10, 0.5)
	C = SparseMatrixLNK{Float64, Int64}(10,10)
	C[2,1] = -1.0
	C[7,1] = -1.0
	C[9,1] = -1.0
	
	C[3,1] = -1.0
	
	
	C[2,2] = -1.0
	C[7,2] = -1.0
	C[9,2] = -1.0
	
	C[1,4] = -1.0
	C[2,4] = -1.0
	C[3,4] = -1.0
	
	C[1,7] = -1.0
	C[2,7] = -1.0
	C[3,7] = -1.0
	
	C[2,9] = -1.0
	C[5,9] = -1.0
	C[8,9] = -1.0
	
	C = SparseMatrixCSC(C)
	
	A = SuperSparseMatrixLNK{Float64, Int64}(10,10)
	A[1,1] = 1.0
	A[3,1] = 1.0
	A[5,1] = 1.0
	
	A[5,3] = 1.0
	
	A[9,4] = 1.0
	A[4,4] = 1.0
	
	A[8,8] = 1.0
	A[10,10] = 1.0
	
	gi = collect(1:10)
	
	#@info C
	#@info A
	
	AC = SparseArrays.SparseMatrixCSC(A)
	
	S = (plus_remap(A, C, gi))
	
	count_doubles_CSC(C)
	
	count_doubles_CSC(AC)
	
	count_doubles_CSC(S)
	
	@info "S's zeros: ", count_zeros(S.rowval)
	
	
	C, AC, S #dropzeros!(SparseMatrixCSC(A)), S
end


function LNKs()
	A = SuperSparseMatrixLNK{Float64, Int64}(8,5)
	#A[1,1] = 1.0
	#A[3,1] = 1.0
	#A[5,1] = 1.0
	
	#A[5,3] = 1.0
	
	A[8,5] = 1.0
	A[2,5] = 1.0
	#A[7,5] = 1.0
	#A[4,5] = 1.0
	
	
	B = SuperSparseMatrixLNK{Float64, Int64}(8,4)
	B[2,1] = 1.0
	#B[3,1] = 1.0
	#B[5,1] = 1.0

	#B[5,3] = 1.0
	
	#B[7,4] = 1.0
	#B[4,4] = 1.0

	A, B
end

function test_plus3(C)
	#C = sprand(8,8,0.3)
	giA = [1,2,3,4,5] # [5,4,3,2,1]
	giB = [5,6,7,8] # [5,7,6,8]
	
	A, B = LNKs()
	
	S1 = plus_remap(A, C, giA)
	S2 = plus_remap(B, S1, giB)
	
	A, B, C, S1, S2
end
	
	

function test_plus4()
	C = sprand(8,8,0.3)
	
	giA = [1,2,3,4,5] # [5,4,3,2,1]
	giB = [5,6,7,8] # [5,7,6,8]
	
	gi = [giA, giB]
	
	
	A, B = LNKs()
	
	lnks = Vector{SuperSparseMatrixLNK{Float64, Int64}}(undef, 2)
	lnks[1] = A
	lnks[2] = B
	#@info typeof(lnks)
	
	S = plus_remap(lnks, C, gi)
	
	#S1 = plus_remap(A, C, giA)
	#S2 = plus_remap(B, S1, giB)
	
	A, B, C, S #1, S2
end

function compare34()
	a, b, c, s = test_plus4()
	A, B, C, S1, S2 = test_plus3(c)
	
	S2-s
end
	
	

function test_plus2()
	#C = sprand(10, 10, 0.5)
	C = SparseMatrixLNK{Float64, Int64}(10,10)
	C[2,1] = -1.0
	C[7,1] = -1.0
	C[9,1] = -1.0
	
	C[3,1] = -1.0
	
	
	C[2,2] = -1.0
	C[7,2] = -1.0
	C[9,2] = -1.0
	
	C[1,4] = -1.0
	C[2,4] = -1.0
	C[3,4] = -1.0
	
	C[1,7] = -1.0
	C[2,7] = -1.0
	C[3,7] = -1.0
	
	C[2,9] = -1.0
	C[5,9] = -1.0
	C[8,9] = -1.0
	
	C = SparseMatrixCSC(C)
	
	A = SuperSparseMatrixLNK{Float64, Int64}(10,10)
	A[1,1] = 1.0
	A[3,1] = 1.0
	A[5,1] = 1.0
	
	A[5,3] = 1.0
	
	A[9,4] = 1.0
	A[4,4] = 1.0
	
	#A[8,8] = 1.0
	#A[10,10] = 1.0
	
	gi = [5,4,3,2,1]  #collect(1:10)
	
	#@info C
	#@info A
	
	AC = SparseArrays.SparseMatrixCSC(A)
	
	S = (plus_remap(A, C, gi))
	
	S2 = (plus_remap(A, S, gi))
	
	count_doubles_CSC(C)
	
	count_doubles_CSC(AC)
	
	count_doubles_CSC(S)
	
	count_doubles_CSC(S2)
	
	
	@info "S 's zeros: ", count_zeros(S.rowval)
	
	@info "S2's zeros: ", count_zeros(S2.rowval)
	
	C, AC, S, S2 #dropzeros!(SparseMatrixCSC(A)), S
end

function twodisjointsets(n, k)
	A = rand(1:n, k)
	B = zeros(Int64, k)
	done = false
	ctr = 0
	while ctr != k
		v = rand(1:n)
		if !(v in A)
			ctr += 1
			B[ctr] = v
		end
	end

	A, B
end

function distinct(x, n)
	y = zeros(typeof(x[1]), n)
	ctr = 0
	while ctr != n
		v = rand(x)
		if !(v in y[1:ctr])
			ctr += 1
			y[ctr] = v
		end
	end
	y
end 

function assembly(n, nnzcol, supercolnum)
	js = collect(1:n)
	sjs = distinct(js, supercolnum) #rand(js, supercolnum)
	#@info sjs
	
	Super  = SuperSparseMatrixLNK{Float64, Int64}(n, n) 
	Normal = SparseMatrixLNK{Float64, Int64}(n, n)
	Basis  = SparseMatrixLNK{Float64, Int64}(n, n)
	
	for j in js
		if j in sjs
			rows1, rows2 = twodisjointsets(n, nnzcol)
			for row in rows1
				Basis[row,j] += 1.0
			end
			for row in rows2
				Super[row,j]  += 1.0
				Normal[row,j] += 1.0
			end
		else
			rows = rand(1:n, 2*nnzcol)
			for row in rows
				#Super[row,j]  += 1.0
				#Normal[row,j] += 1.0
				Basis[row,j]  += 1.0
			end
		end
		
	end
		
	Super, Normal, SparseMatrixCSC(Basis)
	
end

function mean(x)
	sum(x)/length(x)
end

function form(x)
	[minimum(x), mean(x), maximum(x)]
end

function benchmark(n, nnzcol, supercolnum, num)
	
	#validation
	S, N, B = assembly(n, nnzcol, supercolnum)
	RS = plus(S, B)
	RN = plus(N, B)
	
	@info length((RS-RN).nzval)
	
	if num > 0
		
		t_s = zeros(num)
		t_l = zeros(num)
		t_n = zeros(num)
		
		for i=1:num
			S, N, B = assembly(n, nnzcol, supercolnum)
			t_s[i] = @elapsed plus(S, B)
			#t_l[i] = @elapsed plus_loop(S, B)
			#t_n[i] = @elapsed plus(N, B)
			GC.gc()
		end
		
		@info "Super : $(form(t_s))"# $(minimum(t_s))  max: $(maximum(t_s))"
		
		for i=1:num
			S, N, B = assembly(n, nnzcol, supercolnum)
			#t_s[i] = @elapsed plus(S, B)
			t_l[i] = @elapsed plus_loop(S, B)
			#t_n[i] = @elapsed plus(N, B)
			GC.gc()
		end
		
		@info "SuLoop: $(form(t_l))"# $(minimum(t_l))  max: $(maximum(t_l))"
		
		
		for i=1:num
			S, N, B = assembly(n, nnzcol, supercolnum)
			#t_s[i] = @elapsed plus(S, B)
			#t_l[i] = @elapsed plus_loop(S, B)
			t_n[i] = @elapsed plus(N, B)
			GC.gc()
		end
		
		@info "Normal: $(form(t_n))"# $(minimum(t_n))  max: $(maximum(t_n))"
	
	elseif num == 0
		S, N, B = assembly(n, nnzcol, supercolnum)
		t = plus_time(S, B)
		@info t, sum(t)
	
	elseif num == -1
		S, N, B = assembly(n, nnzcol, supercolnum)
		RS = plus_loop(S, B)
		RN = plus(N, B)
		
		@info length((RS-RN).nzval)
		
		S, N, B = assembly(n, nnzcol, supercolnum)
		t = plus_loop_time(S, B)
		@info t, sum(t)
	
	elseif num == -2
		S, N, B = assembly(n, nnzcol, supercolnum)
		RSl = plus_loop(S, B)
		RSv = plus(S, B)
		RN = plus(N, B)
		
		@info length((RSl-RN).nzval), length((RSv-RN).nzval)
		
		S, N, B = assembly(n, nnzcol, supercolnum)
		tl = plus_loop_time(S, B)
		tv = plus_time(S, B)
		@info "LOOP:"
		@info tl, sum(tl)
		@info "VIEW:"
		@info tv, sum(tv)
		
	else
		num *= -1
		S, N, B = assembly(n, nnzcol, supercolnum)
		RSl = plus_loop(S, B)
		RSv = plus(S, B)
		RN = plus(N, B)
		
		@info length((RSl-RN).nzval), length((RSv-RN).nzval)
		
		for i=1:num
			@info "----------------------------"
			S, N, B = assembly(n, nnzcol, supercolnum)
			tl = plus_loop_time(S, B)
			tv = plus_time(S, B)
			@info "LOOP:"
			@info tl, sum(tl)
			@info "VIEW:"
			@info tv, sum(tv)
		end
	end

end



#=

function give()
	#C = sprand(10, 10, 0.5)
	C = SparseMatrixLNK{Float64, Int64}(10,10)
	C[2,1] = -1.0
	C[7,1] = -1.0
	C[9,1] = -1.0
	
	C[2,2] = -1.0
	C[7,2] = -1.0
	C[9,2] = -1.0
	
	C[5,4] = -1.0
	C[7,4] = -1.0
	C[9,4] = -1.0
	
	C[1,7] = -1.0
	C[2,7] = -1.0
	C[3,7] = -1.0
	
	C[2,9] = -1.0
	C[5,9] = -1.0
	C[8,9] = -1.0
	
	C = SparseMatrixCSC(C)
	
	
	A = SuperSparseMatrixLNK{Float64, Int64}(10,10)
	A[1,1] = 1.0
	A[2,1] = 1.0
	
	A[5,3] = 1.0
	A[5,3] = 1.0 
	
	A[4,4] = 1.0
	A[1,4] = 1.0
	
	A[8,8] = 1.0
	
	
	C, A #, S
end
=#



function count_doubles_CSC(A)
	m = length(A.colptr)-1 # number of columns
	
	@info "---------------------→ counting doubles now"
	
	ctr = 0
	for j=1:m
		last = 0
		for i=A.colptr[j]:A.colptr[j+1]-1
			if A.rowval[i] == last
				@info "DOUBLE at i=$i"
				ctr += 1
			else
				last = A.rowval[i]
			end
		end
	end
		
	ctr

end	


function count_zeros(x)
	length(findall(y->y==0,x))
end










