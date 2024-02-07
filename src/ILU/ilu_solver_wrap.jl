function ILUZero_solve(ILU, b)
	yy = copy(b)
	forward_substitution!(yy, ILU, b)
	zz = copy(b)
	backward_substitution!(zz, ILU, yy)
	zz
end

function ILUZero_solve_time(ILU, b)
	time = zeros(4)
	time[1] = @elapsed (yy = copy(b))
	time[2] = @elapsed (forward_substitution!(yy, ILU, b))
	time[3] = @elapsed (zz = copy(b))
	time[4] = @elapsed (backward_substitution!(zz, ILU, yy))
	@info time
end



function MA_solve_seq(ILU::ILUPrecon, b)
	y = copy(b)
	forward_subst!(y, b, ILU)
	z = copy(b)
	backward_subst!(z, y, ILU)
	z
end

function MA_solve_seq_old(ILU::ILUPrecon, b)
	nzval = ILU.nzval
	diag = ILU.diag
	A = ILU.A
	y = copy(b)
	forward_subst_old!(y, b, nzval, diag, A)
	z = copy(b)
	backward_subst_old!(z, y, nzval, diag, A)
	z
end

function MA_solve_seq_old_time(ILU::ILUPrecon, b)
	time = zeros(5)
	time[1] = @elapsed begin
		nzval = ILU.nzval
		diag = ILU.diag
		A = ILU.A
	end
	
	time[2] = @elapsed (y = copy(b))
	time[3] = @elapsed (forward_subst_old!(y, b, nzval, diag, A))
	time[4] = @elapsed (z = copy(b))
	time[5] = @elapsed (backward_subst_old!(z, y, nzval, diag, A))
	@info time
end

function MA_solve_seq_time(ILU::ILUPrecon, b)
	@info "----------"
	y = copy(b)
	forward_subst_time!(y, b, ILU)
	z = copy(b)
	backward_subst_time!(z, y, ILU)
	z
end


function MA_solve_seq_time_old(ILU::ILUPrecon, b)
	@info "----------"
	y = copy(b)
	nzval = ILU.nzval
	diag = ILU.diag
	A = ILU.A
	forward_subst_time_old!(y, b, nzval, diag, A)
	z = copy(b)
	backward_subst_time!(z, y, ILU)
	z
end

function MA_solve_par(PILU::PILUPrecon, b)
	yp = copy(b)
	forward_subst!(yp, b, PILU)
	zp = copy(b)
	backward_subst!(zp, yp, PILU)
	zp
end

function MA_solve_par_old(PILU::PILUPrecon, b)
	nzval = PILU.nzval
	diag = PILU.diag
	start = PILU.start
	nt = PILU.nt
	depth = PILU.depth
	A = PILU.A
	
	yp = copy(b)
	forward_subst_old!(yp, b, nzval, diag, start, nt, depth, A)
	zp = copy(b)
	backward_subst_old!(zp, yp, nzval, diag, start, nt, depth, A)
	zp
end


#=
function MA_solve_seq(ILU, CSC, diag, b)
	y = copy(b)
	forward_subst!(y, ILU, b, diag, CSC)
	z = copy(b)
	backward_subst!(z, ILU, y, diag, CSC)
	z
end

function MA_solve_par(ILU, CSC, diag, b, starts, nt, depth)
	yp = copy(b)
	forward_subst!(yp, ILU, b, diag, starts, nt, depth, CSC)
	zp = copy(b)
	backward_subst!(zp, ILU, yp, diag, starts, nt, depth, CSC)
	zp
end
=#


