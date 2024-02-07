# https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122100X03628/1-s2.0-S0898122103001548/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECwaCXVzLWVhc3QtMSJIMEYCIQDqZI2d3zAPzu3jzKCUJcbdIFHESmBDMLNELM5MWIFU9AIhALrzOzG2r9PCDybVhvtMcglK%2BSZpJHhb91QjfGyB4UW2KrsFCNX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwTtvmR2IuTp5k%2BLjgqjwXwDV%2FLpK0b64hCiVEVwfAAP%2Fkem4b7eZKhw%2B%2BJ81H1DvSKGTm1ggcwTWOBLeq56EsHbIMsSmaVHh4GhG1BFgSWxOMPvvtu0me22ybKEEHG2q67qHKPAh8SgEITQzBA17pm3FtAElxSbA5odA70qbrpKaewueWXOGrwV0H5lFtZ7vd1bDCznKzcDeiUacro7nQWRFVrOeHb2aNB1okc%2BzKiDyESWyNiFcAgBi4sZy9RlCCBROvkY8KB5I4%2FjImfy60B8A%2BezZoQmC7s0MF10BoffQwhHPgWmrueTkWpeqm7eyPTWtdH2y9%2B0JuKU3TCGBmrvSdZ1OfnIPJ8K1A%2FUmi5Olc1o%2BnkphIv66OB1RkPLSOEyB0ComgrilymL56YDQdIFq4dLAwK6iGORSuU2XQx%2B5clOQDAemfHNo21lkvq6CwbYt0o7MYUaJPcBrpxbP8vUqifWmXBYKmfcq%2FfdmtftvN%2BJBNshsJoxrsHySqrguPCAws%2ByQCDEQWhWR0ut6SYK%2FsKWdWYWvTDTmiKdcodDlACLem1oad%2FoQT54TRpx8RpK8eRM7PveCnchyc3TEMBCZa6AqPNqJF4JT%2FCPYx4icob3BAxgjKObGmnXo20xU8lu461ppqOBczBhGzAdrykBKx%2B7TJnSw4WDZNvzgMmBVNfpwzTuUmW7AxoP%2FtYxrm%2BcC5Tu5OZ1gLby0MgvpkPi%2BmzWnki15c%2BQPl54v4jxwF2oPRvFyU1L5JZr0ddd36BAcmjkurXEXn0RpzhO6u%2FQDUcXTU%2BNFiS7Az83yK%2ByqIfeSJu8jFnlJIpWl1hjo8AMbAGpdbuMAsxGqta%2FPJNdqlKZyHisG6spS85H6jDGf7nk3DnqK%2FLLKHgJQRRMKKhpK0GOrABd3QZ05XSUTqmSJGws8H1PVC8a5SeiyV5yWdu0XQ3KDAZMjnL%2BEiGM4ixXhvFQa1DR5PpSUuHTITuBv2o0xvEBLKDuNXB6aXiiDxPTxZnp7%2FygmYKYvct8eawnU%2BukNTwHhSmJIlibTb4EXw80tBVUd4Xnb19XzrwAkpA5FAftkWaaG%2BhIQmk7uZqpCR39dEW13T%2BWyZsd2Sw61nzvIBkaircl29%2Fpi8o%2BPZ1rNVTB7E%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240118T130827Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRSSK5WKL%2F20240118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6e74de2553b2759260437456edb10c29d3d8ed54b5fdd6461ab93b18bf547e1e&hash=adf84da67f1c362b7388afb179c4f4636b86c5e614274a2fb83b25bff460e0b1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0898122103001548&tid=spdf-26410b0a-8f71-4078-a302-8cd53dd990fb&sid=7943cffa85ea284bc48ba2422171a4974b6egxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e03575355535f5355520f&rr=84771116afb36a76&cc=de


using SparseArrays
using ILUZero
using LinearAlgebra

#include("test.jl")

mutable struct ILUPrecon

	nzval::AbstractVector
	diag::AbstractVector
	A::AbstractMatrix
	

end


"""
Calculationg L and U using the aforementioned paper of a matrix A.
It needs to have all the diagonal entries.
"""
function create_ILU(A::ExtendableSparseMatrixParallel{Tv,Ti}) where {Tv, Ti <:Integer}
	#ILU = copy(A)
	nzval = copy(A.cscmatrix.nzval)
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
	#nzval  = ILU.nzval
	n = A.cscmatrix.n # number of columns
	point = zeros(Ti, n) #Vector{Ti}(undef, n)
	diag  = Vector{Ti}(undef, n)
	
	# find diagonal entries
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1
			if rowval[v] == j
				diag[j] = v
				break
			end
			#elseif rowval[v] 
		end
	end
	
	# compute L and U
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1  ## start at colptr[j]+1 ??
			point[rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			#nzval[v] /= nzval[diag[i]]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		
		for v=colptr[j]:colptr[j+1]-1
			point[rowval[v]] = zero(Ti)
		end
	end
	#nzval, diag
	ILUPrecon(nzval, diag, A.cscmatrix)
end



"""
Solving Ly = v for the lower-triangular matrix L `encoded` in ILU.
"""
function forward_subst!(y, v, ilu::ILUPrecon)
	nzval = ilu.nzval
	n = ilu.A.n
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	y .= 0
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end

function forward_subst_time!(y, v, ilu::ILUPrecon)
	t = @elapsed begin
		nzval = ilu.nzval
		n = ilu.A.n
		colptr = ilu.A.colptr
		rowval = ilu.A.rowval
		diag = ilu.diag
		y .= 0
	end
	@info t
	t = @elapsed @inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	@info t
	
end

function forward_subst_time_old!(y, v, nzval, diag, A)
	t = @elapsed begin
		n = A.n
		colptr = A.colptr
		rowval = A.rowval
		y .= 0
	end
	@info t
	t = @elapsed @inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	@info t
	
end

function forward_subst(v, ilu::ILUPrecon)
	nzval = ilu.nzval
	n = ilu.A.n
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	y = zeros(typeof(v[1]), n)
	for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	y
end



"""
Solving Ux = y for the upper-triangular matrix U `encoded` in ILU.
"""
function backward_subst!(x, y, ilu::ILUPrecon)
	nzval = ilu.nzval
	n = ilu.A.n
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	wrk = copy(y)
	@inbounds for j=n:-1:1
		x[j] = wrk[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
end

function backward_subst_time!(x, y, ilu::ILUPrecon)
	t = @elapsed begin
		nzval = ilu.nzval
		n = ilu.A.n
		colptr = ilu.A.colptr
		rowval = ilu.A.rowval
		diag = ilu.diag
		wrk = copy(y)
	end
	@info t
	t = @elapsed @inbounds for j=n:-1:1
		x[j] = wrk[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
	end
	@info t
end

function backward_subst(y, ilu::ILUPrecon)
	nzval = ilu.nzval
	n = ilu.A.n
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	x = zeros(typeof(y[1]), n)
	wrk = copy(y)
	for j=n:-1:1
		x[j] = wrk[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	x
end



function convert_LU_to_one(A, ILU)
	M = copy(A)
	n = A.n
	u_ctr = 1
	l_ctr = 1
	
	
	for j = 1:n
		for v=A.colptr[j]:A.colptr[j+1]-1
			i=A.rowval[v]
			if i>j
				M.nzval[v] = ILU.l_nzval[l_ctr]
				l_ctr += 1
			else
				M.nzval[v] = ILU.u_nzval[u_ctr]
				u_ctr += 1
			end
		end
	end
	M
end

function convert_LU_to_one(ilu::ILUPrecon)
	SparseMatrixCSC(ilu.A.n, ilu.A.m, ilu.A.colptr, ilu.A.rowval, ilu.nzval)
end



function compare_creation(n, r; num=10)
	A = sprand(n, n, r)+10I
	ILU_MA, diag = create_ILU(A)
	ILUZ = ilu0(A)
	MZ = convert_LU_to_one(A, ILUZ)
	@info "max diff: $(maximum(abs.((MZ-ILU_MA).nzval)))"
	
	t1 = zeros(num)
	t2 = zeros(num)
	
	for i=1:num
		t1[i] = @elapsed create_ILU(A)
		GC.gc()
	end
	
	for i=1:num
		t2[i] = @elapsed ilu0(A)
		GC.gc()
	end
	
	@info "My way:", form(t1)
	@info "Pkg   :", form(t2)
	
end


function compare_solution(n, r; num=10)
	A = sprand(n, n, r)+10I
	ILU_MA, diag = create_ILU(A)
	ILUZ = ilu0(A)
	MZ = convert_LU_to_one(A, ILUZ)
	
	b = 2*rand(n).-1.0
	
	#@info "max diff ILU:", maximum(abs.((MZ-ILU_MA).nzval))
	
	
	# my way
	y = forward_subst(ILU_MA, b, diag, A)
	z = backward_subst(ILU_MA, y, diag, A)

	# pkg
	yy = copy(b)
	forward_substitution!(yy, ILUZ, b)
	zz = copy(b)
	backward_substitution!(zz, ILUZ, yy)
	
	#t = copy(b)
	#ldiv!(t, ILUZ, b)
	
	@info "max diff y," maximum(abs.(yy-y))
	@info "max diff z," maximum(abs.(zz-z))
	
	
	t1 = zeros(num)
	t2 = zeros(num)
	
	for i=1:num
		t1[i] = @elapsed begin
			yy = copy(y)
			forward_subst!(yy, ILU_MA, b, diag, A)
			zz = copy(b)
			backward_subst!(zz, ILU_MA, y, diag, A)
		end
		GC.gc()
	end
	
	for i=1:num
		t2[i] = @elapsed begin
			yy = copy(b)
			forward_substitution!(yy, ILUZ, b)
			zz = copy(b)
			backward_substitution!(zz, ILUZ, yy)			
		end
		GC.gc()
	end
	
	@info "My way:", form(t1)
	@info "Pkg   :", form(t2)
	
end	


function forward_subst_old!(y, v, nzval, diag, A)
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	
    for i in eachindex(y)
        y[i] = zero(Float64)
    end
    
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end

function backward_subst_old!(x, y, nzval, diag, A)
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	#wrk = copy(y)
	@inbounds for j=n:-1:1
		x[j] = y[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			y[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
end


	



	
