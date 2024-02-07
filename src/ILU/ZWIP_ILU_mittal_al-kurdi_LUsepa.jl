# https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122100X03628/1-s2.0-S0898122103001548/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECwaCXVzLWVhc3QtMSJIMEYCIQDqZI2d3zAPzu3jzKCUJcbdIFHESmBDMLNELM5MWIFU9AIhALrzOzG2r9PCDybVhvtMcglK%2BSZpJHhb91QjfGyB4UW2KrsFCNX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgwTtvmR2IuTp5k%2BLjgqjwXwDV%2FLpK0b64hCiVEVwfAAP%2Fkem4b7eZKhw%2B%2BJ81H1DvSKGTm1ggcwTWOBLeq56EsHbIMsSmaVHh4GhG1BFgSWxOMPvvtu0me22ybKEEHG2q67qHKPAh8SgEITQzBA17pm3FtAElxSbA5odA70qbrpKaewueWXOGrwV0H5lFtZ7vd1bDCznKzcDeiUacro7nQWRFVrOeHb2aNB1okc%2BzKiDyESWyNiFcAgBi4sZy9RlCCBROvkY8KB5I4%2FjImfy60B8A%2BezZoQmC7s0MF10BoffQwhHPgWmrueTkWpeqm7eyPTWtdH2y9%2B0JuKU3TCGBmrvSdZ1OfnIPJ8K1A%2FUmi5Olc1o%2BnkphIv66OB1RkPLSOEyB0ComgrilymL56YDQdIFq4dLAwK6iGORSuU2XQx%2B5clOQDAemfHNo21lkvq6CwbYt0o7MYUaJPcBrpxbP8vUqifWmXBYKmfcq%2FfdmtftvN%2BJBNshsJoxrsHySqrguPCAws%2ByQCDEQWhWR0ut6SYK%2FsKWdWYWvTDTmiKdcodDlACLem1oad%2FoQT54TRpx8RpK8eRM7PveCnchyc3TEMBCZa6AqPNqJF4JT%2FCPYx4icob3BAxgjKObGmnXo20xU8lu461ppqOBczBhGzAdrykBKx%2B7TJnSw4WDZNvzgMmBVNfpwzTuUmW7AxoP%2FtYxrm%2BcC5Tu5OZ1gLby0MgvpkPi%2BmzWnki15c%2BQPl54v4jxwF2oPRvFyU1L5JZr0ddd36BAcmjkurXEXn0RpzhO6u%2FQDUcXTU%2BNFiS7Az83yK%2ByqIfeSJu8jFnlJIpWl1hjo8AMbAGpdbuMAsxGqta%2FPJNdqlKZyHisG6spS85H6jDGf7nk3DnqK%2FLLKHgJQRRMKKhpK0GOrABd3QZ05XSUTqmSJGws8H1PVC8a5SeiyV5yWdu0XQ3KDAZMjnL%2BEiGM4ixXhvFQa1DR5PpSUuHTITuBv2o0xvEBLKDuNXB6aXiiDxPTxZnp7%2FygmYKYvct8eawnU%2BukNTwHhSmJIlibTb4EXw80tBVUd4Xnb19XzrwAkpA5FAftkWaaG%2BhIQmk7uZqpCR39dEW13T%2BWyZsd2Sw61nzvIBkaircl29%2Fpi8o%2BPZ1rNVTB7E%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240118T130827Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRSSK5WKL%2F20240118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6e74de2553b2759260437456edb10c29d3d8ed54b5fdd6461ab93b18bf547e1e&hash=adf84da67f1c362b7388afb179c4f4636b86c5e614274a2fb83b25bff460e0b1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0898122103001548&tid=spdf-26410b0a-8f71-4078-a302-8cd53dd990fb&sid=7943cffa85ea284bc48ba2422171a4974b6egxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e03575355535f5355520f&rr=84771116afb36a76&cc=de


using SparseArrays
using ILUZero
using LinearAlgebra

#include("test.jl")




"""
Calculationg L and U using the aforementioned paper of a matrix A.
It needs to have all the diagonal entries.
"""
function create_ILU_shape(A::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti <:Integer}
	n = A.n
	lnz = 0
	unz = 0
	
	nzval = (A.cscmatrix.nzval)
	colptr = A.colptr
	rowval = A.rowval
	
	@inbounds for i = 1:n
        for j = colptr[i]:colptr[i + 1] - 1
            if rowval[j] > i
                lnz += 1
            else
                unz += 1
            end
        end
    end
	
	l_colptr = zeros(Ti, n + 1)
    u_colptr = zeros(Ti, n + 1)
    l_nzval = zeros(Tv, lnz)
    u_nzval = zeros(Tv, unz)
    l_rowval = zeros(Ti, lnz)
    u_rowval = zeros(Ti, unz)
    l_map = Vector{Ti}(undef, lnz)
    u_map = Vector{Ti}(undef, unz)
    #wrk = zeros(Ti, n)
    l_colptr[1] = 1
    u_colptr[1] = 1
	
	
	lit = 1
    uit = 1
    @inbounds for i = 1:n
        l_colptr[i + 1] = l_colptr[i]
        u_colptr[i + 1] = u_colptr[i]
        for j = colptr[i]:colptr[i + 1] - 1
            if rowval[j] > i
                l_colptr[i + 1] += 1
                l_rowval[lit] = rowval[j]
                l_nzval[lit] = nzval[j]
                l_map[lit] = j
                lit += 1
            else
                u_colptr[i + 1] += 1
                u_rowval[uit] = rowval[j]
                u_nzval[uit] = nzval[j]
                u_map[uit] = j
                uit += 1
            end
        end
    end
	
	
	(l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map)
end

function fill_ILU(arrays, A)
	l_colptr, l_rowval, l_nzval, u_colptr, u_rowval, u_nzval, l_map, u_map = arrays
	
	nzval = A.nzval)
	colptr = A.colptr
	rowval = A.rowval
	n = A.n
	point = zeros(Ti, n) #Vector{Ti}(undef, n)
	
	# compute L and U
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1  ## start at colptr[j]+1 ??
			point[rowval[v]] = v
		end
		
		for v=u_colptr[j]:u_colptr[j+1]-1
			i = l_rowval[v]
			for w=l_rowval[i]:l_rowval[i+1]-1
				k = point[u_rowval[w]]
				if k>0
					nzval[k] -= 
		
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


	



	
