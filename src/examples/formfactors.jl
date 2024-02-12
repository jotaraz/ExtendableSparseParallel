function trifactors!(ω, e, itri, pointlist, trianglelist)
	# Obtain the node numbers for triangle itri
	i1=trianglelist[1,itri]
	i2=trianglelist[2,itri]
	i3=trianglelist[3,itri]
	# Calculate triangle area:
	# Matrix of edge vectors
	V11= pointlist[1,i2]- pointlist[1,i1]
	V21= pointlist[2,i2]- pointlist[2,i1]
	V12= pointlist[1,i3]- pointlist[1,i1]
	V22= pointlist[2,i3]- pointlist[2,i1]
	V13= pointlist[1,i3]- pointlist[1,i2]
	V23= pointlist[2,i3]- pointlist[2,i2]
	# Compute determinant
	det=V11*V22 - V12*V21
	# Area
	area=0.5*det
	# Squares of edge lengths
	dd1=V13*V13+V23*V23 # l32
	dd2=V12*V12+V22*V22 # l31
	dd3=V11*V11+V21*V21 # l21
	# Contributions to e_kl=σ_kl/h_kl
	e[1]= (dd2+dd3-dd1)*0.125/area
	e[2]= (dd3+dd1-dd2)*0.125/area
	e[3]= (dd1+dd2-dd3)*0.125/area
	# Contributions to ω_k
	ω[1]= (e[3]*dd3+e[2]*dd2)*0.25
	ω[2]= (e[1]*dd1+e[3]*dd3)*0.25
	ω[3]= (e[2]*dd2+e[1]*dd1)*0.25
end

function bfacefactors!(γ,ibface, pointlist, segmentlist)
	i1=segmentlist[1,ibface]
	i2=segmentlist[2,ibface]
	dx=pointlist[1,i1]-pointlist[1,i2]
	dy=pointlist[2,i1]-pointlist[2,i2]
	d=0.5*sqrt(dx*dx+dy*dy)
	γ[1]=d
	γ[2]=d
end

function trifactors2!(ω, e, itri, pointlist, trianglelist)
	# Obtain the node numbers for triangle itri
	i1 = trianglelist[1,itri]
	i2 = trianglelist[2,itri]
	i3 = trianglelist[3,itri]
	# Calculate triangle area:
	# Matrix of edge vectors
	V11 = pointlist[1,i2]- pointlist[1,i1]
	V21 = pointlist[2,i2]- pointlist[2,i1]
	V12 = pointlist[1,i3]- pointlist[1,i1]
	V22 = pointlist[2,i3]- pointlist[2,i1]
	V13 = pointlist[1,i3]- pointlist[1,i2]
	V23 = pointlist[2,i3]- pointlist[2,i2]
	# Compute determinant
	det  = V11*V22 - V12*V21
	# Area
	area = 0.5*det
	# Squares of edge lengths
	dd1 = V13*V13+V23*V23 # l32
	dd2 = V12*V12+V22*V22 # l31
	dd3 = V11*V11+V21*V21 # l21
	# Contributions to e_kl=σ_kl/h_kl
	e[1,itri] = (dd2+dd3-dd1)*0.125/area
	e[2,itri] = (dd3+dd1-dd2)*0.125/area
	e[3,itri] = (dd1+dd2-dd3)*0.125/area
	# Contributions to ω_k
	ω[1,itri] = (e[3,itri]*dd3+e[2,itri]*dd2)*0.25
	ω[2,itri] = (e[1,itri]*dd1+e[3,itri]*dd3)*0.25
	ω[3,itri] = (e[2,itri]*dd2+e[1,itri]*dd1)*0.25
end

function bfacefactors2!(γ,ibface, pointlist, segmentlist)
	i1          = segmentlist[1,ibface]
	i2          = segmentlist[2,ibface]
	dx          = pointlist[1,i1]-pointlist[1,i2]
	dy          = pointlist[2,i1]-pointlist[2,i2]
	d           = 0.5*sqrt(dx*dx+dy*dy)
	γ[1,ibface] = d
	γ[2,ibface] = d
end

function compute_formfactors(grid)
	trianglelist = grid[CellNodes]
	pointlist    = grid[Coordinates]
	segmentlist  = grid[BEdgeNodes]
	
	ntri   = num_cells(grid) #size(trianglelist,2)
	nbface = num_bedges(grid)
	num_nodes_per_cell=3;
	num_edges_per_cell=3;
	num_nodes_per_bface=2
	
	e=zeros(num_nodes_per_cell, ntri)
	ω=zeros(num_edges_per_cell, ntri)
	γ=zeros(num_nodes_per_bface, nbface)
	
	for itri=1:ntri
		trifactors2!(ω, e, itri, pointlist, trianglelist)
	end
	
	for ibface=1:nbface
		bfacefactors2!(γ,ibface, pointlist, segmentlist)
	end
	
	e, ω, γ
end


