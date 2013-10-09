
#include "xtalk_sim.h"
#include "DiffEqModel.h"
#include "utils.h"

// builds the well structure in the bottom wellz elements of the simulation
void DiffEqModel::BuildHexagonalWellPattern(int wellz,DATA_TYPE pitch,DATA_TYPE well_fraction)
{
	// first, define the pattern the wells make along the x and y plane
	// at the bottom of the simulation volume
	bool *well_pattern = new bool[ni*nj];
	bool *all_well_pattern = new bool[ni*nj];
	
	memset(all_well_pattern,0,sizeof(bool[ni*nj]));

	// draw the hexagon pattern of wells 
	int row = 0;
	int col = 0;
	int num_set;
	Coordinate loc;
	loc.x = 0.0f;
	loc.y = 0.0f;

	while(loc.x < dx*ni)
	{
		loc.y = 0.0f;
		row = 0;
		while(loc.y < dy*nj)
		{
			loc = GetWellLocation(row,col,pitch);
			HexagonDescriptor descr = GetHexagon(loc,pitch,well_fraction);
			
			memset(well_pattern,0,sizeof(bool[ni*nj]));
			num_set = MarkInHexagon(descr,well_pattern,ni,nj,dx,dy,pitch);
			
			if (num_set != 0)
			{
				SignalAverager *newAvg = new SignalAverager(well_pattern,ni*nj);
				newAvg->SetRow(row);
				newAvg->SetCol(col);
				
				if (head_avg == NULL)
					head_avg = newAvg;
				else
				{
					newAvg->SetNext(head_avg);
					head_avg = newAvg;
				}

				for (int i=0;i < (ni*nj);i++)
					all_well_pattern[i] = all_well_pattern[i] || well_pattern[i];
			}
			
			row++;
		}
		
		col++;
	}
	
	FILE *dbgfile = fopen("well_pattern_debug.txt","wt");
	
	for (int i=0;i < ni;i++)
	{
		for (int j=0;j < nj;j++)
			fprintf(dbgfile,"%d\t",all_well_pattern[i+j*ni] ? 1 : 0);
			
		fprintf(dbgfile,"\n");
	}
	
	fclose(dbgfile);
	
	// figure out px,mx,py,my boundaries caused by well walls
	for (int j=0;j < nj;j++)
		for (int i=0;i < ni;i++)
		{
			if (all_well_pattern[i+j*ni])
			{
				if ((j==0) || (i==0) || (j==(nj-1)) || (i==(ni-1)))
					continue;
			
				if (!all_well_pattern[i+j*ni-1])
					boundary_class[i+j*ni] |= MX_BOUND;
				if (!all_well_pattern[i+j*ni+1])
					boundary_class[i+j*ni] |= PX_BOUND;
				if (!all_well_pattern[i+(j-1)*ni])
					boundary_class[i+j*ni] |= MY_BOUND;
				if (!all_well_pattern[i+(j+1)*ni])
					boundary_class[i+j*ni] |= PY_BOUND;
			}
			else
				boundary_class[i+j*ni] = WALL_ELEMENT;
		}
	
	// now propagate the bottom layer through the bottom wellz layers
	for (int z=1;z < wellz;z++)
		for (int i=0;i < (ni*nj);i++)
			boundary_class[i+z*(ni*nj)] |= boundary_class[i];

	// now set the tops of the well walls to have a minus-z boundary
	for (int i=0;i < (ni*nj);i++)
		if (!all_well_pattern[i])
			boundary_class[i+wellz*(ni*nj)] |= MZ_BOUND;

	delete [] well_pattern;
	delete [] all_well_pattern;
}

void DiffEqModel::SetBufferEffect(DATA_TYPE wash_buffer_cap)
{
	DATA_TYPE voxel_volume = dx*dy*dz;
	DATA_TYPE voxel_bc = wash_buffer_cap*voxel_volume;

	// area of various sidewall combinations
	DATA_TYPE dxdy = dx*dy;
	DATA_TYPE dxdz = dx*dz;
	DATA_TYPE dydz = dy*dz;

	for (int i=0;i < (ni*nj*nk);i++)
	{
		buffer_effect[i] = voxel_bc;
		if ((boundary_class[i] & PX_BOUND) || (boundary_class[i] & MX_BOUND))
			buffer_effect[i] += dydz;
		if ((boundary_class[i] & PY_BOUND) || (boundary_class[i] & MY_BOUND))
			buffer_effect[i] += dxdz;
		if ((boundary_class[i] & PZ_BOUND) || (boundary_class[i] & MZ_BOUND))
			buffer_effect[i] += dxdy;
			
		// for good reasons (quirk in the diffeq computation code) we don't mark the well bottoms
		// with a MZ bound (the mz boundary is automatically handled by the way we compute the diffeq
	    // at the edge of the matrix, and setting MZ_BOUND for pixels at the edge of the matrix 
		// results in an incorrect calculation)
		// ...BUT we still need to added extra buffer capacity to the elements that interface with the
		// well bottom...so add it to all k=0 elements
		if (i < (ni*nj))
			buffer_effect[i] += dxdy;
			
		if (boundary_class[i] & WALL_ELEMENT)
			buffer_effect[i] = 0.0f;
		else
			buffer_effect[i] = voxel_bc / buffer_effect[i];
	}
}

void DiffEqModel::SetEdgeBoundaryClass()
{
	for (int k=0;k < nk;k++)
		for (int j=0;j < nj;j++)
			for (int i=0;i < ni;i++)
			{
				if (k==0)
					boundary_class[i+j*ni+k*ni*nj] |= MZ_BOUND;
				if (k==nk-1)
					boundary_class[i+j*ni+k*ni*nj] |= PZ_BOUND;
				if (j==0)
					boundary_class[i+j*ni+k*ni*nj] |= MY_BOUND;
				if (j==nj-1)
					boundary_class[i+j*ni+k*ni*nj] |= PY_BOUND;
				if (i==0)
					boundary_class[i+j*ni+k*ni*nj] |= MX_BOUND;
				if (i==ni-1)
					boundary_class[i+j*ni+k*ni*nj] |= PX_BOUND;
			}
}

void DiffEqModel::SetConvection(int wellz)
{
	// average flow cell height
	DATA_TYPE flow_cell_height = (40.0E-6f)*1000.0f*pow((1E+6f/100.0f),3.0f)/(22000.0f*19000.0f);

	// typical vmax in x-direction inside proton flow cell
	DATA_TYPE vol_per_sec = 40.0E-6f;  // liters/second
	DATA_TYPE ml_per_sec = vol_per_sec*1000.0f;
	DATA_TYPE cc_per_sec = ml_per_sec;
	DATA_TYPE cubic_microns_per_sec = cc_per_sec*pow((1E+6f/100.0f),3.0f);
	DATA_TYPE vavg = cubic_microns_per_sec / (19000.0f*flow_cell_height);
	DATA_TYPE vmax = 2.0f*vavg;
	DATA_TYPE fcR = flow_cell_height/2.0f;
	DATA_TYPE flow_cell_dimz = nk - wellz;  // flow starts just above the wells
	
	for (int k=0;k < flow_cell_dimz;k++)
	{
		// distance from vertical center of flow cell
		DATA_TYPE rval = (fcR - dz*k+0.5f);
		
		// x-velocity in layer k+wellz
		xvel[k+wellz] = vmax*(1.0f-(rval*rval)/(fcR*fcR));
		
		layer_step_frac[k+wellz] = xvel[k+wellz]*dt/dx;
	}
}

void DiffEqModel::SetSignalInWell(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE signal)
{
	bool *well_pattern = new bool[ni*nj];
	memset(well_pattern,0,sizeof(bool[ni*nj]));
	Coordinate loc;

	loc = GetWellLocation(row,col,pitch);
	HexagonDescriptor descr = GetHexagon(loc,pitch,well_fraction);
	MarkInHexagon(descr,well_pattern,ni,nj,dx,dy,pitch);

	// set all elements within this well through the total well height (wellz) to the specified signal
	for (int k=0;k < wellz;k++)
	{
		for (int i=0;i < (ni*nj);i++)
			if (well_pattern[i])
				cmatrix[i+k*(ni*nj)] = signal;
	}
	
	delete well_pattern;
}

// predicts [dNTP] at the boundaries of the simulation 
void DiffEqModel::DynamicDNTPBoundaryCondition(DATA_TYPE simTime)
{
	DATA_TYPE travelDistance = 10000.0f;

	for (int k=0;k < nk;k++)
	{
		if (xvel[k] <= 0.0f)
			continue;
		
		DATA_TYPE travelTime = travelDistance / (xvel[k]);
		
		if (travelTime >= simTime)
		{
			for (int j=0;j < nj;j++)
				cmatrix[k*(ni*nj)+j*ni] = 500.0f;
		}
	}
}

DATA_TYPE DiffEqModel::AddUpWellBuffering(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE wash_buffer_cap)
{
	bool *well_pattern = new bool[ni*nj];
	memset(well_pattern,0,sizeof(bool[ni*nj]));
	Coordinate loc;
	DATA_TYPE voxel_volume = dx*dy*dz;
	DATA_TYPE voxel_bc = wash_buffer_cap*voxel_volume;

	loc = GetWellLocation(row,col,pitch);
	HexagonDescriptor descr = GetHexagon(loc,pitch,well_fraction);
	MarkInHexagon(descr,well_pattern,ni,nj,dx,dy,pitch);

	DATA_TYPE buffer_sum = 0.0f;
	int cnt = 0;
	
	// set all elements within this well through the total well height (wellz) to the specified signal
	for (int k=0;k < wellz;k++)
	{
		for (int i=0;i < (ni*nj);i++)
			if (well_pattern[i])
			{
				if (buffer_effect[i+k*ni*nj] != 0.0f)
				{
					buffer_sum += voxel_bc / buffer_effect[i+k*ni*nj];
					cnt++;
				}
			}
	}
	
	delete well_pattern;
	
	return(buffer_sum/cnt);
}

void DiffEqModel::AddBeadBufferingToAllWells(int wellz,DATA_TYPE bead_buffer_amount,DATA_TYPE wash_buffer_cap)
{
	DATA_TYPE voxel_volume = dx*dy*dz;
	DATA_TYPE voxel_bc = wash_buffer_cap*voxel_volume;

	for (int k=0;k < wellz;k++)
	{
		for (int i=0;i < (ni*nj);i++)
			if (!(boundary_class[i+k*ni*nj] & WALL_ELEMENT))
			{
				if (buffer_effect[i+k*ni*nj] != 0.0f)
				{
					buffer_effect[i+k*ni*nj] = voxel_bc / (bead_buffer_amount + voxel_bc / buffer_effect[i+k*ni*nj]);
				}
			}
	}
}


// adds specified amount of buffering to all voxel in a single well
void DiffEqModel::AddBeadBufferingToSingleWell(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction,DATA_TYPE bead_buffer_amount,DATA_TYPE wash_buffer_cap)
{
	bool *well_pattern = new bool[ni*nj];
	memset(well_pattern,0,sizeof(bool[ni*nj]));
	Coordinate loc;
	DATA_TYPE voxel_volume = dx*dy*dz;
	DATA_TYPE voxel_bc = wash_buffer_cap*voxel_volume;

	loc = GetWellLocation(row,col,pitch);
	HexagonDescriptor descr = GetHexagon(loc,pitch,well_fraction);
	MarkInHexagon(descr,well_pattern,ni,nj,dx,dy,pitch);

	for (int k=0;k < wellz;k++)
	{
		for (int i=0;i < (ni*nj);i++)
			if (well_pattern[i])
			{
				if (buffer_effect[i+k*ni*nj] != 0.0f)
				{
					buffer_effect[i+k*ni*nj] = voxel_bc / (bead_buffer_amount + voxel_bc / buffer_effect[i+k*ni*nj]);
				}
			}
	}
	
	delete well_pattern;
}


void DiffEqModel::SetupIncorpSignalInjection(int wellz,int row,int col,DATA_TYPE pitch,DATA_TYPE well_fraction)
{
	bool *well_pattern = new bool[ni*nj];
	memset(well_pattern,0,sizeof(bool[ni*nj]));
	Coordinate loc;

	loc = GetWellLocation(row,col,pitch);
	HexagonDescriptor descr = GetHexagon(loc,pitch,well_fraction);
	MarkInHexagon(descr,well_pattern,ni,nj,dx,dy,pitch);

	pwell = new WellVoxelTracker(well_pattern,ni*nj);
	
	delete well_pattern;
}

// this call must be made (after ConfigureBeadSize and SetupIncorpSignalInjection) in order to build an array of indicies and weights
// that are used by the GPU code to do signal injection on the GPU instead of on the CPU.
void DiffEqModel::SetupIncorporationSignalInjectionOnGPU()
{
	if ((bead_volume_per_slice == NULL) || (pwell == NULL))
		return;

	int ninject = bead_z_slices*pwell->GetNumIndicies();

	index_array = new size_t[ninject];
	weight_array = new DATA_TYPE[ninject];
	int cnt=0;
	
	// set all elements within this well through the total well height (wellz) to the specified signal
	for (int k=0;k < bead_z_slices;k++)
	{
		for (int i=0;i < pwell->GetNumIndicies();i++)
		{
			index_array[cnt] = (*pwell)[i]+k*(ni*nj);
			weight_array[cnt] = bead_volume_per_slice[k];
			cnt++;
		}
	}
	
	incorp_inject_cnt = ninject;
}

void DiffEqModel::AddSignalToWell(WellVoxelTracker *pwell,DATA_TYPE signal)
{
	if (bead_volume_per_slice == NULL)
		return;
	
	// set all elements within this well through the total well height (wellz) to the specified signal
	for (int k=0;k < bead_z_slices;k++)
	{
		for (int i=0;i < pwell->GetNumIndicies();i++)
			cmatrix[(*pwell)[i]+k*(ni*nj)] += signal*bead_volume_per_slice[i];
	}
}

void DiffEqModel::ConfigureBeadSize(DATA_TYPE bead_radius)
{
	bead_z_slices = (int)((bead_radius*2.0f / dz) + 0.99);
	
	bead_volume_per_slice = new DATA_TYPE[bead_z_slices];
	
	DATA_TYPE total_bead_volume = 0.0f;
	for (int i=0;i < bead_z_slices;i++)
	{
		DATA_TYPE zcoord = dz/2 + i*dz - bead_radius;
		DATA_TYPE slice_volume = 3.14159f*dz*(bead_radius*bead_radius - zcoord*zcoord);
		
		if (slice_volume > 0.0f)
			bead_volume_per_slice[i] = slice_volume;
		else
			bead_volume_per_slice[i] = 0.0f;
			
		total_bead_volume += bead_volume_per_slice[i];
	}

	// normalize to total bead volume
	for (int i=0;i < bead_z_slices;i++)
		bead_volume_per_slice[i] /= total_bead_volume;
}







	

