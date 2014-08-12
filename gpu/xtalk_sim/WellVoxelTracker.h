/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef WELLVOXELTRACKER_H
#define WELLVOXELTRACKER_H

#include <assert.h>

class WellVoxelTracker
{
public:

	WellVoxelTracker(bool *mask,int numElements)
	{
		int num_voxels = 0;
		
		// figure out how many are set in the mask
		for (int i=0;i < numElements;i++)
		{
			if (mask[i])
				num_voxels++;
		}

		nVoxels = num_voxels;
		voxel_indicies = new int[nVoxels];

		num_voxels = 0;
		for (int i=0;(i < numElements) && (num_voxels < nVoxels);i++)
		{
			if (mask[i])
				voxel_indicies[num_voxels++] = i;
		}
	}

	int GetNumIndicies(void)
	{
		return nVoxels;
	}
	
	int GetIndex(int num)
	{
		assert(num < nVoxels);
		return(voxel_indicies[num]);
	}
	
	int operator[] (int num)
	{
		return(GetIndex(num));
	}
	
	~WellVoxelTracker()
	{
		delete [] voxel_indicies;
	}

private:
	int nVoxels;
	int *voxel_indicies;
	

};



#endif // WELLVOXELTRACKER_H


