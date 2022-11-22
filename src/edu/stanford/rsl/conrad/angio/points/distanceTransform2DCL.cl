
typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

__kernel void distanceTransformKernel(
		// the distance map
		__global TvoxelValue* distanceMap,
		// the volume origin
		__constant Tcoord_dev* gVolumeOrigin,
		// the volume size
		__constant Tcoord_dev* gVolumeSize,
		// the voxel size
		__constant Tcoord_dev* gVoxelElementSize,
		// the projection matrices
		__global Tcoord_dev* gPoints,
		// the number of points
		int numPoints
		)
{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int x = mad24(gidx,locSizex,lidx);
    int y = mad24(gidy,locSizey,lidy);
	
	unsigned int yStride = gVolumeSize[0];
		
	if (x >= gVolumeSize[0] || y >= gVolumeSize[1])
	{
		return;
	}
	// x and y will be constant in this thread;
	float xcoord = (x * gVoxelElementSize[0]) + gVolumeOrigin[0];
	float ycoord = (y * gVoxelElementSize[1]) + gVolumeOrigin[1];
	unsigned long idx = y*yStride + x;
	float minDist = gVolumeSize[0]*gVoxelElementSize[0];
	for(int i = 0; i < numPoints; i++){
		float xpoint = gPoints[i*2+0];
		float ypoint = gPoints[i*2+1];
		float dist = (xpoint-xcoord)*(xpoint-xcoord) + (ypoint-ycoord)*(ypoint-ycoord);
		dist = sqrt(dist);
		if(dist < minDist){
			minDist = dist;
		}
	}
	distanceMap[idx] = minDist;
	
	return;
}



/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
