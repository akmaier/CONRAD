/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

float4 applyMotion(global float * motion, float4 coord, uint4 voxelCoord, unsigned int zStride, unsigned int yStride){
	long idx = voxelCoord.z*zStride + voxelCoord.y*yStride + voxelCoord.x;
	float4 localMotion = (float4){motion[idx*3], motion[(idx*3)+1], motion[(idx*3)+2], 0.0f};
	return coord + localMotion;
	
}