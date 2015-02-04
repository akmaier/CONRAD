/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

float4 getPointFromArray(global float* inputvalues, int i){
	unsigned long idx = i;
	float4 voxel = (float4){inputvalues[idx*3], inputvalues[(idx*3)+1], inputvalues[(idx*3)+2], 0};
	return voxel;
}

float4 getPointFromSearchArray(global const float* inputvalues, int i){
	unsigned long idx = i;
	float4 voxel = (float4){inputvalues[idx*5], inputvalues[(idx*5)+1], inputvalues[(idx*5)+2], 0};
	return voxel;
}

void setPointToArray(global float* array, int x, int y, int z, unsigned int yPitch, unsigned int zPitch, float4 vector){
	unsigned long idx = z*zPitch + y*yPitch + x;
	array[idx*3]   = vector.x;
	array[idx*3+1] = vector.y;
	array[idx*3+2] = vector.z;
}

/**
 * 1-D Version that is compatible with the motion field interface.
 */
kernel void evaluateParzen1D(global const float* inputPoints, global const float * outputPoint, global float* outputBuffer, int bufferLength, int numPointsPerWorker, float sigma) {
	int gidx = get_group_id(0);
	int lidx = get_local_id(0);
	int locSizex = get_local_size(0);	
	int gloSizex = get_global_size(0);
	unsigned int x = gidx*locSizex+lidx;
    
    if (x >= bufferLength)
		return;
        
    float4 summation = (float4) {0,0,0,0};
	float4 initial = getPointFromArray(outputBuffer, x);
	float weightsum = 0;
	float factor = -0.5f/(sigma*sigma);
	float acc = 70; // Evaluated with heart phantom
	// Highest number before summations yield NaN
	for (int i=0; i< numPointsPerWorker; i++){
		float4 from = getPointFromArray(inputPoints, i);
		float4 to = getPointFromArray(outputPoint, i);
		float4 direction = to - from;
		float dist = distance(from, initial);
		float weight = exp((factor*dist*dist)+acc);
		weightsum += weight;
		summation +=(direction*weight);
	}
	if(fabs(weightsum)> 0.00000001){
	summation /= weightsum;// (float4) {weightsum, factor, originZ, 0};
	} else {
		summation = (float4){0,0,0,0};
	}
	outputBuffer[x*3]   = summation.x;
	outputBuffer[x*3+1] = summation.y;
	outputBuffer[x*3+2] = summation.z;
	return;
}


kernel void evaluateParzen(global const float* inputPoints, global const float * outputPoint, global float* outputBuffer, int reconDimX, int reconDimY, int maxZ, int numPoints, float sigma, float voxelSpacingX, float voxelSpacingY, float voxelSpacingZ, float originX, float originY, float originZ) {
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
    
    unsigned int yPitch = gloSizex;
    unsigned int zPitch = gloSizex * gloSizey;
	
    if (x >= reconDimX || y >= reconDimY)
		return;
        
    float xcoord = (x * voxelSpacingX) +originX;
    float ycoord = (y * voxelSpacingY) +originY;
    int z=maxZ;
	float zcoord = (z * voxelSpacingZ) +originZ;
	float4 summation = (float4) {0,0,0,0};
	float4 initial = (float4) {xcoord, ycoord, zcoord, 0};
	float weightsum = 0;
	float factor = -0.5f/(sigma*sigma);
	float acc = 70; // Evaluated with heart phantom
	// Highest number before summations yield NaN
	for (int i=0; i< numPoints; i++){
		float4 from = getPointFromArray(inputPoints, i);
		float4 to = getPointFromArray(outputPoint, i);
		float4 direction = to - from;
		float dist = distance(from, initial);
		float weight = exp((factor*dist*dist)+acc);
		weightsum += weight;
		summation +=(direction*weight);
	}
	if(fabs(weightsum)> 0.00000001){
	summation /= weightsum;// (float4) {weightsum, factor, originZ, 0};
	} else {
		summation = (float4){0,0,0,0};
	}
	setPointToArray(outputBuffer, x, y, z, yPitch, zPitch, summation);
	
	
	return;
        
}

kernel void evaluateNN(global const float* inputPoints, global const float * outputPoint, global float* outputBuffer, int reconDimX, int reconDimY, int maxZ, int numPoints, float sigma, float voxelSpacingX, float voxelSpacingY, float voxelSpacingZ, float originX, float originY, float originZ) {
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
    
    unsigned int yPitch = gloSizex;
    unsigned int zPitch = gloSizex * gloSizey;
	
    if (x >= reconDimX || y >= reconDimY)
		return;
        
    float xcoord = (x * voxelSpacingX) +originX;
    float ycoord = (y * voxelSpacingY) +originY;
    int z=maxZ;
	float zcoord = (z * voxelSpacingZ) +originZ;
	float4 initial = (float4) {xcoord, ycoord, zcoord, 0};
	// Highest number before summations yield NaN
	float minDist = 100000;
	int minInd = -1;
	for (int i=0; i< numPoints; i++){
		float4 from = getPointFromArray(inputPoints, i);
		float dist = distance(from, initial);
		if (dist < minDist){
			minDist = dist;
			minInd = i;
		}
	}
	float4 direction = (float4){0,0,0,0};
	if (minInd != -1){
		float4 from = getPointFromArray(inputPoints, minInd);
		float4 to = getPointFromArray(outputPoint, minInd);
		direction = to - from;
	}
	setPointToArray(outputBuffer, x, y, z, yPitch, zPitch, direction);	
	return;
        
}


kernel void evaluateParzenLocal(global const float * search, global const int * searchIdx, global const float * local1, global const float * local2, global float* outputBuffer, int reconDimX, int reconDimY,  int searchLen, int maxZ, float sigma, float voxelSpacingX, float voxelSpacingY, float voxelSpacingZ, float originX, float originY, float originZ) {
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
    
    unsigned int yPitch = gloSizex;
    unsigned int zPitch = gloSizex * gloSizey;
	
    if (x >= reconDimX || y >= reconDimY)
		return;
        
    float xcoord = (x * voxelSpacingX) +originX;
    float ycoord = (y * voxelSpacingY) +originY;
    int z=maxZ;
	float zcoord = (z * voxelSpacingZ) +originZ;
	float4 summation = (float4) {0,0,0,0};
	float4 initial = (float4) {xcoord, ycoord, zcoord, 0};
	float weightsum = 0;
	float factor = -0.5f/(sigma*sigma);
	int closePoint = -1;
	float dist = 100000;
	for (int i=0; i< searchLen; i++){
		float4 from = getPointFromArray(search, i);
		float dist2 = distance(from, initial);
		if (dist2 < dist){
			dist = dist2;
			closePoint = i;
		}
	}
	if (closePoint == -1) return;
	int start = (int) searchIdx[closePoint*2];
	int len = (int) searchIdx[closePoint*2+1];
	float acc = 70; // Evaluated with heart phantom
	for (int i=start; i< start + len; i++){
		float4 from = getPointFromArray(local1, i);
		float4 to = getPointFromArray(local2, i);
		float4 direction = to - from;
		float dist = distance(from, initial);
		float weight = exp((factor*dist*dist)+acc);
		weightsum += weight;
		summation +=(direction*weight);
	}
	if(fabs(weightsum)> 0.00000001){
		summation /= weightsum;// (float4) {weightsum, factor, originZ, 0};
	} else {
		summation = (float4){0,0,0,0};
	}
	setPointToArray(outputBuffer, x, y, z, yPitch, zPitch, summation);
	
	
	return;
        
}