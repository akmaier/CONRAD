__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

kernel void backProjectPixelDrivenCL(
	image3d_t sino,
	global float* imgGrid,
	global float* gProjMatrix,
	global float* gShiftsX,
	global float* gShiftsY,
	global float* gShiftsZ,	
	/* not yet... float2 gridSpacing, */
	int numP,
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ
) {

	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int x = mad24(gidx,locSizex,lidx);
    int y = mad24(gidy,locSizey,lidy);
    
	// check if inside image boundaries
	if ( x>= imgSizeX || y >= imgSizeY)
		return;
	
	int sizeXY = mul24(imgSizeX,imgSizeY);
	
	for(int p = 0; p < numP; p++){
		float4 pos = {(float)x*spacingX+originX+gShiftsX[p],(float)y*spacingY+originY+gShiftsY[p],0.0f, 1.0f};
		float precomputeR = gProjMatrix[p*12 + 3]  + pos.y * gProjMatrix[p*12 + 1] + pos.x * gProjMatrix[p*12 + 0];
		float precomputeS = gProjMatrix[p*12 + 7]  + pos.y * gProjMatrix[p*12 + 5] + pos.x * gProjMatrix[p*12 + 4];
		float precomputeT = gProjMatrix[p*12 + 11] + pos.y * gProjMatrix[p*12 + 9] + pos.x * gProjMatrix[p*12 + 8];

		for (int z=0;z<imgSizeZ;++z) {
			pos.z = ((float)z * spacingZ)+originZ+gShiftsZ[p];
			float r = pos.z * gProjMatrix[p*12 + 2]  + precomputeR;
			float s = pos.z * gProjMatrix[p*12 + 6]  + precomputeS;
			float t = pos.z * gProjMatrix[p*12 + 10] + precomputeT;
			
			// compute projection coordinates 
			float denom = 1.0f / t;
			float fu = r* denom;
			float fv = s* denom;
			float4 posUVP = {fu+0.5f, fv+0.5f, (float)p+0.5f, 0.f};
			float proj_val = read_imagef(sino, linearSampler, posUVP).x;   // <--- correct for non-padded detector

			// add value to imageGrid
			int idx = mad24(z,sizeXY,mad24(y,imgSizeX,x));
			imgGrid[idx] = max(imgGrid[idx],proj_val);
			//imgGrid[idx] /= (1.0f+proj_val);			
		}
	}
}

/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/