__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;


kernel void backProjectPixelDrivenCL(
	image2d_t sino,
	global float* imgGrid,
	global float* gProjMatrix, 
	/* not yet... float2 gridSpacing, */
	int p,
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ
	
	//float spacingU,
	//float spacingV,

) {

	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	
	// check if inside image boundaries
	if ( x>= imgSizeX || y >= imgSizeY)
		return;
	
	float4 pos = {(float)x*spacingX-originX,(float)y*spacingY -originY,0.0, 1.0};
	float precomputeR = gProjMatrix[p*12 + 3]  + pos.y * gProjMatrix[p*12 + 1] + pos.x * gProjMatrix[p*12 + 0];
	float precomputeS = gProjMatrix[p*12 + 7]  + pos.y * gProjMatrix[p*12 + 5] + pos.x * gProjMatrix[p*12 + 4];
  float precomputeT = gProjMatrix[p*12 + 11] + pos.y * gProjMatrix[p*12 + 9] + pos.x * gProjMatrix[p*12 + 8];
		
	for (int z=0;z<imgSizeZ;++z) {
	
		pos.z = ((float)z * spacingZ) -originZ;
		float r = pos.z * gProjMatrix[p*12 + 2]  + precomputeR;
		float s = pos.z * gProjMatrix[p*12 + 6]  + precomputeS;
		float t = pos.z * gProjMatrix[p*12 + 10] + precomputeT;
		
		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r* denom;
		float fv = s* denom;
		float2 posUV = {fu+0.5f, fv+0.5f};  //,(float)p+0.5f,0.f};
		float proj_val = read_imagef(sino, linearSampler, posUV).x;   // <--- correct for non-padded detector

		// compute volume index for x,y,z
		int idx = z*imgSizeX*imgSizeY + y * imgSizeX + x;
		
		// add value to imageGrid
		imgGrid[idx] += proj_val*denom*denom;
		
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/