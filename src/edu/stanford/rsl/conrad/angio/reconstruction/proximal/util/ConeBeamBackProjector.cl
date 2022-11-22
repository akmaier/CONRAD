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
	float spacingZ,
	float normalizer
	//float spacingU,
	//float spacingV,

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
	
	float4 pos = {(float)x*spacingX-originX,(float)y*spacingY -originY,0.0f, 1.0f};
	float precomputeR = gProjMatrix[p*12 + 3]  + pos.y * gProjMatrix[p*12 + 1] + pos.x * gProjMatrix[p*12 + 0];
	float precomputeS = gProjMatrix[p*12 + 7]  + pos.y * gProjMatrix[p*12 + 5] + pos.x * gProjMatrix[p*12 + 4];
  	float precomputeT = gProjMatrix[p*12 + 11] + pos.y * gProjMatrix[p*12 + 9] + pos.x * gProjMatrix[p*12 + 8];
	
	int sizeXY = mul24(imgSizeX,imgSizeY);

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
		int idx = mad24(z,sizeXY,mad24(y,imgSizeX,x));
		// add value to imageGrid
		imgGrid[idx] += proj_val;//*denom*denom*normalizer;
		
	}
}

kernel void backProjectPixelDrivenCL3D(
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
	float spacingZ,
	float normalizer
	//float spacingU,
	//float spacingV,

) {

	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int gidz = get_group_id(2);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int lidz = get_local_id(2);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	int locSizez = get_local_size(2);
	
	int x = mad24(gidx,locSizex,lidx);
    int y = mad24(gidy,locSizey,lidy);
    int z = mad24(gidz,locSizez,lidz);
    
	// check if inside image boundaries
	if ( x>= imgSizeX || y >= imgSizeY || z >= imgSizeZ)
		return;
	
	float4 pos = {(float)x*spacingX-originX,(float)y*spacingY -originY,0.0f, 1.0f};
	float precomputeR = gProjMatrix[p*12 + 3]  + pos.y * gProjMatrix[p*12 + 1] + pos.x * gProjMatrix[p*12 + 0];
	float precomputeS = gProjMatrix[p*12 + 7]  + pos.y * gProjMatrix[p*12 + 5] + pos.x * gProjMatrix[p*12 + 4];
  	float precomputeT = gProjMatrix[p*12 + 11] + pos.y * gProjMatrix[p*12 + 9] + pos.x * gProjMatrix[p*12 + 8];
	
	int sizeXY = mul24(imgSizeX,imgSizeY);
	
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
	int idx = mad24(z,sizeXY,mad24(y,imgSizeX,x));
	
	// add value to imageGrid
	imgGrid[idx] += proj_val*denom*denom*normalizer;

}

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/