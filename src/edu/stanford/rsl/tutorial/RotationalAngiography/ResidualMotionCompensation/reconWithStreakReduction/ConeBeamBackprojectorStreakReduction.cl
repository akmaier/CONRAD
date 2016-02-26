__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

kernel void backProjectPixelDrivenCL(image2d_t sino,global float* imgGrid,global float* destGrid,global float* gProjMatrix, /* not yet... float2 gridSpacing, */ int p,int imgSizeX,int imgSizeY,int imgSizeZ,float originX,
	float originY,float originZ,float spacingX,float spacingY,float spacingZ, int gat_ign);
kernel void ignprocessing(global float* vol_data_src,global float* vol_data_dest,global float* gProjMatrix, /* not yet... float2 gridSpacing, */ int p,int imgSizeX,int imgSizeY,int imgSizeZ,float originX,
	float originY,float originZ,float spacingX,float spacingY,float spacingZ, int gat_ign);


kernel void backProjectPixelDrivenCL(
	image2d_t sino,
	global float* imgGrid,
	global float* destGrid,
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
	int gat_ign
	
	//float spacingU,
	//float spacingV,

) {

	
	
	//x and y are fixed here 
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	
	// check if inside image boundaries
	if ( x>= imgSizeX || y >= imgSizeY)
		return;
		
	const unsigned int ignfac = (1 + 2 * gat_ign);
	
	//projectionMatrix may be transponized
	float4 pos = {(float)x*spacingX-originX,(float)y*spacingY -originY,0.0, 1.0};

	float precomputeR = gProjMatrix[p*12 + 3]  + pos.y * gProjMatrix[p*12 + 1] + pos.x * gProjMatrix[p*12 + 0];
	float precomputeS = gProjMatrix[p*12 + 7]  + pos.y * gProjMatrix[p*12 + 5] + pos.x * gProjMatrix[p*12 + 4];
    float precomputeT = gProjMatrix[p*12 + 11] + pos.y * gProjMatrix[p*12 + 9] + pos.x * gProjMatrix[p*12 + 8];
		
	//we are going along z while x and y are fixed
	for (int z=0;z<imgSizeZ;++z) {
	
		//projectionMatrices
		pos.z = ((float)z * spacingZ) - originZ;
		float r = pos.z * gProjMatrix[p*12 + 2]  + precomputeR;
		float s = pos.z * gProjMatrix[p*12 + 6]  + precomputeS;
		float t = pos.z * gProjMatrix[p*12 + 10] + precomputeT;
		
		// compute projection coordinates 
		float denom = 1.0f / t; //denominator for dehomogeinzation
		float fu = r* denom;
		float fv = s* denom;
		float2 posUV = {fu+0.5f, fv+0.5f};  //,(float)p+0.5f,0.f};
		//float proj_val = read_imagef(sino, linearSampler,posUV).x;   // <--- correct for non-padded detector
		float cval = read_imagef(sino, linearSampler,posUV).x;   // <--- correct for non-padded detector
		cval = cval*denom*denom;
		
		
		// compute volume index for x,y,z
		int idx = x*imgSizeZ*imgSizeY*ignfac + y * imgSizeZ*ignfac + z*ignfac;
		int idx_ign = x*imgSizeZ*imgSizeY* ignfac + y * imgSizeZ*ignfac + z*ignfac + 1;//equals idx + 1
		unsigned int idxworst = idx_ign;
		
		
		// search smallest of the largest values
		// the smallest of the large values is the first one to be replaced
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (imgGrid[idx_ign + i] < imgGrid[idxworst])
				idxworst = idx_ign + i;
		}
		

		if (imgGrid[idxworst] < cval)
			imgGrid[idxworst] = cval;

		idxworst = idx_ign + 2 * gat_ign - 1;

		// search largest of the smallest values
		// the largest of the smallest values is the first one to be replaced
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (imgGrid[idx_ign + 2 * gat_ign - i - 1] > imgGrid[idxworst])
				idxworst = idx_ign + 2 * gat_ign - i - 1;
		}
		

		if (imgGrid[idxworst] > cval)
			imgGrid[idxworst] = cval;
		
		// add value to imageGrid
		//imgGrid[idx] += proj_val*denom*denom;
		//imgGrid[idx] += cval*denom*denom;
		imgGrid[idx] += cval;
		
	}
	
	/*ignprocessing(
	imgGrid, 
	destGrid, 
	gProjMatrix,
	p,
	imgSizeX,
	imgSizeY,
	imgSizeZ,
	originX,
	originY,
	originZ,
	spacingX,
	spacingY,
	spacingZ,
	gat_ign);*/
	
	
}

	
kernel void ignprocessing(
	global float* vol_data_src,
	global float* vol_data_dest,
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
	int gat_ign){
	
	//x and y are fixed here 
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	
	//const unsigned gat_ign = 4;
	
	// check if inside image boundaries
	if ( x>= imgSizeX || y >= imgSizeY)
		return;
		
	const unsigned int ignfac = (1 + 2 * gat_ign);
	const unsigned int vidx_src = x * imgSizeY * imgSizeZ * ignfac + y * imgSizeZ * ignfac;
	const unsigned int vidx_dest = x * imgSizeY * imgSizeZ + y * imgSizeZ;
	
	// we walk over the y-component
	for (unsigned int z = 0; z < imgSizeZ; ++z) {
		const unsigned int idx_src = vidx_src + z * ignfac;
		const unsigned int idx_dest = vidx_dest + z;
		
		vol_data_dest[idx_dest] = vol_data_src[idx_src];
		
		for(unsigned int i = 0; i < 2*gat_ign; ++i){
			const float cv = vol_data_src[idx_src + i + 1];

			if ((cv < 100000.0f) && (cv > -100000.0f))
				vol_data_dest[idx_dest] -= cv;
		}
	}
}




/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/