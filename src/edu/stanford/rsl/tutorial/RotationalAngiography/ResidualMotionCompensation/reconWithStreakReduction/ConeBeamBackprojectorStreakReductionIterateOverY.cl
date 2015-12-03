__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

kernel void backProjectPixelDrivenCL(image2d_t sino,global float* imgGrid,global float* destGrid,global float* gProjMatrix, /* not yet... float2 gridSpacing, */ int p,int imgSizeX,int imgSizeY,int imgSizeZ,float originX,
	float originY,float originZ,float spacingX,float spacingY,float spacingZ);
kernel void ignprocessing(global float* vol_data_src,global float* vol_data_dest,global float* gProjMatrix, /* not yet... float2 gridSpacing, */ int p,int imgSizeX,int imgSizeY,int imgSizeZ,float originX,
	float originY,float originZ,float spacingX,float spacingY,float spacingZ);

kernel void backProjectPixelDrivenCL(
	image2d_t sino,
	global float* vol_data,
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
	float spacingZ
	
	//float spacingU,
	//float spacingV,

){


	const unsigned gat_ign = 3;
	
	const unsigned int ix = get_global_id(0);// x index
	const unsigned int iz = get_global_id(2);// z index
	
	// check if inside image boundaries
	if ( ix>= imgSizeX || iz >= imgSizeZ)
		return;
	
	const unsigned int ignfac = (1 + 2 * gat_ign);
	//const unsigned int vidx = iz * imgSizeX * imgSizeY * ignfac + ix * ignfac;
	const unsigned int vidx = ix * imgSizeZ * imgSizeY *ignfac +  iz * ignfac;
	
	//projectionMatrix may be transponed
	//float4 pos = {(float)ix*spacingX-originX,(float)y*spacingY -originY,0.0, 1.0};
	
	float4 pos = {(float)ix*spacingX - originX,0.0,(float)iz*spacingZ - originZ, 1.0};
	float precomputeR = pos.z * gProjMatrix[p*12 + 2]  +  gProjMatrix[p*12 + 3] + pos.x * gProjMatrix[p*12 + 0];
	float precomputeS = pos.z * gProjMatrix[p*12 + 6]  +  gProjMatrix[p*12 + 7] + pos.x * gProjMatrix[p*12 + 4];
    float precomputeT = pos.z * gProjMatrix[p*12 + 10] +  gProjMatrix[p*12 + 11] + pos.x * gProjMatrix[p*12 + 8];
    
    for (unsigned int iy = 0; iy < imgSizeY; ++iy) {
	
		//projectionMatrices
		pos.y = ((float)iy*spacingY) - originY;
		float r = pos.y * gProjMatrix[p*12 + 1]  + precomputeR;
		float s = pos.y * gProjMatrix[p*12 + 5]  + precomputeS;
		float t = pos.y * gProjMatrix[p*12 + 9] + precomputeT;
		
		
		float denom = 1.0f / t; //denominator for dehomogeinzation
		float fu = r* denom;
		float fv = s* denom;
		float2 posUV = {fu+0.5f, fv+0.5f};  //,(float)p+0.5f,0.f};
		float cval = read_imagef(sino, linearSampler,posUV).x;   // <--- correct for non-padded detector
		cval = cval*denom*denom;
		
		
		//unsigned int idx_ign = vidx + iy * imgSizeX * ignfac + 1;
		unsigned int idx_ign = vidx + iy * imgSizeZ * ignfac + 1;
		unsigned int idxworst = idx_ign;

		// search smallest of the largest values
		// because the smallest of the largest values is the first to fall out 
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (vol_data[idx_ign + i] < vol_data[idxworst])
				idxworst = idx_ign + i;
		}
		
		// if value in ignore volume is smaller than cval, value in ignore volume = cval to find largest values
		if (vol_data[idxworst] < cval)
			vol_data[idxworst] = cval;

		idxworst = idx_ign + 2 * gat_ign - 1;

		// search largest of the smallest values
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (vol_data[idx_ign + 2 * gat_ign - i - 1] > vol_data[idxworst])
				idxworst = idx_ign + 2 * gat_ign - i - 1;
		}

		if (vol_data[idxworst] > cval)
			vol_data[idxworst] = cval;
		
		//vol_data[vidx + iy * imgSizeX * ignfac] += cval;
		vol_data[vidx + iy * imgSizeZ * ignfac] += cval;
	}
	
	ignprocessing(
	vol_data, 
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
	spacingZ);
	
	
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
	float spacingZ){
	
	const unsigned int ix = get_global_id(0);// x index
	const unsigned int iy = get_global_id(1);// y index
	const unsigned int iz = get_global_id(2);// z index
	
	const unsigned gat_ign = 3;
	
	
	// check if the volume coordinates are valid
	if ((ix >= imgSizeX) || (iz >= imgSizeZ))
		return;

	const unsigned int ignfac = (1 + 2 * gat_ign);
	
	//const unsigned int vidx_src = iz * imgSizeX * imgSizeY * ignfac + ix * ignfac;
	//const unsigned int vidx_dest = iz * imgSizeX * imgSizeY + ix;
	
	const unsigned int vidx_src = ix * imgSizeZ * imgSizeY * ignfac + iz*ignfac;
	const unsigned int vidx_dest = ix * imgSizeZ * imgSizeY + iz;

	// we walk over the y-component
	for (unsigned int iy = 0; iy < imgSizeY; ++iy) {
		//const unsigned int idx_src = vidx_src + iy * imgSizeX * ignfac;
		//const unsigned int idx_dest = vidx_dest + iy * imgSizeX;
		
		const unsigned int idx_src = vidx_src + iy * imgSizeZ * ignfac;
		const unsigned int idx_dest = vidx_dest + iy * imgSizeZ;

		vol_data_dest[idx_dest] = vol_data_src[idx_src];
				
		for (unsigned int i = 0; i < 2 * gat_ign; ++i) {
			const float cv = vol_data_src[idx_src + i + 1];

			if ((cv < 100000.0f) && (cv > -100000.0f))
				vol_data_dest[idx_dest] -= cv;
		}
	}
}
	
	
