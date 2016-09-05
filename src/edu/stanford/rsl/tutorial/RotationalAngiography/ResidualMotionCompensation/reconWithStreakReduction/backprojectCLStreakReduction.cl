
typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


__kernel void ignprocessing(
	__global TvoxelValue* vol_data_src,
	__global float* vol_data_dest,
	/* not yet... float2 gridSpacing, */
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ);
	
__kernel void backprojectKernel(
		__global TvoxelValue* pVolume,
		__global float* destGrid,
		int offset,
		int zoffset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__constant int* gVolStride,
		__constant Tcoord_dev* gProjMatrix);

//__constant__ int gVolStride[2];  

// System geometry related
//__constant__ Tcoord_dev gProjMatrix[12];  


/* --------------------------------------------------------------------------
 *
 *
 *    Voxel-based BP algorithm implementation in OpenCL kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

__kernel void backprojectKernel(
		__global TvoxelValue* pVolume,
		__global float* destGrid,
		int offset,
		int zoffset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__constant int* gVolStride,
		__constant Tcoord_dev* gProjMatrix)

{

	int gat_ign = 3;
	
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
	
	if (x >= gloSizex || y >= gloSizey)
		return;
		
	//int idx = vdx*projStride+udx;
    //int z = (int) blockIdx.z;

	// x and y will be constant in this thread;
    float xcoord = (x * voxelSpacingX) - offsetX;
    float ycoord = (y * voxelSpacingY) - offsetY;

	// precompute everything except z from the matrix multiplication
	float precomputeR = gProjMatrix[3]  + ycoord * gProjMatrix[1] + xcoord * gProjMatrix[0];
	float precomputeS = gProjMatrix[7]  + ycoord * gProjMatrix[5] + xcoord * gProjMatrix[4];
    float precomputeT = gProjMatrix[11] + ycoord * gProjMatrix[9] + xcoord * gProjMatrix[8];

	for (unsigned int z = get_group_id(2); z < get_group_id(2)+zoffset; z++){

		float zcoord = ((z) * voxelSpacingZ) - offsetZ;

		// compute homogeneous coordinates
		float r = zcoord * gProjMatrix[2]  + precomputeR;
		float s = zcoord * gProjMatrix[6]  + precomputeS;
		float t = zcoord * gProjMatrix[10] + precomputeT;

		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;

		//float proj_val = tex2D(gTex2D, fu, fv);   // <--- visible error!
		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + offset, fv + 0.5f)).x;
		//float proj_val = tex2D(gTex2D, fu+0.5f + offset, fv+0.5f);   // <--- correct for non-padded detector

		float cval = proj_val*denom*denom;
		
		// compute volume index for x,y,z
		unsigned long idx = (2*gat_ign+1)*z*gVolStride[1] + (2*gat_ign+1)*y*gVolStride[0] + (2*gat_ign+1)*x;
		unsigned long idx_ign = idx + 1;
		unsigned long idxworst = idx_ign;
		
				
		// search smallest of the largest values
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (pVolume[idx_ign + i] < pVolume[idxworst])
				idxworst = idx_ign + i;
		}

		if (pVolume[idxworst] < cval)
			pVolume[idxworst] = cval;

		idxworst = idx_ign + 2 * gat_ign - 1;

		// search largest of the smallest values
		for (unsigned int i = 1; i < gat_ign; ++i) {
			if (pVolume[idx_ign + 2 * gat_ign - i - 1] > pVolume[idxworst])
				idxworst = idx_ign + 2 * gat_ign - i - 1;
		}

		if (pVolume[idxworst] > cval)
			pVolume[idxworst] = cval;
		
		// add value to imageGrid
		pVolume[idx] += cval;
		
		
	}
	
	float imgSizeX = gVolStride[0];
	float imgSizeY = gVolStride[1];
	float imgSizeZ = gVolStride[2];
	
	ignprocessing(
	pVolume, 
	destGrid, 
	imgSizeX,
	imgSizeY,
	imgSizeZ,
	offsetX,
	offsetY, 
	offsetZ,
	voxelSpacingX,
	voxelSpacingY,
	voxelSpacingZ);
	
    //return;
}

__kernel void ignprocessing(
	__global TvoxelValue* vol_data_src,
	__global float* vol_data_dest,
	/* not yet... float2 gridSpacing, */
	int imgSizeX,
	int imgSizeY,
	int imgSizeZ,
	float originX,
	float originY,
	float originZ,
	float spacingX,
	float spacingY,
	float spacingZ){
	
	//x and y are fixed here 
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	
	const unsigned gat_ign = 3;
	
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
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/