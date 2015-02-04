typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

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
		float respoffset,
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

		float zcoord = ((z+respoffset) * voxelSpacingZ) - offsetZ;

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
		//float proj_val = tex2D(gTex2D, fu+1.5, fv+1.5);


		// compute volume index for x,y,z
		unsigned long idx = z*gVolStride[1] + y*gVolStride[0] + x;
    

		// Because of dx/s and dy/s, the value itself is scaled by 1/s^2
		pVolume[idx] += proj_val * denom * denom ;
		//pVolume[idx] += proj_val;
	}
	
    return;
}


/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/