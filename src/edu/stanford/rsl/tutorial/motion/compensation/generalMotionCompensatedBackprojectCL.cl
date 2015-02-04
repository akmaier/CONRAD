/*
 * Copyright (C) 2010-2014 Marco Bögel and Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
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
		int sizeX,
		int sizeY,
		int recoSizeZ,
		int offset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__constant Tcoord_dev* gProjMatrix,
		float projMultiplier,
		global float* motion)

{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
	
	if (x >= sizeX || y >= sizeY)
		return;
		
	unsigned int zStride = sizeX*sizeY;
	unsigned int yStride = sizeX;

	for (unsigned int z = 0; z < recoSizeZ; z++){


		// x and y will chance due to motion.
	    float4 coord = (float4){(x * voxelSpacingX) - offsetX, (y * voxelSpacingY) - offsetY, (z * voxelSpacingZ) - offsetZ, 0.0f};
	    uint4 voxelCoord = (uint4){x, y, z, 0};	
	    coord = applyMotion(motion, coord, voxelCoord, zStride, yStride);
	    
		float r = gProjMatrix[3] + coord.z * gProjMatrix[2] + coord.y * gProjMatrix[1] + coord.x * gProjMatrix[0];
		float s = gProjMatrix[7] + coord.z * gProjMatrix[6]  + coord.y * gProjMatrix[5] + coord.x * gProjMatrix[4];
		float t = gProjMatrix[11] + coord.z * gProjMatrix[10] + coord.y * gProjMatrix[9] + coord.x * gProjMatrix[8];

		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;

		// 0.5 pixel offset to hit pixel centers
		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + offset, fv + 0.5f)).x;

		// compute volume index for x,y,z
		unsigned long idx = z*zStride + y*yStride + x;
    

		// Distance weighting by 1/t^2
		pVolume[idx] += proj_val * denom * denom * projMultiplier;
		//pVolume[idx] += proj_val;
	}
	
    return;
}


