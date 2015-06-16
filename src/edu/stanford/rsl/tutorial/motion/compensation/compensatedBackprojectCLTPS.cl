typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

//__constant__ int gVolStride[2];  

// System geometry related
//__constant__ Tcoord_dev gProjMatrix[12];  

float euclideanDist(float4 vec1, float4 vec2);

/* --------------------------------------------------------------------------
 *
 *
 *    Voxel-based BP algorithm implementation in OpenCL kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

__kernel void backprojectKernel_returnMotion(
		__global TvoxelValue* pVolume,
		__global float* coeffsGlobal,
		__local float* coeffsLocal,
		__global float* ptsGlobal,
		__local float* ptsLocal,
		__global float* A,
		__global float* b,
		int ptsNr,
		int reconDimX,
		int reconDimY,
		int reconDimZ,
		int lineOffset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__global int* gVolStride,
		__global Tcoord_dev* gProjMatrix,
		float projMultiplier,
		__global TvoxelValue* deformationOutX,
		__global TvoxelValue* deformationOutY,
		__global TvoxelValue* deformationOutZ)
{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);
	
	int x = gidx*locSizex+lidx;
    int y = gidy*locSizey+lidy;
	
	if (x >= reconDimX || y >= reconDimY)
		return;
		
	int ptsFloatLength = mul24(ptsNr,3);
	int nrOfLocalThreads = mul24(locSizex,locSizey);
	int nrOfPtsReadInLoops = (ptsFloatLength%nrOfLocalThreads) ? (ptsFloatLength/nrOfLocalThreads + 1) : (ptsFloatLength/nrOfLocalThreads);
	
	for (int i = 0; i < nrOfPtsReadInLoops; i++){
		int linOffset = mad24(i,nrOfLocalThreads,lidx);
		int ptsIdx = mad24(lidy,locSizex,linOffset);
		if (ptsIdx >= ptsFloatLength)
			break;
		ptsLocal[ptsIdx]=ptsGlobal[ptsIdx];
		coeffsLocal[ptsIdx]=coeffsGlobal[ptsIdx];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE); // make sure that all points are in local memory
		

	// x and y will be constant in this thread;
    float xcoord = (x * voxelSpacingX) - offsetX;
    float ycoord = (y * voxelSpacingY) - offsetY;

	
	for (int z = 0; z < reconDimZ; z++){

		float4 coord = (float4)(xcoord, ycoord,(z * voxelSpacingZ)- offsetZ, 0.f);
		
		float xi = 0;
		float yi = 0;
		float zi = 0;
		
		for(int i = 0; i < ptsNr; i++) {
			int xc = mul24(i,3);
			int yc = mad24(i,3,1);
			int zc = mad24(i,3,2);
			float dist = euclideanDist(coord,(float4)(ptsLocal[xc], ptsLocal[yc], ptsLocal[zc],0.f));
			xi += coeffsLocal[xc]*dist;		
			yi += coeffsLocal[yc]*dist;
			zi += coeffsLocal[zc]*dist;

		}
		
		xi += dot((float4)(A[0],A[3],A[6],0.f),coord);
		yi += dot((float4)(A[1],A[4],A[7],0.f),coord);
		zi += dot((float4)(A[2],A[5],A[8],0.f),coord);
		
		xi += b[0];
		yi += b[1];
		zi += b[2];

		coord.x += xi;
		coord.y += yi;
		coord.z += zi;
		coord.w = 1.0f;
		
		// forward projection with projection matrix
		float r = dot((float4)(gProjMatrix[0],gProjMatrix[1],gProjMatrix[2],gProjMatrix[3]), coord);
		float s = dot((float4)(gProjMatrix[4],gProjMatrix[5],gProjMatrix[6],gProjMatrix[7]), coord);
		float t = dot((float4)(gProjMatrix[8],gProjMatrix[9],gProjMatrix[10],gProjMatrix[11]), coord);

		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;

		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + lineOffset, fv + 0.5f)).x;

		unsigned long idx = z*gVolStride[1] + y*gVolStride[0] + x;
		// Because of dx/s and dy/s, the value itself is scaled by 1/s^2
		pVolume[idx] += proj_val * denom * denom * projMultiplier;
		deformationOutX[idx] = xi;
		deformationOutY[idx] = yi;
		deformationOutZ[idx] = zi;
	}
    return;
}

/* --------------------------------------------------------------------------
 *
 *
 *    Voxel-based BP algorithm implementation in OpenCL kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

__kernel void backprojectKernel(
		__global TvoxelValue* pVolume,
		__global float* coeffsGlobal,
		__local float* coeffsLocal,
		__global float* ptsGlobal,
		__local float* ptsLocal,
		__global float* A,
		__global float* b,
		int ptsNr,
		int reconDimX,
		int reconDimY,
		int reconDimZ,
		int lineOffset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__global int* gVolStride,
		__global Tcoord_dev* gProjMatrix,
		float projMultiplier)
{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);
	
	int x = gidx*locSizex+lidx;
    int y = gidy*locSizey+lidy;
	
	if (x >= reconDimX || y >= reconDimY)
		return;
		
	int ptsFloatLength = mul24(ptsNr,3);
	int nrOfLocalThreads = mul24(locSizex,locSizey);
	int nrOfPtsReadInLoops = (ptsFloatLength%nrOfLocalThreads) ? (ptsFloatLength/nrOfLocalThreads + 1) : (ptsFloatLength/nrOfLocalThreads);
	
	for (int i = 0; i < nrOfPtsReadInLoops; i++){
		int linOffset = mad24(i,nrOfLocalThreads,lidx);
		int ptsIdx = mad24(lidy,locSizex,linOffset);
		if (ptsIdx >= ptsFloatLength)
			break;
		ptsLocal[ptsIdx]=ptsGlobal[ptsIdx];
		coeffsLocal[ptsIdx]=coeffsGlobal[ptsIdx];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE); // make sure that all points are in local memory
		

	// x and y will be constant in this thread;
    float xcoord = (x * voxelSpacingX) - offsetX;
    float ycoord = (y * voxelSpacingY) - offsetY;

	
	for (int z = 0; z < reconDimZ; z++){

		float4 coord = (float4)(xcoord, ycoord,(z * voxelSpacingZ)- offsetZ, 0.f);
		
		float xi = 0;
		float yi = 0;
		float zi = 0;
		
		for(int i = 0; i < ptsNr; i++) {
			int xc = mul24(i,3);
			int yc = mad24(i,3,1);
			int zc = mad24(i,3,2);
			float dist = euclideanDist(coord,(float4)(ptsLocal[xc], ptsLocal[yc], ptsLocal[zc],0.f));
			xi += coeffsLocal[xc]*dist;		
			yi += coeffsLocal[yc]*dist;
			zi += coeffsLocal[zc]*dist;

		}
		
		xi += dot((float4)(A[0],A[3],A[6],0.f),coord);
		yi += dot((float4)(A[1],A[4],A[7],0.f),coord);
		zi += dot((float4)(A[2],A[5],A[8],0.f),coord);
		
		xi += b[0];
		yi += b[1];
		zi += b[2];

		coord.x += xi;
		coord.y += yi;
		coord.z += zi;
		coord.w = 1.0f;
		
		// forward projection with projection matrix
		float r = dot((float4)(gProjMatrix[0],gProjMatrix[1],gProjMatrix[2],gProjMatrix[3]), coord);
		float s = dot((float4)(gProjMatrix[4],gProjMatrix[5],gProjMatrix[6],gProjMatrix[7]), coord);
		float t = dot((float4)(gProjMatrix[8],gProjMatrix[9],gProjMatrix[10],gProjMatrix[11]), coord);

		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;

		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + lineOffset, fv + 0.5f)).x;

		unsigned long idx = z*gVolStride[1] + y*gVolStride[0] + x;
		// Because of dx/s and dy/s, the value itself is scaled by 1/s^2
		pVolume[idx] += proj_val * denom * denom * projMultiplier;
	}
    return;
}

float euclideanDist(float4 vec1, float4 vec2) {
	return distance(vec1,vec2);
}

/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/


