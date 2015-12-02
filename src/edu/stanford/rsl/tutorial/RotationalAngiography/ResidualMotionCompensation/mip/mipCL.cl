
typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;


/* --------------------------------------------------------------------------
 *
 *
 *    Ray-tracing algorithm implementation in CUDA kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

float project_ray(
	float sx, float sy, float sz,	// X-ray source position
	float rx, float ry, float rz,	// Ray direction
	float stepsize, image3d_t gTex3D,			// ALPHA_STEP_SIZE Step size in ray direction
	__constant TvoxelValue* gVolumeEdgeMinPoint,
	__constant TvoxelValue* gVolumeEdgeMaxPoint)
{
    // Step 1: compute alpha value at entry and exit point of the volume
    float minAlpha, maxAlpha;
	minAlpha = 0;
	maxAlpha = INFINITY;

    if (0.0f != rx)
    {
        float reci = 1.0f / rx;
        float alpha0 = (gVolumeEdgeMinPoint[0] - sx) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[0] - sx) * reci;
        minAlpha = fmin(alpha0, alpha1);
        maxAlpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ry)
    {
        float reci = 1.0f / ry;
        float alpha0 = (gVolumeEdgeMinPoint[1] - sy) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[1] - sy) * reci;
        minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
        maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
    }

    if (0.0f != rz)
    {
        float reci = 1.0f / rz;
        float alpha0 = (gVolumeEdgeMinPoint[2] - sz) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[2] - sz) * reci;
        minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
        maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
    }

    // we start not at the exact entry point 
    // => we can be sure to be inside the volume
    //minAlpha += stepsize * 0.5f;
	float currentValue = 0;

    // Step 2: Cast ray if it intersects the volume
    float pixel = 0.0f;

      
    // Trapezoidal rule (interpolating function = piecewise linear func)
    float px, py, pz;
	
    // Entrance boundary
    // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
    //  whereas, SwVolume has voxel centers at integers.
    // For the initial interpolated value, only a half stepsize is
    //  considered in the computation.
    if (minAlpha < maxAlpha) {
        px = sx + minAlpha * rx;
        py = sy + minAlpha * ry;
        pz = sz + minAlpha * rz;
        pixel = 0.5 * read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
		if(pixel > currentValue) {
			currentValue = pixel;
		}
        minAlpha += stepsize;
    }

    // Mid segments
    while (minAlpha < maxAlpha)
    {
        px = sx + minAlpha * rx;
        py = sy + minAlpha * ry;
        pz = sz + minAlpha * rz;
        pixel = read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
		if(pixel > currentValue) {
			currentValue = pixel;
		}
        minAlpha += stepsize;
    }
    // Scaling by stepsize;
    pixel *= stepsize;
    
    // Last segment of the line
    //if (pixel > 0.0f ) {
    //    pixel -= 0.5 * stepsize * read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
    //    minAlpha -= stepsize;
    //    float lastStepsize = maxAlpha - minAlpha;
    //    pixel += 0.5 * lastStepsize * read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;

    //    px = sx + maxAlpha * rx;
    //    py = sy + maxAlpha * ry;
    //    pz = sz + maxAlpha * rz;
        // The last segment of the line integral takes care of the
        // varying length.
    //    pixel += 0.5 * lastStepsize * read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;

    //}
    // -------------------------------------------------------------------
    
    return currentValue;
} // ProjectRay



__kernel void projectKernel(__global TdetValue* pProjection,
 		int projWidth, 
		int projHeight,
 		float stepsize, 
 		__read_only image3d_t gTex3D,
 		// System geometry (for FP) parameters
		__constant Tcoord_dev* gVoxelElementSize,
		__constant Tcoord_dev* gVolumeEdgeMinPoint,
		__constant Tcoord_dev* gVolumeEdgeMaxPoint,
		__constant Tcoord_dev* gSrcPoint,
		__constant Tcoord_dev* gInvARmatrix,
		int projectionNumber)
{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int udx = mad24(gidx,locSizex,lidx);
    int vdx = mad24(gidy,locSizey,lidy);
    
	
	if (udx >= projWidth || vdx >= projHeight) 
	{
		return;
	}
	
	int idx = mad24(vdx,projWidth,udx);
	
	float u = udx;
	float v = vdx;
    
    gSrcPoint+=projectionNumber * 3;
	gInvARmatrix+=projectionNumber * 9;

    // compute ray direction
    float rx = gInvARmatrix[2] + v * gInvARmatrix[1] + u * gInvARmatrix[0];
    float ry = gInvARmatrix[5] + v * gInvARmatrix[4] + u * gInvARmatrix[3];
    float rz = gInvARmatrix[8] + v * gInvARmatrix[7] + u * gInvARmatrix[6];

    // normalize ray direction
    float normFactor = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
    rx *= normFactor;
    ry *= normFactor;
    rz *= normFactor;

    // compute forward projection
    float pixel = project_ray(
            gSrcPoint[0], 
            gSrcPoint[1], 
            gSrcPoint[2], 
            rx, ry, rz, stepsize, gTex3D,
			gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint);
			
    // normalize pixel value to world coordinate system units
    pixel *= sqrt((rx * gVoxelElementSize[0])*(rx * gVoxelElementSize[0])
                    + (ry * gVoxelElementSize[1])*(ry * gVoxelElementSize[1])
                    + (rz * gVoxelElementSize[2])*(rz * gVoxelElementSize[2]));
    
    pProjection[idx] = pixel;

    return;
};
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
