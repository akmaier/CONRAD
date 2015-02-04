
typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

float project_edges_at_ray(
	float sx, float sy, float sz,	// X-ray source position
	float rx_l, float ry_l, float rz_l,	// Ray directions of left neighbor
	float rx_r, float ry_r, float rz_r,	// Ray directions of right neighbor
	float ux, float uy, float uz,	// Detector u coordinate vector
	float stepsize, image3d_t gTex3D,			// ALPHA_STEP_SIZE Step size in ray direction
	__constant TvoxelValue* gVolumeEdgeMinPoint,
	__constant TvoxelValue* gVolumeEdgeMaxPoint)
{
    // Step 1: compute alpha value at entry and exit point of the volume
    float minAlpha, maxAlpha;
	minAlpha = 0;
	maxAlpha = INFINITY;

    if (0.0f != rx_l)
    {
        float reci = 1.0f / rx_l;
        float alpha0 = (gVolumeEdgeMinPoint[0] - sx) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[0] - sx) * reci;
        minAlpha = fmin(alpha0, alpha1);
        maxAlpha = fmax(alpha0, alpha1);
    }

    if (0.0f != ry_l)
    {
        float reci = 1.0f / ry_l;
        float alpha0 = (gVolumeEdgeMinPoint[1] - sy) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[1] - sy) * reci;
        minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
        maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
    }

    if (0.0f != rz_l)
    {
        float reci = 1.0f / rz_l;
        float alpha0 = (gVolumeEdgeMinPoint[2] - sz) * reci;
        float alpha1 = (gVolumeEdgeMaxPoint[2] - sz) * reci;
        minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
        maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
    }


    // Step 2: Cast ray if it intersects the volume
    float pixel = 0.0f;

      
    // Trapezoidal rule (interpolating function = piecewise linear func)
    float px, py, pz;
    minAlpha += stepsize;

    // Mid segments
    while (minAlpha < maxAlpha)
    {
        px = sx + minAlpha * rx_l;
        py = sy + minAlpha * ry_l;
        pz = sz + minAlpha * rz_l;
        float pixel_cur = read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py +  0.5f, pz - gVolumeEdgeMinPoint[2],0)).x;
        px = sx + minAlpha * rx_r;
        py = sy + minAlpha * ry_r;
        pz = sz + minAlpha * rz_r;           
        pixel_cur = fabs(pixel_cur-read_imagef(gTex3D, sampler, (float4)(px + 0.5f, py + 0.5f, pz - gVolumeEdgeMinPoint[2],0)).x);   
        pixel = max(pixel, pixel_cur);              
        minAlpha += stepsize;
    }

    
    
    return pixel;
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
    
	
	if (udx == 0 || vdx == 0 || udx >= projWidth-1 || vdx >= projHeight-1) 
	{
		return;
	}
	
	int idx = mad24(vdx,projWidth,udx);
	
	float u = udx;
	float v = vdx;
    
    gSrcPoint+=projectionNumber * 3;
	gInvARmatrix+=projectionNumber * 9;

    
    // derivative in u direction
    
    // compute left neighbor ray direction
    float rx_l = gInvARmatrix[2] + v * gInvARmatrix[1] + (u-1) * gInvARmatrix[0];
    float ry_l = gInvARmatrix[5] + v * gInvARmatrix[4] + (u-1) * gInvARmatrix[3];
    float rz_l = gInvARmatrix[8] + v * gInvARmatrix[7] + (u-1) * gInvARmatrix[6];

    // normalize ray direction
    float normFactor = 1.0f / (sqrt((rx_l * rx_l) + (ry_l * ry_l) + (rz_l * rz_l)));
    rx_l *= normFactor;
    ry_l *= normFactor;
    rz_l *= normFactor;

    // compute right neighbor ray direction
    float rx_r = gInvARmatrix[2] + v * gInvARmatrix[1] + (u+1) * gInvARmatrix[0];
    float ry_r = gInvARmatrix[5] + v * gInvARmatrix[4] + (u+1) * gInvARmatrix[3];
    float rz_r = gInvARmatrix[8] + v * gInvARmatrix[7] + (u+1) * gInvARmatrix[6];

    // normalize ray direction
    normFactor = 1.0f / (sqrt((rx_r * rx_r) + (ry_r * ry_r) + (rz_r * rz_r)));
    rx_r *= normFactor;
    ry_r *= normFactor;
    rz_r *= normFactor;

    // compute forward projection
    pProjection[idx] = project_edges_at_ray(
            gSrcPoint[0], 
            gSrcPoint[1], 
            gSrcPoint[2], 
            rx_l, ry_l, rz_l, rx_r, ry_r, rz_r, gInvARmatrix[0], gInvARmatrix[1], gInvARmatrix[2], stepsize, gTex3D,
			gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint);

	// derivative in v direction

    // compute left neighbor ray direction
    rx_l = gInvARmatrix[2] + (v-1) * gInvARmatrix[1] + u * gInvARmatrix[0];
    ry_l = gInvARmatrix[5] + (v-1) * gInvARmatrix[4] + u * gInvARmatrix[3];
    rz_l = gInvARmatrix[8] + (v-1) * gInvARmatrix[7] + u * gInvARmatrix[6];

    // normalize ray direction
    normFactor = 1.0f / (sqrt((rx_l * rx_l) + (ry_l * ry_l) + (rz_l * rz_l)));
    rx_l *= normFactor;
    ry_l *= normFactor;
    rz_l *= normFactor;

    // compute right neighbor ray direction
    rx_r = gInvARmatrix[2] + (v+1) * gInvARmatrix[1] + u * gInvARmatrix[0];
    ry_r = gInvARmatrix[5] + (v+1) * gInvARmatrix[4] + u * gInvARmatrix[3];
    rz_r = gInvARmatrix[8] + (v+1) * gInvARmatrix[7] + u * gInvARmatrix[6];

    // normalize ray direction
    normFactor = 1.0f / (sqrt((rx_r * rx_r) + (ry_r * ry_r) + (rz_r * rz_r)));
    rx_r *= normFactor;
    ry_r *= normFactor;
    rz_r *= normFactor;

    // compute forward projection
    pProjection[idx] += project_edges_at_ray(
            gSrcPoint[0], 
            gSrcPoint[1], 
            gSrcPoint[2], 
            rx_l, ry_l, rz_l, rx_r, ry_r, rz_r, gInvARmatrix[0], gInvARmatrix[1], gInvARmatrix[2], stepsize, gTex3D,
			gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint);


    return;
};


__kernel void projectKernel_UEdges(__global TdetValue* pProjection,
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
    
	
	if (udx == 0 || vdx == 0 || udx >= projWidth-1 || vdx >= projHeight-1) 
	{
		return;
	}
	
	int idx = mad24(vdx,projWidth,udx);
	
	float u = udx;
	float v = vdx;
    
    gSrcPoint+=projectionNumber * 3;
	gInvARmatrix+=projectionNumber * 9;

    
    // derivative in u direction
    
    // compute left neighbor ray direction
    float rx_l = gInvARmatrix[2] + v * gInvARmatrix[1] + (u-1) * gInvARmatrix[0];
    float ry_l = gInvARmatrix[5] + v * gInvARmatrix[4] + (u-1) * gInvARmatrix[3];
    float rz_l = gInvARmatrix[8] + v * gInvARmatrix[7] + (u-1) * gInvARmatrix[6];

    // normalize ray direction
    float normFactor = 1.0f / (sqrt((rx_l * rx_l) + (ry_l * ry_l) + (rz_l * rz_l)));
    rx_l *= normFactor;
    ry_l *= normFactor;
    rz_l *= normFactor;

    // compute right neighbor ray direction
    float rx_r = gInvARmatrix[2] + v * gInvARmatrix[1] + (u+1) * gInvARmatrix[0];
    float ry_r = gInvARmatrix[5] + v * gInvARmatrix[4] + (u+1) * gInvARmatrix[3];
    float rz_r = gInvARmatrix[8] + v * gInvARmatrix[7] + (u+1) * gInvARmatrix[6];

    // normalize ray direction
    normFactor = 1.0f / (sqrt((rx_r * rx_r) + (ry_r * ry_r) + (rz_r * rz_r)));
    rx_r *= normFactor;
    ry_r *= normFactor;
    rz_r *= normFactor;

    // compute forward projection
    pProjection[idx] = project_edges_at_ray(
            gSrcPoint[0], 
            gSrcPoint[1], 
            gSrcPoint[2], 
            rx_l, ry_l, rz_l, rx_r, ry_r, rz_r, gInvARmatrix[0], gInvARmatrix[1], gInvARmatrix[2], stepsize, gTex3D,
			gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint);

    return;
};


__kernel void projectKernel_VEdges(__global TdetValue* pProjection,
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
    
	
	if (udx == 0 || vdx == 0 || udx >= projWidth-1 || vdx >= projHeight-1) 
	{
		return;
	}
	
	int idx = mad24(vdx,projWidth,udx);
	
	float u = udx;
	float v = vdx;
    
    gSrcPoint+=projectionNumber * 3;
	gInvARmatrix+=projectionNumber * 9;

   

	// derivative in v direction

    // compute left neighbor ray direction
    float rx_l = gInvARmatrix[2] + (v-1) * gInvARmatrix[1] + u * gInvARmatrix[0];
    float ry_l = gInvARmatrix[5] + (v-1) * gInvARmatrix[4] + u * gInvARmatrix[3];
    float rz_l = gInvARmatrix[8] + (v-1) * gInvARmatrix[7] + u * gInvARmatrix[6];

    // normalize ray direction
    float normFactor = 1.0f / (sqrt((rx_l * rx_l) + (ry_l * ry_l) + (rz_l * rz_l)));
    rx_l *= normFactor;
    ry_l *= normFactor;
    rz_l *= normFactor;

    // compute right neighbor ray direction
    float rx_r = gInvARmatrix[2] + (v+1) * gInvARmatrix[1] + u * gInvARmatrix[0];
    float ry_r = gInvARmatrix[5] + (v+1) * gInvARmatrix[4] + u * gInvARmatrix[3];
    float rz_r = gInvARmatrix[8] + (v+1) * gInvARmatrix[7] + u * gInvARmatrix[6];

    // normalize ray direction
    normFactor = 1.0f / (sqrt((rx_r * rx_r) + (ry_r * ry_r) + (rz_r * rz_r)));
    rx_r *= normFactor;
    ry_r *= normFactor;
    rz_r *= normFactor;

    // compute forward projection
    pProjection[idx] = project_edges_at_ray(
            gSrcPoint[0], 
            gSrcPoint[1], 
            gSrcPoint[2], 
            rx_l, ry_l, rz_l, rx_r, ry_r, rz_r, gInvARmatrix[0], gInvARmatrix[1], gInvARmatrix[2], stepsize, gTex3D,
			gVolumeEdgeMinPoint, gVolumeEdgeMaxPoint);


    return;
};

/*
 * Copyright (C) 2010-2014 Michael Manhart, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
