//IN imgGrid	image3d
//IN sinogram	buffer
//IN pmatrix	buffer
//

//__constant for cohen-sutherland 3D
typedef int OutCode;
__constant int INSIDE = 0;  // 000000
__constant int LEFT = 1;    // 000001
__constant int RIGHT = 2;   // 000010
__constant int BOTTOM = 4;  // 000100
__constant int TOP = 8;     // 001000
__constant int NEAR = 16;   // 010000
__constant int FAR = 32;    // 100000

float8 CohenSutherlandLineClip3D(
	float x0, float y0, float z0,
	float x1, float y1, float z1,
	float xmin, float ymin, float zmin,
	float xmax, float ymax, float zmax);

__constant float samplingRate = 3.0f;
__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

kernel void projectRayDrivenCL(
	image3d_t grid,
	global float* sino, 
	//global float* pmatrix
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
	int maxU,
	int maxV,
	float spacingU,
	float spacingV,
	float sourcePosX,
	float sourcePosY,
	float sourcePosZ,
	float pdx, float pdy, float pdz
	//float vx, float vy, float vz
	) {
	const unsigned int u = get_global_id(0);// u index
	const unsigned int v = get_global_id(1);// v index
	if (u >= maxU || v >= maxV) {
		return;
	}
	const float imgSizeX_MM = imgSizeX * spacingX;
	const float imgSizeY_MM = imgSizeY * spacingY;
	const float imgSizeZ_MM = imgSizeZ * spacingZ;
	const float4 originXYZ = {originX, originY, originZ, 0.0f};
	const float4 spacingXYZ = {spacingX, spacingY, spacingZ, 1.0f};
	const float2 spacingUV = {spacingU, spacingV};
	float4 posSource = {sourcePosX, sourcePosY, sourcePosZ, 0.0f};
	//compute position of detector element
	const float4 principalAxis = {pdx, pdy, pdz, 0.f};
	const float4 dirV = {0.f, 0.f, 1.f, 0.f};
	float4 dirU = cross(dirV, -principalAxis);
	//dirU = dirU / length(dirU);
	float4 detPos = (u-maxU/2.f) * dirU*spacingUV.x  + (v-maxV/2.f)*dirV*spacingUV.y;
	
	float4 dirSourceDetector = detPos - posSource;
	detPos = detPos + dirSourceDetector;//extend ray
if (true) { // activate clipping?
	//compute intersection between ray and object
	float xmin = - imgSizeX_MM / 2.f;
	float ymin = - imgSizeY_MM / 2.f;
	float zmin = - imgSizeZ_MM / 2.f;
	float xmax = imgSizeX_MM -1 - imgSizeX_MM / 2.f;
	float ymax = imgSizeY_MM -1 - imgSizeY_MM / 2.f;
	float zmax = imgSizeZ_MM -1 - imgSizeZ_MM / 2.f;

	float8 intersectionPoints = CohenSutherlandLineClip3D(posSource.x, posSource.y, posSource.z, detPos.x, detPos.y, detPos.z, xmin, ymin, zmin, xmax, ymax, zmax);
	if (isnan(intersectionPoints.x)) {
		const unsigned int idx = u + v*maxU;
		sino[idx] = 0;
		return; // no intersection
	}
	const float4 start = intersectionPoints.lo;
	const float4 end   = intersectionPoints.hi;
	
	//ray from source to current position on detector plane;
	dirSourceDetector = end - start;
	posSource = start;
} else {
	dirSourceDetector = detPos - posSource;
}
	
	//compute line integral
	const float distance = length(dirSourceDetector);
	dirSourceDetector /= (distance * samplingRate);
	
	float sum = .0f;
	// compute the integral along the line.
	for (float tLine = 0.f; tLine < distance * samplingRate; tLine += 1.f) {
		float4 current = (posSource + originXYZ + tLine * dirSourceDetector) / spacingXYZ + 0.5f;
		sum += read_imagef(grid, linearSampler, current).x;
	}

	// normalize by the number of interpolation points
	sum /= samplingRate;
	
	//write to sinogram
	const unsigned int idx = u + v*maxU;
	sino[idx] = sum;
}

// Cohen-Sutherland 3D

inline OutCode ComputeOutCode3D(float x, float y, float z, float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
	OutCode code;

	code = INSIDE;          // initialised as being inside of clip window

	if (x < xmin)           // to the left of clip window
		code |= LEFT;
	else if (x > xmax)      // to the right of clip window
		code |= RIGHT;
	if (y < ymin)           // below the clip window
		code |= BOTTOM;
	else if (y > ymax)      // above the clip window
		code |= TOP;
	if (z < zmin)           // near side of clip window
		code |= NEAR;
	else if (z > zmax)      // far side of clip window
		code |= FAR;

	return code;
}

// returns 	float8 intersections = {x0, y0, z0, 0.0f, x1, y1, z1, 0.0f};
// returns NAN in .0-argument if no intersection
// use intersections.lo and intersections.hi to get the two intersection points
float8 CohenSutherlandLineClip3D(
	float x0, float y0, float z0,
	float x1, float y1, float z1,
	float xmin, float ymin, float zmin,
	float xmax, float ymax, float zmax)
{
	// compute outcodes for P0, P1, and whatever point lies outside the clip rectangle
	OutCode outcode0 = ComputeOutCode3D(x0, y0, z0, xmin, ymin, zmin, xmax, ymax, zmax);
	OutCode outcode1 = ComputeOutCode3D(x1, y1, z1, xmin, ymin, zmin, xmax, ymax, zmax);
	bool accept = false;

	while (true) {
		if (!(outcode0 | outcode1)) { // Bitwise OR is 0. Trivially accept and get out of loop
			accept = true;
			break;
		} else if (outcode0 & outcode1) { // Bitwise AND is not 0. Trivially reject and get out of loop
			x0 = NAN;
			break;
		} else {
			// failed both tests, so calculate the line segment to clip
			// from an outside point to an intersection with clip edge
			float x, y, z;

			// At least one endpoint is outside the clip rectangle; pick it.
			OutCode outcodeOut = outcode0? outcode0 : outcode1;

			// Now find the intersection point;
			// use formulas y = y0 + slope * (y1 - y0), x = x0 + slope * (x1 - x0), z = z0 + slope * (z1 - z0)
			if (outcodeOut & FAR) {   // point is far of clip rectangle
				const float t = (zmax - z0) / (z1 - z0);
				x = x0 + (x1 - x0) * t;
				y = y0 + (y1 - y0) * t;
				z = zmax;
			} else if (outcodeOut & NEAR) {   // point is in front of clip rectangle
				const float t = (zmin - z0) / (z1 - z0);
				x = x0 + (x1 - x0) * t;
				y = y0 + (y1 - y0) * t;
				z = zmin;
			} else if (outcodeOut & TOP) {           // point is above the clip rectangle
				const float t = (ymax - y0) / (y1 - y0);
				x = x0 + (x1 - x0) * t;
				z = z0 + (z1 - z0) * t;
				y = ymax;
			} else if (outcodeOut & BOTTOM) { // point is below the clip rectangle
				const float t = (ymin - y0) / (y1 - y0);
				x = x0 + (x1 - x0) * t;
				z = z0 + (z1 - z0) * t;
				y = ymin;
			} else if (outcodeOut & RIGHT) {  // point is to the right of clip rectangle
				const float t = (xmax - x0) / (x1 - x0);
				y = y0 + (y1 - y0) * t;
				z = z0 + (z1 - z0) * t;
				x = xmax;
			} else if (outcodeOut & LEFT) {   // point is to the left of clip rectangle
				const float t = (xmin - x0) / (x1 - x0);
				y = y0 + (y1 - y0) * t;
				z = z0 + (z1 - z0) * t;
				x = xmin;
			}

			//NOTE:*****************************************************************************************

			/* if you follow this algorithm exactly(at least for c#), then you will fall into an infinite loop 
			in case a line crosses more than two segments. to avoid that problem, leave out the last else
			if(outcodeOut & LEFT) and just make it else*/

			//**********************************************************************************************

			// Now we move outside point to intersection point to clip
			// and get ready for next pass.
			if (outcodeOut == outcode0) {
				x0 = x;
				y0 = y;
				z0 = z;
				outcode0 = ComputeOutCode3D(x0, y0, z0, xmin, ymin, zmin, xmax, ymax, zmax);
			} else {
				x1 = x;
				y1 = y;
				z1 = z;
				outcode1 = ComputeOutCode3D(x1, y1, z1, xmin, ymin, zmin, xmax, ymax, zmax);
			}
		}
	}
	float8 intersections = {x0, y0, z0, 0.0f, x1, y1, z1, 0.0f};
	return intersections;
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/