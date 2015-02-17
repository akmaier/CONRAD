#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

// bi-linear texture sampler
__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
//__constant float GRID_SPACING = 1.0f;
__constant float2 GRID_SPACING = {1.0f, 1.0f};

//__constant for cohen-sutherland
typedef int OutCode;
 
__constant int INSIDE = 0; // 0000
__constant int LEFT = 1;   // 0001
__constant int RIGHT = 2;  // 0010
__constant int BOTTOM = 4; // 0100
__constant int TOP = 8;    // 1000

float4 CohenSutherlandLineClip(float x0, float y0, float x1, float y1, float xmin, float ymin, float xmax, float ymax);

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

// OpenCL kernel for ray-driven parallel backward-projection
//
// /* not yet :) IN  gridSpacing */
// IN  deltaTheta
// IN  deltaS
// IN  maxTheta
// IN  maxS
// buffers & images
// out  grid
// IN sino
kernel void backprojectRayDriven2DCL(
	/* not yet... float2 gridSpacing, */
	global float* sino,
	global float* grid,
	int gridSizeX,
	int gridSizeY, 
	float maxT,
	float deltaT,
	float maxBeta,
	float deltaBeta,
	float focalLength,
	int maxTIndex,
	int maxBetaIndex
	) {

	// compute e, i from thread idx
	const unsigned int b = get_global_id(0);// beta index
	const unsigned int t = get_global_id(1);// t index
	
	if (b >= maxBetaIndex || t >= maxTIndex) {
		return;
	}
	const float beta = deltaBeta * b;
	
	const float samplingRate = 3.0; // # of samples per pixel
	
	// compute starting point & direction of ray
	float2 a = {focalLength * cos(beta), focalLength *sin(beta)};
	float2 p0 = {-maxT/2.0 *sin(beta) ,maxT/2.0 * cos(beta)};
	float2 gridXH = {gridSizeX/2, gridSizeY/2};
	
	float2 dir = - p0;
	dir = normalize(dir);
	
	p0 += (0.5f * deltaT + t * deltaT)*dir;
	p0 = 2*p0-a;
	
	//compute intersection between ray and object
	float xmin =  - gridSizeX/2;
	float xmax = gridSizeX-1-gridSizeX/2;
	float ymax = gridSizeY-1-gridSizeY/2;
	float ymin = -gridSizeY / 2;
	
	float4 intersectionPoints = CohenSutherlandLineClip(a.x,a.y,p0.x,p0.y, xmin, ymin, xmax, ymax);
	float2 start = {intersectionPoints.x + gridSizeX/2, intersectionPoints.y + gridSizeY/2};
	float2 end   = {intersectionPoints.z + gridSizeX/2, intersectionPoints.w + gridSizeY/2};

	if (isnan(intersectionPoints.x)) return; // no intersection

	// get the normalized increment
	float2 increment = end - start;
	float distance = length(increment);
	increment /= (distance * samplingRate);

	const float val = sino[t +b*maxTIndex];
	const float normalization = ( samplingRate * maxBetaIndex / deltaT / M_PI_F);
	float sum = .0;
	// compute the integral along the line.
	for (float tLine = 0.0; tLine < distance * samplingRate; tLine += 1.0) {
		const float2 current = (start + increment * tLine) / GRID_SPACING;

		float valtemp = val;
if (true) {
		//DistanceWeighting
		float2 c2 = current - gridXH;
		float radius = length(c2);
		float phi = M_PI_F/2.0 + atan2(c2.y, c2.x);
		float dWeight = (focalLength + radius * sin(beta - phi))/focalLength;
		valtemp /= (dWeight*dWeight);
}

		int2 lower;
		lower.x = convert_int_rtz(floor(current.x));
		lower.y =convert_int_rtz(floor(current.y));
		if(lower.x < 0.0f*(gridSizeX-1) || lower.y < 0.0f*(gridSizeY-1))
			continue;
		if(lower.x > 1.0f*(gridSizeX-1) || lower.y > 1.0f*(gridSizeY -1))
			continue;

		int idxTL = lower.x + lower.y * gridSizeX;
		int idxTR = idxTL + 1;
		int idxBL = idxTL + gridSizeX;
		int idxBR = idxTR + gridSizeX;
		float2 d;
		d.x = current.x - convert_float(lower.x);
		d.y =current.y - convert_float(lower.y);
		valtemp /= normalization;

		float left = (1.0-d.x) * valtemp;
		float right = d.x * valtemp;
		float tl = (1.0-d.y) * left;
		float tr = (1.0-d.y) * right;
		float bl = d.y * left;
		float br = d.y * right;

		AtomicAdd(grid+idxTL, tl);
		AtomicAdd(grid+idxTR, tr);
		AtomicAdd(grid+idxBL, bl);
		AtomicAdd(grid+idxBR, br);
	}
	
	return;
}


inline OutCode ComputeOutCode(float x, float y, float xmin, float ymin, float xmax, float ymax)
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
 
        return code;
}

//returns NAN in .x-argument if no intersection
float4 CohenSutherlandLineClip(float x0, float y0, float x1, float y1, float xmin, float ymin, float xmax, float ymax)
{
        // compute outcodes for P0, P1, and whatever point lies outside the clip rectangle
        OutCode outcode0 = ComputeOutCode(x0, y0, xmin, ymin, xmax, ymax);
        OutCode outcode1 = ComputeOutCode(x1, y1, xmin, ymin, xmax, ymax);
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
                        float x, y;
 
                        // At least one endpoint is outside the clip rectangle; pick it.
                        OutCode outcodeOut = outcode0? outcode0 : outcode1;
 
                        // Now find the intersection point;
                        // use formulas y = y0 + slope * (x - x0), x = x0 + (1 / slope) * (y - y0)
                        if (outcodeOut & TOP) {           // point is above the clip rectangle
                                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
                                y = ymax;
                        } else if (outcodeOut & BOTTOM) { // point is below the clip rectangle
                                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
                                y = ymin;
                        } else if (outcodeOut & RIGHT) {  // point is to the right of clip rectangle
                                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
                                x = xmax;
                        } else if (outcodeOut & LEFT) {   // point is to the left of clip rectangle
                                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
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
                                outcode0 = ComputeOutCode(x0, y0, xmin, ymin, xmax, ymax);
                        } else {
                                x1 = x;
                                y1 = y;
                                outcode1 = ComputeOutCode(x1, y1, xmin, ymin, xmax, ymax);
                        }
                }
        }
       float4 intersects = {x0,y0,x1,y1};
       return intersects;
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/