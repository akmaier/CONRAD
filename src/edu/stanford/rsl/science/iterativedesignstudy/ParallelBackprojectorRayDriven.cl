// bi-linear texture sampler
__constant sampler_t linearSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant float2 GRID_SPACING = {1.0f, 1.0f};
__constant float2 POINT5 = {.5f, .5f};

//__constant for cohen-sutherland
typedef int OutCode;
__constant int INSIDE = 0; // 0000
__constant int LEFT = 1;   // 0001
__constant int RIGHT = 2;  // 0010
__constant int BOTTOM = 4; // 0100
__constant int TOP = 8;    // 1000

float4 CohenSutherlandLineClip(float x0, float y0, float x1, float y1, float xmin, float ymin, float xmax, float ymax);


// OpenCL kernel for ray-driven parallel forward-projection
//
// /* not yet :) IN  gridSpacing */
// IN  deltaTheta
// IN  deltaS
// IN  maxTheta
// IN  maxS
// buffers & images
// IN  grid
// OUT sino
kernel void projectRayDriven1DCL(
	image2d_t sino,
	global float* image,
	float maxS,
	float deltaS,
	float maxTheta,
	float deltaTheta,
	int maxSIndex,
	int maxThetaIndex,
	int index
) {
	return;
	// TODO -> interpolation
	
	// compute e, i from thread idx
	const unsigned int i = get_global_id(0); // s index
	if (i > maxSIndex) {
		return;
	}
	
	const float samplingRate = 3.f; // # of samples per pixel
	// compute theta [rad] and angular functions.
	const float theta = deltaTheta * index;
	const float cosTheta = cos(theta);
	const float sinTheta = sin(theta);
	
	const int gridSizeX = get_image_width(grid);
	const int gridSizeY = get_image_height(grid);
	float2 gridXH = {gridSizeX/2.f, gridSizeY/2.f};

	// compute s, the distance from the detector edge in WC [mm]
	const float s = deltaS * i - maxS / 2.f;

	// compute two points on the line through s and theta
	// We use PointND for Points in 3D space and SimpleVector for directions.
	// ensure p1 and p2 are outside of the image by setting t to a high value
	float t1 = 1000.f;
	float t2 = -1000.f;
	float2 p1 = {(s * cosTheta) - (t1 * sinTheta), (s * sinTheta) + (t1 * cosTheta)};
	float2 p2 = {(s * cosTheta) - (t2 * sinTheta), (s * sinTheta) + (t2 * cosTheta)};

	//compute intersection between ray and object
	float xmin = - gridSizeX / 2.f;
	float xmax = gridSizeX -1 - gridSizeX / 2.f;
	float ymax = gridSizeY -1 - gridSizeY / 2.f;
	float ymin = - gridSizeY / 2.f;
	
	float4 intersectionPoints = CohenSutherlandLineClip(p1.x, p1.y, p2.x, p2.y, xmin, ymin, xmax, ymax);
	float2 start = {intersectionPoints.x, intersectionPoints.y};
	float2 end   = {intersectionPoints.z, intersectionPoints.w};
	if (isnan(intersectionPoints.x))
	{
		const int idx = i;
		img[idx] = .0f;
		return; // no intersection
	}
	// transform to grid coords
	start += gridXH;
	end   += gridXH;

	// get the normalized increment
	float2 increment = end - start;
	float distance = length(increment);
	increment /= (distance * samplingRate);

	float sum = .0f;

	// compute the integral along the line.
	for (float t = 0.0; t < distance * samplingRate; t += 1.0f) {
		float2 current = (start + increment * t) / GRID_SPACING;
		current += POINT5;
		
	//TODO	
		const int idx = i;
		sum += read_imagef(grid, linearSampler, current).x;
		img[idx] = val;
		

		if (x + 1 > grid.getSize()[0] || y + 1 >= grid.getSize()[1]
							|| x < 0 || y < 0)
				continue;

		InterpolationOperators.addInterpolateLinear(grid, x, y, val);
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