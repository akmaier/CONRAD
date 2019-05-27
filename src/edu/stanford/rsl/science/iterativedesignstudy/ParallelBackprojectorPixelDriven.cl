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

// /* not yet :) IN  gridSpacing */
// IN  deltaTheta
// IN  deltaS
// IN  maxTheta
// IN  maxS
// buffers & images
// IN  grid
// OUT sino
kernel void backprojectPixelDriven2DOpenCL(
		global image2d_t sino,
		global float* img,
		int imgSizeX,
		int imgSizeY,
		float maxS,
		float deltaS,
		float deltaTheta,
		float spacingX,
		float spacingY,
		int maxSIndex,
		int index
) {
	const unsigned int x = get_global_id(0);
	const unsigned int y = get_global_id(1);

	if ((x > imgSizeX)||(y > imgSizeY)) {
		return;
	}

	int idx = x+y*imgSizeX;
	img[idx]=0.0;
	// compute theta [rad] and angular functions.
	const float theta = deltaTheta * index;
	const float cosTheta = cos(theta);
	const float sinTheta = sin(theta);

	float2 gridXH = {-(imgSizeX*spacingX)/2.f, -(imgSizeY*spacingY)/2.f};
	float2 dirDetector = {(sinTheta), (cosTheta)};
	float2 pixel;

	// voxel to world coordinates
	pixel.x = x*(spacingX)+gridXH.x;
	pixel.y = y*(spacingY)+gridXH.y;

	float s = dot(pixel, dirDetector);
	s+=maxS/2;
	s/=deltaS;

	if ((s >= 0) && (s < maxSIndex)){
		int2 pt = {s+0.5f, 0.5f};
		float val = read_imagef(sino, linearSampler, pt).x;

		img[idx]= val;
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
 * Copyright (C) 2010-2014 Andreas Maier, Karoline Kallis, Anja Jaeger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
