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

float2 intersectLines(float2 p1, float2 p2, float2 p3, float2 p4);

kernel void backprojectPixelDriven2DCL(
	/* not yet... float2 gridSpacing, */
	global image2d_t sino,
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
	// compute x, y from thread idx
	const unsigned int x = get_global_id(0);// x index
	const unsigned int y = get_global_id(1);// y index
	
	if (x >= gridSizeX || y >= gridSizeY) {
		return;
	}
	float normalizationFactor = maxBetaIndex / M_PI_F;
	int idx = x + y*gridSizeX;
	grid[idx] = 0;
	
	for(int b=0; b<maxBetaIndex; ++b){
		// compute beta [rad] and angular functions.
		float beta = deltaBeta * b;
		float cosBeta = cos(beta);
		float sinBeta = sin(beta);

		float2 a = {focalLength * cosBeta, focalLength * sinBeta};
		float2 p0 = {-maxT / 2.f * sinBeta, maxT / 2.f * cosBeta};
		
		// compute two points on the line through t and beta
		// We use PointND for points in 3D space and SimpleVector for directions.
		float2 point = {x-gridSizeX/2.0f, y-gridSizeY/2.0f};
		float2 origin = {0.0f,0.0f};
		
		float2 detectorPixel = intersectLines(point, a, p0, origin);
	 	if (isnan(detectorPixel.x))
	 		continue;
		float2 p = detectorPixel -p0;
		float len = length(p);
		if((p.x*p0.x+p.y*p0.y)>0)//truncation, wrong direction//FIXME
		//len=-len;
			continue;
		float t = len/deltaT -0.5f;
		float2 bt = {t+0.5f, b+0.5f};
		float val = read_imagef(sino, linearSampler, bt).x;
		//DistanceWeighting
		float radius = length(point);
		float phi = (float) ((M_PI_F/2) + atan2(point.y, point.x));
		float dWeight = (focalLength  +radius*sin(beta - phi))/focalLength;
		float valtemp = val / (dWeight*dWeight*normalizationFactor);

		grid[idx] += valtemp;
	} // end for
	
}

float2 intersectLines(float2 p1, float2 p2, float2 p3, float2 p4) {
	float dNom = (p1.x - p2.x)*(p3.y - p4.y) - (p1.y -p2.y)*(p3.x - p4.x);
	
	if( dNom < 0.000001 && dNom >-0.000001){
		float2 retValue = {NAN,NAN};
		return retValue;
	}
	float x = (p1.x*p2.y-p1.y*p2.x)*(p3.x-p4.x)-(p1.x-p2.x)*(p3.x*p4.y-p3.y*p4.x);
	float y = (p1.x*p2.y-p1.y*p2.x)*(p3.y-p4.y)-(p1.y-p2.y)*(p3.x*p4.y-p3.y*p4.x);
	
	x /=dNom;
	y /=dNom;
	float2 isectPt = {x,y};
	return isectPt;
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