/**
 *
 * projection.cl
 * basically implements the OpenGL pipeline as OpenCL code.
 * Added enhanced functionality simulate x-ray transmission images.
 *
 * triangle rasterization is implemented after joshbeam.com
 * (http://joshbeam.com/articles/triangle_rasterization/)
 *
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable



/**
 * draws a horizontal line into the buffer
 */
private void drawSpan(global float * buffer, int width, int x1, int x2, int y, int id){
	if (x1 > x2){
	   int temp = x1;
	   x1=x2;
	   x2=temp;
	}
	for (int i = x1; i <=x2; i++){
		buffer[(y*width)+i] = id;
	}
}

/**
 * draws a point into a buffer
 */
private void drawPoint(global float * buffer, int width, int x, int y, int id){
	buffer[(y*width)+x] = id;
}

/**
 * draws three points into a buffer
 */ 
private void drawPoints(global float * buffer, int width, int x1, int y1, int x2, int y2, int x3, int y3, int id){
	drawPoint(buffer, width, x1, y1, id);
	drawPoint(buffer, width, x2, y2, id);
	drawPoint(buffer, width, x3, y3, id);
}

/**
 * draws a triangle span into the buffer. Note that we use int4 to describe edges:
 * edge1.x = x1
 * edge1.y = y1
 * edge1.z = x2
 * edge1.w = y2
 */
private void drawSpans(global float * buffer, int width, float4 edge1, float4 edge2, int id){
	
	
	// calculate difference between the y coordinates
	// of the first edge and return if 0
	float e1ydiff = (float)(edge1.w - edge1.y);
	
	
	if(e1ydiff == 0.0f)
		return;

	// calculate difference between the y coordinates
	// of the second edge and return if 0
	float e2ydiff = (float)(edge2.w - edge2.y);
	if(e2ydiff == 0.0f)
		return;	
	
	
	
	
	// calculate differences between the x coordinates
	// and colors of the points of the edges
	float e1xdiff = (float)(edge1.z - edge1.x);
	float e2xdiff = (float)(edge2.z - edge2.x);	
	// calculate factors to use for interpolation
	// with the edges and the step values to increase
	// them by after drawing each span
	float factor1 = (float)(edge2.y - edge1.y) / e1ydiff;
	float factorStep1 = 1.0f / e1ydiff;
	float factor2 = 0.0f;
	float factorStep2 = 1.0f / e2ydiff;
	// loop through the lines between the edges and draw spans
	int start = round(edge2.y);
	int stop = round(edge2.w);
	for(int y = start; y < stop; y++) {
		// create and draw span
			
		drawSpan(buffer, width, edge1.x + (int)(e1xdiff * factor1),
				edge2.x + (int)(e2xdiff * factor2), y, id);
			
		// increase factors
		factor1 += factorStep1;
		factor2 += factorStep2;
		
	}
}

private void drawTriangle(global float * buffer, int width, float x1, float y1, float x2, float y2, float x3, float y3, int id){
	float4 edge1 = (float4){x1, y1, x2, y2};
	if (y1 > y2) edge1 = (float4){x2, y2, x1, y1};
	float4 edge2 = (float4){x2, y2, x3, y3};
	if (y2 > y3) edge2 = (float4){x3, y3, x2, y2};
	float4 edge3 = (float4){x1, y1, x3, y3};
	if (y1 > y3) edge3 = (float4){x3, y3, x1, y1};
	// compute longest Edge:
	float maxLength = edge1.w - edge1.y;
    int longEdge = 0;
    if (edge2.w - edge2.y > maxLength){
    	longEdge = 1;
    	maxLength = edge2.w - edge2.y;
    }
	if (edge3.w - edge3.y > maxLength){
    	longEdge = 2;
    }
    // move longest edge to edge1
    if (longEdge == 1){
    	float4 temp = edge2;
    	edge2 = edge1;
    	edge1 = temp;
    }
    if (longEdge == 2){
    	float4 temp = edge3;
    	edge3 = edge1;
    	edge1 = temp;
    }
	drawSpans(buffer, width, edge1, edge2, id);
	drawSpans(buffer, width, edge1, edge3, id);
	 
}

private int getCoordinate(int x, int y, int gridWidth){
	return ((y*gridWidth)+x)*3;
}


private float getIndexX(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[((y*gridWidth)+x)*3];
}

private float getIndexY(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[((y*gridWidth)+x)*3+1];
}

private float getIndexZ(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[((y*gridWidth)+x)*3+2];
}

kernel void drawTriangles(global float* coordinateBuffer, global float* screenBuffer, int width, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    
    int gridWidth = (int)sqrt((float)numElements);
	int x = iGID % gridWidth;
	int y = floor(((float)iGID)/gridWidth);
	int x2 = (x + 1)%gridWidth;
	int y2 = (y + 1)%gridWidth;
	
	drawTriangle(screenBuffer, width, 
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	id);
	drawTriangle(screenBuffer, width, 
	round(getIndexX(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexY(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	id);
	
}

void getSemaphor(volatile global int * semaphor) {
	int occupied = atom_xchg(semaphor, 1);
	while (occupied != 0) {
		occupied = atom_xchg(semaphor, 1);		
	}
}

 

 void releaseSemaphor(volatile global int * semaphor)
{
   int prevVal = atom_xchg(semaphor, 0);
}

/**
 * draws a horizontal line into the buffer
 */
private void drawSpanZBuffer(global float * buffer, global int * zBuffer, int width, int x1, float z1, int x2, float z2, int y, int id){
	if (x1 > x2){
	   int temp = x1;
	   x1=x2;
	   x2=temp;
	   float temp2 = z1;
	   z1 = z2;
	   z2 = temp2;
	}
	float zDiff = z1 - z2;
	float stepInc = zDiff / (x2 - x1);
	for (int i = x1; i <=x2; i++){
		int currentZ = z1 - (((float)(i - x1)) * stepInc);
		int old = atom_min(&zBuffer[(y*width)+i], currentZ);
		//int old = zBuffer[(y*width)+i];
		//zBuffer[(y*width)+i] = min(zBuffer[(y*width)+i], currentZ);
		if (old > currentZ) {
			buffer[(y*width)+i] = -currentZ;
		}
	}
}

/**
 * draws a triangle span into the buffer. Note that we use int8 to describe edges:
 * edge1.s0 = x1
 * edge1.s1 = y1
 * edge1.s2 = z1
 * edge1.s3 = 0
 * edge1.s4 = x2
 * edge1.s5 = y2
 * edge1.s6 = z2
 * edge1.s7 = 0
 */
private void drawSpansZBuffer(global float * buffer, global int * zBuffer, int width, float8 edge1, float8 edge2, int id){
	// calculate difference between the y coordinates
	// of the first edge and return if 0
	float e1ydiff = (float)(edge1.s5 - edge1.s1);
	
	if(e1ydiff == 0.0f)
		return;

	// calculate difference between the y coordinates
	// of the second edge and return if 0
	float e2ydiff = (float)(edge2.s5 - edge2.s1);
	if(e2ydiff == 0.0f)
		return;	
	// calculate differences between the x coordinates
	// and colors of the points of the edges
	float e1xdiff = (float)(edge1.s4 - edge1.s0);
	float e2xdiff = (float)(edge2.s4 - edge2.s0);
	float e1colordiff = (edge1.s6 - edge1.s2);
    float e2colordiff = (edge2.s6 - edge2.s2);	
	// calculate factors to use for interpolation
	// with the edges and the step values to increase
	// them by after drawing each span
	float factor1 = (float)(edge2.s1 - edge1.s1) / e1ydiff;
	float factorStep1 = 1.0f / e1ydiff;
	float factor2 = 0.0f;
	float factorStep2 = 1.0f / e2ydiff;
	// loop through the lines between the edges and draw spans
	for(int y = edge2.s1; y < edge2.s5; y++) {
		// create and draw span
		
		drawSpanZBuffer(buffer, zBuffer, width,
			edge1.s0 + (int)(e1xdiff * factor1),
			edge1.s2 + (e1colordiff * factor1),
			edge2.s0 + (int)(e2xdiff * factor2),
			edge2.s2 + (e2colordiff * factor2),
			y, id);

		// increase factors
		factor1 += factorStep1;
		factor2 += factorStep2;
		//drawPoint(buffer,width, edge1.x+1, y+1, edge1.x + (int)(e1xdiff * factor1));
		//drawPoint(buffer,width, edge1.x+1, y+10, edge2.x + (int)(e2xdiff * factor2));
	}
}

private void drawTriangleZBuffer(global float * buffer, global int * zBuffer, int width, int x1, int y1, float z1, int x2, int y2, float z2, int x3, int y3, float z3, int id){
	float8 edge1 = (float8){x1, y1, z1, 0.0f, x2, y2, z2, 0.0f};
	if (y1 > y2) edge1 = (float8){x2, y2, z2, 0.0f, x1, y1, z1, 0.0f};
	float8 edge2 = (float8){x2, y2, z2, 0.0f, x3, y3, z3, 0.0f};
	if (y2 > y3) edge2 = (float8){x3, y3, z3, 0.0f, x2, y2, z2, 0.0f};
	float8 edge3 = (float8){x1, y1, z1, 0.0f, x3, y3, z3, 0.0f};
	if (y1 > y3) edge3 = (float8){x3, y3, z3, 0.0f, x1, y1, z1, 0.0f};
	// compute longest Edge:
	int maxLength = edge1.s5 - edge1.s1;
    int longEdge = 0;
    if (edge2.s5 - edge2.s1 > maxLength){
    	longEdge = 1;
    	maxLength = edge2.s5 - edge2.s1;
    }
	if (edge3.s5 - edge3.s1 > maxLength){
    	longEdge = 2;
    }
    // move longest edge to edge1
    if (longEdge == 1){
    	float8 temp = edge2;
    	edge2 = edge1;
    	edge1 = temp;
    }
    if (longEdge == 2){
    	float8 temp = edge3;
    	edge3 = edge1;
    	edge1 = temp;
    }
	drawSpansZBuffer(buffer, zBuffer, width, edge1, edge2, id);
	drawSpansZBuffer(buffer, zBuffer, width, edge1, edge3, id); 
}


kernel void drawTrianglesZBuffer(global float* coordinateBuffer, global float* screenBuffer,  global int* zBuffer, int width, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int gridWidth = (int)sqrt((float)numElements);
	int x = iGID % gridWidth;
	int y = floor(((float)iGID)/gridWidth);
	int x2 = (x + 1)%gridWidth;
	int y2 = (y + 1)%gridWidth;
	
	drawTriangleZBuffer(screenBuffer, zBuffer, width, 
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexZ(coordinateBuffer, x2, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexZ(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexZ(coordinateBuffer, x2, y, gridWidth)),
	id);
	drawTriangleZBuffer(screenBuffer, zBuffer, width, 
	round(getIndexX(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexY(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexZ(coordinateBuffer, x, y, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexZ(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexZ(coordinateBuffer, x2, y, gridWidth)),
	id);
}

private float clipTriangle(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x, float y){
	float2 v0 = (float2){x2-x1, y2-y1};
	float2 v1 = (float2){x3-x1, y3-y1};
	float2 v2 = (float2){x-x1, y-y1};
	//float v0x = x2-x1;
	//float v0y = y2-y1;
	//float v1x = x3-x1;
	//float v1y = y3-y1;
	//float v2x = x-x1;
	//float v2y = y-y1;
	
	float dot00  = dot(v0, v0);//v0x*v0x+v0y*v0y;//
	float dot01  = dot(v0, v1);//v0x*v1x+v0y*v1y;//
	float dot02  = dot(v0, v2);//v0x*v2x+v0y*v2y;//
	float dot11  = dot(v1, v1);//v1x*v1x+v1y*v1y;//
	float dot12  = dot(v1, v2);//v1x*v2x+v1y*v2y;//
	// Compute barycentric coordinates
	float invDenom = 1.0f / ((dot00 * dot11) - (dot01 * dot01));
	float u = ((dot11 * dot02) - (dot01 * dot12)) * invDenom;
	float v = ((dot00 * dot12) - (dot01 * dot02)) * invDenom;
	float w = 1.0f - (u+v);
	if ((u >= 0) && (v >= 0) && (u + v <= 1)){
		return 1*((z1 * w) + (z2 * u) + (z3 *v)); 
	} else {
		return 0;
	}
}

kernel void fillMaxMinValues(global float * coordinateBuffer, global float * ranges, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    float2 max = (float2){-INFINITY,-INFINITY};
    float2 min = (float2){INFINITY,INFINITY};
    
    for (int i=0; i<numElements; i++){
    	int y2 = (iGID + 1)%numElements;
    	int coord = getCoordinate(i, iGID, numElements);
    	float2 current = (float2){coordinateBuffer[coord], coordinateBuffer[coord+1]};
    	max = fmax(max,current);
    	min = fmin(min,current);
    	coord = getCoordinate(i, y2, numElements);
    	current = (float2){coordinateBuffer[coord], coordinateBuffer[coord+1]};
    	max = fmax(max,current);
    	min = fmin(min,current);
    }
	ranges[(iGID*4)] = max.x;
	ranges[(iGID*4)+1] = max.y;
	ranges[(iGID*4)+2] = min.x;
	ranges[(iGID*4)+3] = min.y;
}  

kernel void drawTrianglesRayCastRanges(global float* coordinateBuffer, global float* ranges, global float* screenBuffer, int width, int controlPoints, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int gridWidth = (int)sqrt((float)controlPoints);
	int pixelx = iGID % width;
	int pixely = floor(((float)iGID)/width);
	float currentValue =0;
	if (screenBuffer[(pixely*width)+pixelx] != 0){
		currentValue = screenBuffer[(pixely*width)+pixelx];
	}
	int lasty = -1;
	int skip = 1;
	for (int i = 0; i < controlPoints; i++) {
		int x = i % gridWidth;
		int y = floor(((float)i)/gridWidth);
		int coordxy = getCoordinate(x, y, gridWidth);
		if (lasty != y) {
			if ( // triangle strip contains (x, y)
			ranges[(y*4)] >= pixelx &&
			ranges[(y*4)+1] >= pixely &&
			ranges[(y*4)+2] <= pixelx &&
			ranges[(y*4)+3] <= pixely){
				skip = 0; // do not skip
			} else {
				skip = 1; // skip
			}
			lasty = y;
		}
		if (skip == 0) {
		int x2 = (x + 1)%gridWidth;
		int y2 = (y + 1)%gridWidth;
		int coordx2y2 = getCoordinate(x2, y2, gridWidth);
		int coordxy2 = getCoordinate(x, y2, gridWidth);
		int coordx2y = getCoordinate(x2, y, gridWidth);
		
		
		float value = clipTriangle( 
			round(coordinateBuffer[coordx2y2]), 
			round(coordinateBuffer[coordx2y2+1]), 
			round(coordinateBuffer[coordx2y2+2]),
			round(coordinateBuffer[coordxy2]),
			round(coordinateBuffer[coordxy2+1]),
			round(coordinateBuffer[coordxy2+2]),
			round(coordinateBuffer[coordx2y]),
			round(coordinateBuffer[coordx2y+1]),
			round(coordinateBuffer[coordx2y+2]),
			pixelx, pixely);
		
		if (value<0 && currentValue > value) currentValue = value;
		value = clipTriangle( 
			round(coordinateBuffer[coordxy]), 
			round(coordinateBuffer[coordxy+1]), 
			round(coordinateBuffer[coordxy+2]),
			round(coordinateBuffer[coordxy2]),
			round(coordinateBuffer[coordxy2+1]),
			round(coordinateBuffer[coordxy2+2]),
			round(coordinateBuffer[coordx2y]),
			round(coordinateBuffer[coordx2y+1]),
			round(coordinateBuffer[coordx2y+2]),
			pixelx,pixely);
		if (value < 0 &&currentValue > value) currentValue = value;
		}
	}
	if (currentValue < 100000) 
		drawPoint(screenBuffer, width, pixelx, pixely, currentValue);

	
}


kernel void drawTrianglesRayCast(global float* coordinateBuffer, global float* screenBuffer, int width, int controlPoints, int start, int stop, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int gridWidth = (int)sqrt((float)controlPoints);
	int pixelx = iGID % width;
	int pixely = floor(((float)iGID)/width);
	float currentValue =0;
	if (screenBuffer[(pixely*width)+pixelx] != 0){
		currentValue = screenBuffer[(pixely*width)+pixelx];
	}
	for (int i = start; i < stop; i++) {
		int x = i % gridWidth;
		int y = floor(((float)i)/gridWidth);
		int x2 = (x + 1)%gridWidth;
		int y2 = (y + 1)%gridWidth;
		int coordx2y2 = getCoordinate(x2, y2, gridWidth);
		int coordxy2 = getCoordinate(x, y2, gridWidth);
		int coordx2y = getCoordinate(x2, y, gridWidth);
		int coordxy = getCoordinate(x, y, gridWidth);
		
		float value = clipTriangle( 
			round(coordinateBuffer[coordx2y2]), 
			round(coordinateBuffer[coordx2y2+1]), 
			round(coordinateBuffer[coordx2y2+2]),
			round(coordinateBuffer[coordxy2]),
			round(coordinateBuffer[coordxy2+1]),
			round(coordinateBuffer[coordxy2+2]),
			round(coordinateBuffer[coordx2y]),
			round(coordinateBuffer[coordx2y+1]),
			round(coordinateBuffer[coordx2y+2]),
			pixelx, pixely);
		
		if (value<0 &&currentValue > value) currentValue = value;
		value = clipTriangle( 
			round(coordinateBuffer[coordxy]), 
			round(coordinateBuffer[coordxy+1]), 
			round(coordinateBuffer[coordxy+2]),
			round(coordinateBuffer[coordxy2]),
			round(coordinateBuffer[coordxy2+1]),
			round(coordinateBuffer[coordxy2+2]),
			round(coordinateBuffer[coordx2y]),
			round(coordinateBuffer[coordx2y+1]),
			round(coordinateBuffer[coordx2y+2]),
			pixelx,pixely);
		if (value < 0 &&currentValue > value) currentValue = value;
	}
	if (currentValue < 100000) 
		drawPoint(screenBuffer, width, pixelx, pixely, currentValue);

	
}




/**
 * projection of a 3D point using a projection matrix on 2D. Third dimension saves the depth of the point.
 * 
 */
kernel void project(global const float * pMatrix, global float* outputBuffer, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    float4 point = (float4){outputBuffer[(iGID*3)],outputBuffer[(iGID*3)+1],outputBuffer[(iGID*3)+2],1.0f};
    float4 row1 = (float4){pMatrix[0],pMatrix[1],pMatrix[2],pMatrix[3]};
    float4 row2 = (float4){pMatrix[4],pMatrix[5],pMatrix[6],pMatrix[7]};
    float4 row3 = (float4){pMatrix[8],pMatrix[9],pMatrix[10],pMatrix[11]};
    outputBuffer[(iGID*3)+2] = dot(row3,point);
    outputBuffer[(iGID*3)+1] = dot(row2,point)/outputBuffer[(iGID*3)+2];
    outputBuffer[(iGID*3)] = dot(row1,point)/outputBuffer[(iGID*3)+2];
    if (outputBuffer[(iGID*3)+2]>0) outputBuffer[(iGID*3)+2] *= -1;
    //if (outputBuffer[(iGID*3)]<0) outputBuffer[(iGID*3)] *= -1;
    //if (outputBuffer[(iGID*3)+1]<0) outputBuffer[(iGID*3)+1] *= -1;
}

/**
 * projection of a translated 3D point using a projection matrix on 2D. Third dimension saves the depth of the point.
 * 
 */
kernel void projectTranslate(global const float * pMatrix, float x, float y, float z, global float* outputBuffer, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    float4 point = (float4){outputBuffer[(iGID*3)]+x,outputBuffer[(iGID*3)+1]+y,outputBuffer[(iGID*3)+2]+z,1.0f};
    float4 row1 = (float4){pMatrix[0],pMatrix[1],pMatrix[2],pMatrix[3]};
    float4 row2 = (float4){pMatrix[4],pMatrix[5],pMatrix[6],pMatrix[7]};
    float4 row3 = (float4){pMatrix[8],pMatrix[9],pMatrix[10],pMatrix[11]};
    outputBuffer[(iGID*3)+2] = dot(row3,point);
    outputBuffer[(iGID*3)+1] = dot(row2,point)/outputBuffer[(iGID*3)+2];
    outputBuffer[(iGID*3)] = dot(row1,point)/outputBuffer[(iGID*3)+2];
    if (outputBuffer[(iGID*3)+2]>0) outputBuffer[(iGID*3)+2] *= -1;
    //if (outputBuffer[(iGID*3)]<0) outputBuffer[(iGID*3)] *= -1;
    //if (outputBuffer[(iGID*3)+1]<0) outputBuffer[(iGID*3)+1] *= -1;
}

/**
 * projection of a 3D point using a projection matrix on 2D. Third dimension saves the depth of the point.
 * Also computes the normal direction of the triangles with respect to the viewing ray.
 */
kernel void projectTriangle(global const float * pMatrix, global float* outputBuffer, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    
    // compute normal directions:
    float4 ray = (float4){pMatrix[12], pMatrix[13], pMatrix[14], 0.0f};
    float4 normal;
    int gridWidth = (int)sqrt((float)numElements);
	int x = iGID % gridWidth;

	int y = floor(((float)iGID)/gridWidth);
	int x2 = (x + 1)%gridWidth;
	int y2 = (y + 1)%gridWidth;
	
	if ((x == 0 || x2 == 0)){ // opening in the surface possible
		int xcoord = 0;
		if (x2 == 0){ 
			xcoord = x;
		}
		float4 one = (float4){getIndexX(outputBuffer, xcoord, y, gridWidth),
			getIndexY(outputBuffer, xcoord, y, gridWidth),			
			getIndexZ(outputBuffer, xcoord, y, gridWidth), 0.f};
		float4 two = (float4){getIndexX(outputBuffer, xcoord, y2, gridWidth),
			getIndexY(outputBuffer, xcoord, y2, gridWidth),
			getIndexZ(outputBuffer, xcoord, y2, gridWidth), 0.0f};
		float4 three = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		if (distance(one, two) > 0.01f){ // we need to fill the hole
			int closingPoints = 2;
			float factor = ((float) gridWidth) / closingPoints;
			for (int i=0; i < closingPoints; i++){
				three += (float4){getIndexX(outputBuffer, xcoord, i * factor, gridWidth),
				getIndexY(outputBuffer, xcoord, i*factor, gridWidth),
				getIndexZ(outputBuffer, xcoord, i*factor, gridWidth), 0.0f};
			}
			three /= closingPoints;
			normal = cross(two-one, three-one);
			int dir = dot(normal, ray);			
		} 
		return;
	}	
	float4 one = (float4){ getIndexX(outputBuffer, x2, y2, gridWidth), 
	getIndexY(outputBuffer, x2, y2, gridWidth), 
	(getIndexZ(outputBuffer, x2, y2, gridWidth)), 0.0f};
	float4 two = (float4){ getIndexX(outputBuffer, x, y, gridWidth),
	(getIndexY(outputBuffer, x, y, gridWidth)),
	(getIndexZ(outputBuffer, x, y, gridWidth)), 0.0f};
	float4 three = (float4){ (getIndexX(outputBuffer, x2, y, gridWidth)),
	(getIndexY(outputBuffer, x2, y, gridWidth)),
	(getIndexZ(outputBuffer, x2, y, gridWidth)), 0.0f};
	
	
	
	one = (float4){round(getIndexX(outputBuffer, x, y, gridWidth)), 
	round(getIndexY(outputBuffer, x, y, gridWidth)), 
	(getIndexZ(outputBuffer, x, y, gridWidth)), 0.0f};
	two = (float4){ (getIndexX(outputBuffer, x, y2, gridWidth)),
	(getIndexY(outputBuffer, x, y2, gridWidth)),
	(getIndexZ(outputBuffer, x, y2, gridWidth)), 0.0f};
	three = (float4){(getIndexX(outputBuffer, x2, y2, gridWidth)),
	(getIndexY(outputBuffer, x2, y2, gridWidth)),
	(getIndexZ(outputBuffer, x2, y2, gridWidth)), 0.0f};
	// compute normal direction:
    
    
    // Actual projection to 2D:
    
    
    float4 point = (float4){outputBuffer[(iGID*3)],outputBuffer[(iGID*3)+1],outputBuffer[(iGID*3)+2],1.0f};
    float4 row1 = (float4){pMatrix[0],pMatrix[1],pMatrix[2],pMatrix[3]};
    float4 row2 = (float4){pMatrix[4],pMatrix[5],pMatrix[6],pMatrix[7]};
    float4 row3 = (float4){pMatrix[8],pMatrix[9],pMatrix[10],pMatrix[11]};
    outputBuffer[(iGID*3)+2] = dot(row3,point);
    outputBuffer[(iGID*3)+1] = dot(row2,point)/outputBuffer[(iGID*3)+2];
    outputBuffer[(iGID*3)] = dot(row1,point)/outputBuffer[(iGID*3)+2];
    
    	
    
    
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
