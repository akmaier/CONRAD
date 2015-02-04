/**
 *  appendBuffer.cl
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

// Local memory size (Tesla C1060): 16 KB = 16384 B
// Sizeof(int) = 4
// 2 values per element (xy * id, currentZ)
// finally, subtract a constant to reserve some mem for register-swapping 
// Hence, max size is: ~ (16384 / 4 / 2) - 100
// Note that this limit mainly concerns the size of the triangles and the local workgroup size.
// The smaller the triangles we paint, the smaller the local size can be.
#define LOCAL_SIZE 2000
#define DEPTH_ACCURACY 500
#define DEPTH_RESOLUTION 50000
#define LOCAL_LIST 100

#define EPSILON 0.0f
#define EPSILON2 0.0f

private float getIndexX(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[((y*gridWidth)+x)*3];
}

private float getIndexY(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[(((y*gridWidth)+x)*3)+1];
}

private float getIndexZ(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[(((y*gridWidth)+x)*3)+2];
}

private void putAppendBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, int x, int y, float currentZ, int id){
	// get the next writeable element in the appendBuffer:
	//int appendBufferLocation = appendBufferPointer[0]++;
	// get the last written element from the pixelCoutner and store the new position:
	//int lastPixelPosition = atom_xchg(&pixelCounter[(y*width)+i], appendBufferLocation);
	// write into the localBuffer:
	//  - xy-location
	//  - current ID
	//  - current Z value
	int appendBufferLocation = atom_inc(&appendBufferPointer[0]);
	if (appendBufferLocation < LOCAL_SIZE-1) {	
		appendBuffer[(appendBufferLocation)*2] = (((y*width)+x)*300)+id;
		//appendBuffer[(appendBufferLocation)*3+1] = id;
		appendBuffer[(appendBufferLocation)*2+1] = -currentZ*1000.0f;;
	}
}

/**
 * draws a horizontal line into the appendBuffer
 * we convert the depth of XCat into units of microns by multiplication with 1000
 * int32 ranges from -2 billion to 2 billion (-2 to the power 31 to 2 to the power 31). Hence, we can
 * compute depth values from -2,147 to 2,147 meters. This should be sufficiently accurate for most
 * medical imaging tasks.
 */
private void drawSpanAppendBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, int x1, float z1, int x2, float z2, int y, int id){
	if (x1 == x2){
		//putAppendBufferLocal(appendBuffer, appendBufferPointer, width, x1, y, z1, id);
		return;
	}
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
		// This is the current value we would like to store in the buffer 
		int currentZ = (int)((z1 - (((float)(i - x1)) * stepInc)));
		putAppendBufferLocal(appendBuffer, appendBufferPointer, width, i, y, currentZ, id);
	}
}

private void putAppendBufferGlobal(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, int x, int y, float currentZ, int id){
	// get the next writeable element in the appendBuffer:
	int appendBufferLocation = atom_inc(&appendBufferPointer[0]);
	//int appendBufferLocation = appendBufferPointer[0]++;
	// get the last written element from the pixelCoutner and store the new position:
	int lastPixelPosition = atom_xchg(&pixelCounter[(y*width)+x], appendBufferLocation);
	// write into the appendBuffer:
	//  - pointer to next element
	//  - current Z value
	//  - current ID
	appendBuffer[(appendBufferLocation)*3] = lastPixelPosition;
	appendBuffer[(appendBufferLocation)*3+1] = -currentZ*1000.0f;
	appendBuffer[(appendBufferLocation)*3+2] = id;	
}

/**
 * draws a horizontal line into the appendBuffer
 * we convert the depth of XCat into units of microns by multiplication with 1000
 * int32 ranges from -2 billion to 2 billion (-2 to the power 31 to 2 to the power 31). Hence, we can
 * compute depth values from -2,147 to 2,147 meters. This should be sufficiently accurate for most
 * medical imaging tasks.
 */
private void drawSpanAppendBufferGlobal(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, int x1, float z1, int x2, float z2, int y, int id){
	if (x1 == x2) {
		//putAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, x1, y, z1, id);
		return;
	}
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
		// This is the current value we would like to store in the buffer
		//float interp = ((float) (i-x1)) / (x2-x1);
		//float currentZ = mix(z1,z2, interp);
		int currentZ = (int)((z1 - (((float)(i - x1)) * stepInc)));
		putAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, i, y, currentZ, id);
	}
}

/**
 * draws a triangle span into the append buffer. Note that we use int8 to describe edges:
 * edge1.s0 = x1
 * edge1.s1 = y1
 * edge1.s2 = z1
 * edge1.s3 = 0
 * edge1.s4 = x2
 * edge1.s5 = y2
 * edge1.s6 = z2
 * edge1.s7 = 0
 */
private void drawSpansAppendBufferGlobal(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, float8 edge1, float8 edge2, int id){
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
	//if (fabs(e1xdiff)+fabs(e2xdiff) < 1.0f) return;
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
		
		drawSpanAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width,
			edge1.s0 + (int)(e1xdiff * factor1),
			edge1.s2 + (e1colordiff * factor1),
			edge2.s0 + (int)(e2xdiff * factor2),
			edge2.s2 + (e2colordiff * factor2),
			y, id);

		//putAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge1.s0 + (int)(e1xdiff * factor1), y, edge1.s2 + (e1colordiff * factor1), id);
		//putAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge2.s0 + (int)(e2xdiff * factor2), y, edge2.s2 + (e2colordiff * factor2), id);


		// increase factors
		factor1 += factorStep1;
		factor2 += factorStep2;
		
	}
}

/**
 * draws a triangle span into the append buffer. Note that we use int8 to describe edges:
 * edge1.s0 = x1
 * edge1.s1 = y1
 * edge1.s2 = z1
 * edge1.s3 = 0
 * edge1.s4 = x2
 * edge1.s5 = y2
 * edge1.s6 = z2
 * edge1.s7 = 0
 */
private void drawSpansAppendBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, float8 edge1, float8 edge2, int id){
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
		
		drawSpanAppendBufferLocal(appendBuffer, appendBufferPointer, width,
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

private void drawTriangleAppendBufferGlobal(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, int x1, 
                                      int y1, float z1, int x2, int y2, float z2, int x3, int y3, float z3, int id){
	float8 edge1 = (float8){x1, y1, z1, 0.0f, x2, y2, z2, 0.0f};
	float8 edge2 = (float8){x2, y2, z2, 0.0f, x3, y3, z3, 0.0f};
	float8 edge3 = (float8){x1, y1, z1, 0.0f, x3, y3, z3, 0.0f};
	
	if (y1 > y3) edge3 = (float8){x3, y3, z3, 0.0f, x1, y1, z1, 0.0f};
	if (y2 > y3) edge2 = (float8){x3, y3, z3, 0.0f, x2, y2, z2, 0.0f};
	if (y1 > y2) edge1 = (float8){x2, y2, z2, 0.0f, x1, y1, z1, 0.0f};
	
	
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

    if (maxLength < EPSILON) return;
    float m = (edge1.s0 - edge1.s4) / (-maxLength);
    float t = edge1.s0 - (edge1.s4 * m);
    float e21 = fabs(((edge2.s1 * m)+t) - edge2.s0);
    float e22 = fabs(((edge2.s5 * m)+t) - edge2.s4);
    if ((e21 >EPSILON)||(e22 > EPSILON)){
		drawSpansAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge1, edge2, id);
	}
	float e31 = fabs(((edge3.s1 * m)+t) - edge3.s0);
    float e32 = fabs(((edge3.s5 * m)+t) - edge3.s4);
    if ((e31 >EPSILON)||(e32 > EPSILON)){
		drawSpansAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge1, edge3, id);
	} 

	//drawSpansAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge1, edge2, id);
	//drawSpansAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, edge1, edge3, id); 
}


private void drawTriangleAppendBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, int x1, 
                                      int y1, float z1, int x2, int y2, float z2, int x3, int y3, float z3, int id){
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
    	maxLength = edge3.s5 - edge3.s1;
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
    
    if (maxLength < EPSILON) return;
    float m = (edge1.s0 - edge1.s4) / (-maxLength);
    float t = edge1.s0 - (edge1.s4 * m);
    float e21 = fabs(((edge2.s1 * m)+t) - edge2.s0);
    float e22 = fabs(((edge2.s5 * m)+t) - edge2.s4);
    if ((e21 >EPSILON)||(e22 > EPSILON)){
		drawSpansAppendBufferLocal(appendBuffer, appendBufferPointer, width, edge1, edge2, id);
	}
	float e31 = fabs(((edge3.s1 * m)+t) - edge3.s0);
    float e32 = fabs(((edge3.s5 * m)+t) - edge3.s4);
    if ((e31 >EPSILON)||(e32 > EPSILON)){
		drawSpansAppendBufferLocal(appendBuffer, appendBufferPointer, width, edge1, edge3, id);
	} 
}

kernel void drawTrianglesAppendBufferGlobal(global float* coordinateBuffer, global int* appendBuffer, global int* appendBufferPointer, global int* pixelCounter, int width, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    float4 span1, span2, cross, normal;
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
		float4 one = (float4){getIndexX(coordinateBuffer, xcoord, y, gridWidth),
			getIndexY(coordinateBuffer, xcoord, y, gridWidth),			
			getIndexZ(coordinateBuffer, xcoord, y, gridWidth), 0.f};
		float4 two = (float4){getIndexX(coordinateBuffer, xcoord, y2, gridWidth),
			getIndexY(coordinateBuffer, xcoord, y2, gridWidth),
			getIndexZ(coordinateBuffer, xcoord, y2, gridWidth), 0.0f};
		float4 three = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		if (distance(one, two) > 0.01f){ // we need to fill the hole
			int closingPoints = 2;
			float factor = ((float) gridWidth) / closingPoints;
			for (int i=0; i < closingPoints; i++){
				three += (float4){getIndexX(coordinateBuffer, xcoord, i * factor, gridWidth),
				getIndexY(coordinateBuffer, xcoord, i*factor, gridWidth),
				getIndexZ(coordinateBuffer, xcoord, i*factor, gridWidth), 0.0f};
			}
			three /= closingPoints;
			drawTriangleAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width,  
			round(one.x), 
			round(one.y), 
			one.z,
			round(two.x),
			round(two.y),
			two.z,
			round(three.x),
			round(three.y),
			three.z,
			id);
			
		} 
		return;
	}	
	/*if (fabs(round(getIndexX(coordinateBuffer, x, y, gridWidth))-round(getIndexX(coordinateBuffer, x2, y, gridWidth))) < 2.5f) {
		span1 = (float4){(getIndexX(coordinateBuffer, x, y, gridWidth)-getIndexX(coordinateBuffer, x2, y2, gridWidth)),(getIndexY(coordinateBuffer, x, y, gridWidth)-getIndexY(coordinateBuffer, x2, y2, gridWidth)),((getIndexZ(coordinateBuffer, x, y, gridWidth)-getIndexZ(coordinateBuffer, x2, y2, gridWidth))/100.0f), 0.0f};
		span2 = (float4){(getIndexX(coordinateBuffer, x2, y, gridWidth)-getIndexX(coordinateBuffer, x2, y2, gridWidth)),(getIndexY(coordinateBuffer, x2, y, gridWidth)-getIndexY(coordinateBuffer, x2, y2, gridWidth)),((getIndexZ(coordinateBuffer, x2, y, gridWidth)-getIndexZ(coordinateBuffer, x2, y2, gridWidth))/100.0f), 0.0f};
		cross = (float4){((span1.s1*span2.s2)-(span1.s2*span2.s1)), ((span1.s2*span2.s0)-(span1.s0*span2.s2)), ((span1.s0*span2.s1)-(span1.s1*span2.s0)), 0.0f};
		normal = normalize(cross);
		if ((fabs(normal.s2) < 0.99f)) return;
	}*/	
	drawTriangleAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, 
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)), 
	(getIndexZ(coordinateBuffer, x2, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y, gridWidth)),
	(getIndexZ(coordinateBuffer, x, y, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	(getIndexZ(coordinateBuffer, x2, y, gridWidth)),
	id);
	/*if (fabs(round(getIndexX(coordinateBuffer, x, y2, gridWidth))-round(getIndexX(coordinateBuffer, x2, y2, gridWidth))) < 2.5f) {
		span1 = (float4){(getIndexX(coordinateBuffer, x, y, gridWidth)-getIndexX(coordinateBuffer, x2, y2, gridWidth)),(getIndexY(coordinateBuffer, x, y, gridWidth)-getIndexY(coordinateBuffer, x2, y2, gridWidth)),((getIndexZ(coordinateBuffer, x, y, gridWidth)-getIndexZ(coordinateBuffer, x2, y2, gridWidth))/100.0f), 0.0f};
		span2 = (float4){(getIndexX(coordinateBuffer, x, y2, gridWidth)-getIndexX(coordinateBuffer, x2, y2, gridWidth)),(getIndexY(coordinateBuffer, x, y2, gridWidth)-getIndexY(coordinateBuffer, x2, y2, gridWidth)),((getIndexZ(coordinateBuffer, x, y2, gridWidth)-getIndexZ(coordinateBuffer, x2, y2, gridWidth))/100.0f), 0.0f};
		cross = (float4){((span1.s1*span2.s2)-(span1.s2*span2.s1)), ((span1.s2*span2.s0)-(span1.s0*span2.s2)), ((span1.s0*span2.s1)-(span1.s1*span2.s0)), 0.0f};
		normal = normalize(cross);
		if ((fabs(normal.s2) < 0.99f)) return;	
	}*/
	drawTriangleAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, 
	round(getIndexX(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexY(coordinateBuffer, x, y, gridWidth)), 
	(getIndexZ(coordinateBuffer, x, y, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	(getIndexZ(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)),
	(getIndexZ(coordinateBuffer, x2, y2, gridWidth)),
	id);
	
}


kernel void drawTrianglesAppendBufferLocal(global float* coordinateBuffer, global int* appendBuffer, global int* appendBufferPointer, global int* pixelCounter, int width, int id, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	// We have two barriers in this kernel.
    	// Hence we have to wait for them before we exit.
    	// We did not have to do this on an Nvidia Tesla C1060.
    	// For the CPU AMD driver however we have to do it.
    	barrier(CLK_LOCAL_MEM_FENCE);
    	barrier(CLK_LOCAL_MEM_FENCE);
    	return;
    }
    int gridWidth = (int)sqrt((float)numElements);
	int x = iGID % gridWidth;
	int y = floor(((float)iGID)/gridWidth);
	int x2 = (x + 1)%gridWidth;
	int y2 = (y + 1)%gridWidth;
	
	local int localBuffer[(LOCAL_SIZE-1)*2];
    local int localBufferPointer[1];
    
    if (get_local_id(0) == 0){
    	localBufferPointer[0] = 0;
    }
	// This barrier ensures that the localBufferPointer is 0
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if ((x == 0 || x2 == 0)){ // opening in the surface possible
		int xcoord = 0;
		if (x2 == 0){ 
			xcoord = x;
		}
		float4 one = (float4){getIndexX(coordinateBuffer, xcoord, y, gridWidth),
			getIndexY(coordinateBuffer, xcoord, y, gridWidth),			
			getIndexZ(coordinateBuffer, xcoord, y, gridWidth), 0.f};
		float4 two = (float4){getIndexX(coordinateBuffer, xcoord, y2, gridWidth),
			getIndexY(coordinateBuffer, xcoord, y2, gridWidth),
			getIndexZ(coordinateBuffer, xcoord, y2, gridWidth), 0.0f};
		float4 three = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		if (distance(one, two) > 0.001f){ // we need to fill the hole
			int closingPoints = 2;
			float factor = ((float) gridWidth) / closingPoints;
			for (int i=0; i < closingPoints; i++){
				three += (float4){getIndexX(coordinateBuffer, xcoord, i * factor, gridWidth),
				getIndexY(coordinateBuffer, xcoord, i*factor, gridWidth),
				getIndexZ(coordinateBuffer, xcoord, i*factor, gridWidth), 0.0f};
			}
			three /= closingPoints;
			drawTriangleAppendBufferLocal(localBuffer, localBufferPointer, width,  
			round(one.x), 
			round(one.y), 
			one.z,
			round(two.x),
			round(two.y),
			two.z,
			round(three.x),
			round(three.y),
			three.z,
			id);
		}	
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}
	
	
	drawTriangleAppendBufferLocal(localBuffer, localBufferPointer, width, 
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)), 
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)), 
	(getIndexZ(coordinateBuffer, x2, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y, gridWidth)),
	(getIndexZ(coordinateBuffer, x, y, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y, gridWidth)),
	(getIndexZ(coordinateBuffer, x2, y, gridWidth)),
	id);
	drawTriangleAppendBufferLocal(localBuffer, localBufferPointer, width, 
	round(getIndexX(coordinateBuffer, x, y, gridWidth)), 
	round(getIndexY(coordinateBuffer, x, y, gridWidth)), 
	(getIndexZ(coordinateBuffer, x, y, gridWidth)),
	round(getIndexX(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x, y2, gridWidth)),
	(getIndexZ(coordinateBuffer, x, y2, gridWidth)),
	round(getIndexX(coordinateBuffer, x2, y2, gridWidth)),
	round(getIndexY(coordinateBuffer, x2, y2, gridWidth)),
	(getIndexZ(coordinateBuffer, x2, y2, gridWidth)),
	id);
	// This barrier ensures that all triangles were painted.
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0){
		if (localBufferPointer[0] > LOCAL_SIZE-1) {
			appendBufferPointer[0] = INFINITY;
		} else {
    		int globalLocation = atom_add(&appendBufferPointer[0], localBufferPointer[0]);
    		for (int i = 0; i < localBufferPointer[0]; i++){
    			// This is the current value we would like to store in the buffer
    			int xy = localBuffer[i*2]/300;
    			int id = localBuffer[i*2]%300; 
				int currentZ = localBuffer[i*2+1];

				// get the next writeable element in the appendBuffer:
				int appendBufferLocation = globalLocation+i;
				// get the last written element from the pixelCoutner and store the new position:
				int lastPixelPosition = atom_xchg(&pixelCounter[xy], appendBufferLocation);
				// write into the appendBuffer:
				//  - pointer to next element
				//  - current Z value
				//  - current ID
				appendBuffer[(appendBufferLocation)*3] = lastPixelPosition;
				appendBuffer[(appendBufferLocation)*3+1] = currentZ;
				appendBuffer[(appendBufferLocation)*3+2] = id;
    		}
    	}
    	
    }
	
}


/**
 * draws the maximum depth value onto the screen. This is similar to a z-Buffer.
 */
kernel void drawAppendBufferScreen(global float * screenBuffer, global int* appendBuffer, global int* pixelCounter, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int x = iGID % width;
	int y = floor(((float)iGID)/width);
	int nextElement = pixelCounter[(y*width)+x];
	float currentValue = 0;
	while (nextElement > 0){
		float currentElement = (float)(appendBuffer[(nextElement)*3+1])/1000.0f;
		currentValue = fmax(currentElement, currentValue);
		nextElement = appendBuffer[(nextElement)*3];
	}
	screenBuffer[(y*width)+x] = currentValue; 
}



void swap(int *a, int one, int two){
	int temp = a[one*2];
	int temp2 = a[one*2+1];
	a[one*2] = a[two*2];
	a[one*2+1] = a[two*2+1];
	a[two*2] = temp;
	a[two*2+1] = temp2;
}

void siftDown(int * a, int start, int end){
     //input:  end represents the limit of how far down the heap
     //              to sift.
	int root = start;
	while ((root * 2 + 1) <= end){ //(While the root has at least one child)
		int child = root * 2 + 1;//        (root*2 + 1 points to the left child)
        int swap2 = root;//        (keeps track of child to swap with)
		//(check if root is smaller than left child)
		if (a[swap2*2] < a[child*2]){
             swap2 = child;
		}
        //(check if right child exists, and if it's bigger than what we're currently swapping with)
		if (child < end && a[swap2*2] < a[(child+1)*2]){
			swap2 = child + 1;
		}
		//(check if we need to swap at all)
		if (swap2 != root){
             swap(a, root, swap2);
             root = swap2;//          (repeat to continue sifting down the child now)
        } else {
             return ;
		}
	}
}

void heapify(int * a, int count){
	//(start is assigned the index in a of the last parent node)
	int start = ((int)count / 2) - 1;
	while (start >= 0){
		//(sift down the node at index start to the proper place such that all nodes below
		// the start index are in heap order)
		siftDown(a, start, count-1);
		start--;
	}
}
//(after sifting down the root all nodes/elements are in heap order)




void heapSort(int *a, int count) {
     heapify(a, count);
     int end = count-1; //in languages with zero-based arrays the children are 2*i+1 and 2*i+2
     while (end > 0) {
         //(swap the root(maximum value) of the heap with the last element of the heap)
         swap(a, end, 0);
         //(put the heap back in max-heap order)
         siftDown(a, 0, end-1);
         //(decrease the size of the heap by one so that the previous max value will
         //stay in its proper placement)
         end--;
	}
}

void resolvePriority(int * rayList, global int* priorities, int length){
	int materialStack [10];
	int stackPointer = 0;
	materialStack[0] = rayList[1];
	for (int k=1; k < length;k++){
		int segment = rayList[k*2] - rayList[(k-1)*2];
		int ID = 0; 
		if (stackPointer >=0){
			ID = materialStack[stackPointer];
			int currentP = priorities[ID];
			for (int i=0; i<=stackPointer; i++){
				int priority = priorities[materialStack[i]];
				if(priority > currentP){
					currentP = priority;
					ID = materialStack[i];
				}
			}
		}
		rayList[(k-1)*2] = segment;
		rayList[(k-1)*2+1] = ID;
		int nextObjectID = rayList[(k*2)+1];
		int del = 0;
		for (int i = 0; i <= stackPointer; i++){
		    // same object
			if (materialStack[i] == nextObjectID){
				if (i < stackPointer) {
					for (int k = i; k <stackPointer; k++){
						materialStack[k] = materialStack[k+1];
					}
					i--;
				}
				del++;
			}
		}
		if (del > 0){
			stackPointer -= del;
		} else {
			stackPointer++;
			materialStack[stackPointer] = nextObjectID;
		}		
	}
}
 
float monochromaticAbsorption(int* segments, global float* mu, int length){
	float sum = 0;
	for (int i=0; i < length; i++){
	    // divide by two to get the real ID.			
		float att = mu[(segments[i*2+1])];
		float len = segments[i*2];
		sum += att*len;
	}
	// length is in [mm]
	// attenuation is in [g/cm^3]
	float t = sum/10000.0f;
	if(t < 0.0001f){
		return 0.0f;
	}
	return t;
}

void removePoint(int * data, int * i, int location){
	for (int k = location; k < i[0]; k++){
		data[(k)*2] = data[(k+1)*2]; 
		data[(k)*2+1] = data[(k+1)*2+1];
	}
	i[0]--;
}

/**
 * This function checks whether two elements in a row are from the same spline.
 * Prerequisite for this method is that the array was sorted.
 * matching points are marked.
 *
 * @param data the array of intersection points
 * @param i the length of the array. This is changed, if points are removed.
 */
void twoSubsequentElementsDetection(int * data, int *i){
    // id of the second intersection point
	int lastID = data[1] / 2;
	// parse the sorted array
	for (int l = 1; l < i[0]; l++){
		int currentID = data[(l*2)+1] / 2;
		// two times the same ID in a row
		if ((lastID == currentID)&&(abs(data[(l)*2] - data[(l-1)*2]) < DEPTH_RESOLUTION)){
		    // all three of them are likely to be located on a triangle parallel to the viewing direction.
			if (data[(l)*2+1] % 2 == 0) {
				data[(l)*2+1] += 1;
			}
			if (data[(l-1)*2+1] % 2 == 0) {
				data[(l-1)*2+1] += 1;
			}
		}
		lastID = currentID;
	}
}

/**
 *
 * Watch out which point to delete.
 *
 * @param data the array of intersection points
 * @param i the length of the array. This is changed, if points are removed.
 */
void twoSubsequentElementsDeletionWithCheck(int * data, int *i){
	int objects [20];
	int objectCount [20];
	int objectPointer = -1;
	// create a list of objects on the ray.
	for (int ii = 0; ii < i[0]; ii++){
		int p = data[(ii*2)+1]/2;
		bool found = false;
		for (int j =0; j<= objectPointer; j++){
			if (objects[j] == p){
				found = true;
				objectCount[j]++;
			}			
		}
		if (!found){
			objectPointer++;
			objects[objectPointer] = p;
			objectCount[objectPointer] = 1;
		}
	}
    // id of the second intersection point
	int lastID = data[1];
	for (int ii=0;ii <=objectPointer; ii++){
		if (objectCount[ii] % 2 == 0) {
			// parse the sorted array
			for (int l = 1; l < i[0]; l++){
				int currentID = data[(l*2)+1];
				// three times the same ID in a row
				if ((lastID %2 == 1) && (lastID == currentID) && (objects[ii] == currentID / 2)){
				    // all two uneven items directly behind each other.
				    // and we have an even number of this object along the ray => Most likely there is a problem
				    // Hence we have to get rind of one.
					if (l < i[0]/2) {
					    // we are at the beginning of the ray.
					    // delete the point after.
						data[(l)*2+1] = -1;
						removePoint(data, i, l);
						if (l < i[0]) {	 
							l--;
						} 
					} else {
					   // we are at the end of the ray
					   // delete the point before.
						data[(l-1)*2+1] = -1;
						removePoint(data, i, l-1);
						if (l < i[0]) {	 
							l--;
						} 
					}
				}
				lastID = data[(l)*2+1];
			}
		}
	}
}

/**
 *
 * Watch out which point to delete.
 *
 * @param data the array of intersection points
 * @param i the length of the array. This is changed, if points are removed.
 */
void twoSubsequentElementsDeletion(int * data, int *i){
	int objects [20];
	int objectCount [20];
	int objectCount2 [20];
	int objectPointer = -1;
	// create a list of objects on the ray.
	for (int ii = 0; ii < i[0]; ii++){
		int p = data[(ii*2)+1]/2;
		bool found = false;
		for (int j =0; j<= objectPointer; j++){
			if (objects[j] == p){
				found = true;
				objectCount[j]++;
			}			
		}
		if (!found){
			objectPointer++;
			objects[objectPointer] = p;
			objectCount[objectPointer] = 1;
			objectCount2[objectPointer] = 0;
		}
	}
    // id of the second intersection point
	int lastID = data[1];
	// parse the sorted array
	for (int l = 1; l < i[0]; l++){
		int currentID = data[(l*2)+1];
		// default: change orientation half way through the ray.
		int border = i[0]/2;
		int currentPos = l;
		bool doit = false;
		if (true) {
			// Now we determine the decision which point to remove on the respective ID.
			for (int hit=0;hit <= objectPointer; hit++){
				if (objects[hit] == currentID / 2){
					border = objectCount[hit]/2;
					objectCount2[hit]++;
					currentPos = objectCount2[hit]-1;
					doit = (objectCount[hit] % 2 == 0)&& (objectCount[hit] > 2);
				}
			}
		}
		// two times the same ID in a row
		if ((lastID %2 == 1) && (lastID == currentID) && doit){
		    // all two uneven items directly behind each other.
		    // Hence we have to get rind of one.
			if (currentPos < border) {
			    // we are at the beginning of the ray.
			    // delete the point after.
				data[(l)*2+1] = -1;
				// remove inconsistency candidate flag
				data[(l-1)*2+1] -= 1;
				removePoint(data, i, l);
				if (l < i[0]) {	 
					l--;
				} 
			} else {
			   // we are at the end of the ray
			   // delete the point before.
				data[(l-1)*2+1] = -1;
				// remove inconsistency candidate flag
				data[(l)*2+1] -= 1;
				removePoint(data, i, l-1);
				if (l < i[0]) {	 
					l--;
				}
				 
			}
		}
		lastID = data[(l)*2+1];
	}
}

/**
 * This function checks whether three elements in a row are from the same spline.
 * Prerequisite for this method is that the array was sorted.
 * Points will only be removed, if all three points are likely to be located on a surface parallel to the viewing direction, i. e. the id % 2 == 1.
 *
 * @param data the array of intersection points
 * @param i the length of the array. This is changed, if points are removed.
 */
void threeSubsequentElementsDetection(int * data, int *i){
    // id of the second intersection point
	int lastID = data[3];
	// id of the first intersection point.
	int lastLastID = data[1];
	// parse the sorted array
	for (int l = 2; l < i[0]; l++){
		int currentID = data[(l*2)+1];
		// three times the same ID in a row
		if ((lastLastID == lastID) && (lastID == currentID)){
		    // all three of them are likely to be located on a triangle parallel to the viewing direction.
			if (currentID % 2 == 1) {
				data[(l-1)*2+1] = -1;
				removePoint(data, i, l-1);
				if (l < i[0]) {	 
					l--;
				} 
			}
		}
		lastID = data[(l)*2+1];
		lastLastID = data[(l-1)*2+1];
	}
}


/**
 * Removes inconsistent points from the data. There must be an even number of intersection points along the ray. Otherwise, the ray cast is not clear.
 */
void inconsistentDataCorrection(int* data, int*length, bool lastTry){
	int objects [20];
	int objectCount [20];
	int objectPointer = -1;
	// create a list of objects on the ray.
	for (int i = 0; i < length[0]; i++){
		int p = data[(i*2)+1]/2;
		bool found = false;
		for (int j =0; j<=objectPointer; j++){
			if (objects[j] == p){
				found = true;
				objectCount[j]++;
			}			
		}
		if (!found){
			objectPointer++;
			objects[objectPointer] = p;
			objectCount[objectPointer] = 1;
		}
	}
	// resolve intersections along the ray.
	bool rewrite = false;
	for (int i = 0; i <= objectPointer; i++){
		float lastDepth = 0;
		int lastEntry = -1;
		int firstHit = -1;
		for (int j = 0; j < length[0]; j++){
			if (data[(j*2)+1]/2 == objects[i]){
				if (fabs(data[(j*2)] - lastDepth) <	DEPTH_ACCURACY){
					if (lastEntry != -1){
						if (lastEntry == firstHit) lastEntry = j;
						data[(lastEntry*2)+1] = -1;
						objectCount[i]--;
						rewrite = true;
					}
				} else {
					if (firstHit == -1) firstHit = j;
					lastDepth = data[(j*2)];
					lastEntry = j;
				}
			}
		}

		
		if (objectCount[i] %2 == 1){

			bool resolved = false;
			for (int j = 0; j < length[0]; j++){
				if (!resolved) {
					if (data[(j*2)+1]/2 == objects[i]){
						// only one hit of this object. no problem
						if (objectCount[i] == 1){
							data[(j*2)+1] = -1;
							rewrite = true;
							resolved = true;
							break;
						}
						// if this is not the first and the last item:
						if (j > 0 && j < length[0]-1) {
							// we found a sequence of three items in a row
							if ((data[((j-1)*2)+1]/2 == objects[i]) && (data[((j+1)*2)+1]/2 == objects[i])){
								// hence we can remove the center one!
								data[(j*2)+1] = -1;
								rewrite = true;
								resolved = true;
								break;
							}
						}
					}
				}
				
			}
			if (!resolved){
				// Still not resolved
				// remove center hit
				if (lastTry) {
					int toRemove = -1;
					int beforeToRemove = -1;
					int lastElement = -1;
					int minDist = 100000;
					// find the element with the shortest distance to another point in z direction
					for (int j = 0; j < length[0]; j++){
						if (data[(j*2)+1]/2 == objects[i]){
							if (lastElement != -1){
								if (beforeToRemove == -1){
									toRemove = j;
									beforeToRemove = lastElement;
									minDist = fabs((float)(data[lastElement*2] - data[j*2]));
								} else {
									int dist = fabs((float)(data[lastElement*2] - data[j*2]));
									if (dist < minDist){
										minDist = dist;
										toRemove = j;
										beforeToRemove = lastElement;
									}
								}
							}
							lastElement = j;
						}
					}
					// if we found the last element, we switch to the element before it.
					if (toRemove == lastElement) toRemove = beforeToRemove;
					data[(toRemove*2)+1] = -1;
					rewrite = true;
				}
			}
		}
	}
	if (rewrite){
		for (int j = 0; j < length[0]; j++){
			if (data[(j*2)+1] == -1){
				removePoint(data, length, j);
				if (j < length[0]) {	 
					j--;
				}
			}
			
		}
	}
}


kernel void drawAppendBufferScreenMonochromaticFinal(global float * screenBuffer, global int* appendBuffer, global int* pixelCounter, global float * mu, global int* priorities, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int data[LOCAL_LIST];
    int x = iGID % width;
	int y = floor(((float)iGID)/width);
	int nextElement = pixelCounter[(y*width)+x];
	int currentValue = 0;
	int lastValue = 0;
	int lastID = -1;
	int i = 0;
	
	// filter entries that appear three times in a row.
	while (nextElement > 0){
		int currentDepth = appendBuffer[(nextElement)*3+1];
		int currentID = appendBuffer[(nextElement)*3+2];
		lastValue = currentValue;
		currentValue = currentDepth;
		nextElement = appendBuffer[(nextElement)*3];
		// filter double entries
		if (fabs((float)(lastValue - currentValue)) > DEPTH_ACCURACY) {
			if (i < (LOCAL_LIST-1)/2) {
				data[i*2] = currentValue;
				data[i*2+1] = currentID;
				i++;
			}
		} else { // only if they describe the same object.
			if (lastID != currentID){
				if (i < (LOCAL_LIST-1)/2) {
					data[i*2] = currentValue;
					data[i*2+1] = currentID;
					i++;			
				}
			}
		}
		lastID = currentID;
	}
	heapSort(data, i);
	int sum = (data[0] / 1000.0f)-500;
	if (i>0){
		threeSubsequentElementsDetection(data, &i);
		inconsistentDataCorrection(data, &i, true);
	}
	float paint = i;
	if (i>=1){
		resolvePriority(data, priorities, i);
		paint = monochromaticAbsorption(data, mu, i-1);
	}
	screenBuffer[(y*width)+x] = paint;	 
} 

kernel void drawAppendBufferScreenMonochromatic(global float * screenBuffer, global int* appendBuffer, global int* pixelCounter, global float * mu, global int* priorities, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    int data[LOCAL_LIST];
    int x = iGID % width;
	int y = floor(((float)iGID)/width);
	int nextElement = pixelCounter[(y*width)+x];
	int currentValue = 0;
	int lastValue = 0;
	int lastID = -1;
	int i = 0;
	
	// filter entries that appear three times in a row
	// and read them from the append buffer.
	int skip = 0;
	while (nextElement > 0){

		int currentDepth = appendBuffer[(nextElement)*3+1];
		int currentID = appendBuffer[(nextElement)*3+2];
		lastValue = currentValue;
		currentValue = currentDepth;
		nextElement = appendBuffer[(nextElement)*3];
		// filter double entries
		if (fabs((float)(lastValue - currentValue)) > DEPTH_ACCURACY) {
		    // This point is far away from the last point.
			if (i < (LOCAL_LIST-1)/2) {
				data[i*2] = currentValue;
				// we multiply by two as we want to mark points that created multiple hits.
				// Such points are likely to be located on a triangle that is almost parallel to the viewing direction.
				data[i*2+1] = currentID * 2 + skip;
				i++;
				skip = 0;
			}
		} else { // only if they describe the same object.
			if (lastID != currentID){
				if (i < (LOCAL_LIST-1)/2) {
					data[i*2] = currentValue;
					// we flag this point with an additional bit.
					data[i*2+1] = (currentID * 2);
					skip = 0;
					i++;			
				}
			} else {
				// schedule the next one for marking
				skip = 0;
				// mark the last one:
				if (data[i*2+1] % 2 != 1){
				  data[i*2+1]+= 0;
				}
			}
		}
		lastID = currentID;
	}
	// sort according to depth value.
	heapSort(data, i);
	int sum = (data[0] / 1000.0f)-500;
	
	if (i>0){
		twoSubsequentElementsDetection(data, &i);
		twoSubsequentElementsDeletion(data, &i);
		threeSubsequentElementsDetection(data, &i);
		inconsistentDataCorrection(data, &i, true);
		twoSubsequentElementsDeletion(data, &i);
		inconsistentDataCorrection(data, &i, true);
	}
	//if (y==190) {
		// if this is set to i we can see the number of intersections per pixel.
		float paint = 0;
		if (i>=1){
			// this will write the intersections array of the current slice to the top of the screen.
			bool paintSortedBuffer = false;
			if (paintSortedBuffer) {
				for (int j = 0; j<i; j++){
					screenBuffer[((j*2)*width+x)] = ((data[2*j])/1000.0f);//priorities[(data[2*j+1])];//
					screenBuffer[((j*2+1)*width+x)] = (data[2*j+1]);
				}
			}
			// this will draw the contents of the appendbuffer of the current slice onto the screen buffer.
			bool paintIntersection = false;
			if (paintIntersection) {
				for (int j = 0; j<i; j++){
					int z = ((data[2*j]/1000.0f)-500);
					screenBuffer[(z*width)+x] = data[2*j+1];
					//screenBuffer[(z*width)+x+1] = j;
				}
			}

			bool drawWithoutPriorities = false;
			if (drawWithoutPriorities){
				int ypaint = sum;
				for (int j = 0; j<i; j++){
					for (int k = ypaint; k < ((data[2*j]/1000.0f))-500; k++) {
						screenBuffer[(k*width)+x] = data[2*j+1];
						//ypaint++;
					}
					ypaint = (data[2*j]/1000.0f)-500;
				}
				//paint = ypaint;
			}
			for (int l=0;l<i;l++){
			  data[l*2+1] /= 2;
			}
			resolvePriority(data, priorities, i);
			bool drawWithPriorities = false;
			if (drawWithPriorities){
				int ypaint = sum;
				paint = ypaint;
				for (int j = 0; j<i-1; j++){
					for (int k = 0; k < floor((data[2*j]/1000.0f)); k++) {
						screenBuffer[(ypaint*width)+x] = mu[data[2*j+1]];
						ypaint++;
					}
				}
				paint = ypaint;
			}
			paint = monochromaticAbsorption(data, mu, i-1);
			
			
		}
		screenBuffer[(y*width)+x] = paint;	 
	//}
	
} 

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
