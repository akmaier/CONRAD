/**
 *  yxdraw.cl
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#define LOCAL_SIZE 2000
#define EPSILON 0.0f

private float getIndexX(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[((y*gridWidth)+x)*3];
}

private float getIndexY(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[(((y*gridWidth)+x)*3)+1];
}

private float getIndexZ(global float * coordinateBuffer, int x, int y, int gridWidth){
	return  coordinateBuffer[(((y*gridWidth)+x)*3)+2];
}

private void putYBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, int x, int y, float currentZ, int id){
	// get the next writeable element in the appendBuffer:
	//int appendBufferLocation = appendBufferPointer[0]++;
	// get the last written element from the pixelCoutner and store the new position:
	//int lastPixelPosition = atom_xchg(&pixelCounter[(y*width)+i], appendBufferLocation);
	// write into the localBuffer:
	//  - xy-location*300+ currentID
	//  - current Z value
	int appendBufferLocation = atom_inc(&appendBufferPointer[0]);
	if (appendBufferLocation < LOCAL_SIZE-1) {	
		appendBuffer[(appendBufferLocation)*2] = (((y*width)+x)*300)+id;
		appendBuffer[(appendBufferLocation)*2+1] = -currentZ*1000.0f;;
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
private void drawSpansYBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, float8 edge1, float8 edge2, int id){
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
		
		putYBufferLocal(appendBuffer, appendBufferPointer, width,
			edge1.s0 + (int)(e1xdiff * factor1),
			y, 
			edge1.s2 + (e1colordiff * factor1),
			id);
			
		putYBufferLocal(appendBuffer, appendBufferPointer, width,
			edge2.s0 + (int)(e2xdiff * factor2),
			y,
			edge2.s2 + (e2colordiff * factor2),
			id);
			

		// increase factors
		factor1 += factorStep1;
		factor2 += factorStep2;
	}
}

private void drawTriangleYBufferLocal(local int * appendBuffer, local int * appendBufferPointer, int width, int x1, 
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
		drawSpansYBufferLocal(appendBuffer, appendBufferPointer, width, edge1, edge2, id);
	}
	float e31 = fabs(((edge3.s1 * m)+t) - edge3.s0);
    float e32 = fabs(((edge3.s5 * m)+t) - edge3.s4);
    if ((e31 >EPSILON)||(e32 > EPSILON)){
		drawSpansYBufferLocal(appendBuffer, appendBufferPointer, width, edge1, edge3, id);
	} 
}

kernel void drawTrianglesYBufferLocal(global float* coordinateBuffer, global int* yBuffer, global int* yBufferPointer, global int* pixelCounter, int width, int id, int numElements){
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
			drawTriangleYBufferLocal(localBuffer, localBufferPointer, width,  
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
	
	drawTriangleYBufferLocal(localBuffer, localBufferPointer, width, 
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
	drawTriangleYBufferLocal(localBuffer, localBufferPointer, width, 
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
			yBufferPointer[0] = INFINITY;
		} else {
    		int globalLocation = atom_add(&yBufferPointer[0], localBufferPointer[0]);
    		for (int i = 0; i < localBufferPointer[0]; i++){
    			// This is the current value we would like to store in the buffer
    			int xy = localBuffer[i*2]/300;
    			x = xy%width;
    			y = xy/width;
    			int id = localBuffer[i*2]%300; 
				int currentZ = localBuffer[i*2+1];

				// get the next writeable element in the appendBuffer:
				int yBufferLocation = globalLocation+i;
				// get the last written element from the pixelCoutner and store the new position:
				int lastPixelPosition = atom_xchg(&pixelCounter[y], yBufferLocation);
				// write into the appendBuffer:
				//  - pointer to next element
				//  - current Z value
				//  - current ID
				yBuffer[(yBufferLocation)*3] = lastPixelPosition;
				yBuffer[(yBufferLocation)*3+1] = currentZ;
				yBuffer[(yBufferLocation)*3+2] = (x*300)+id;
    		}
    	}
    	
    }
	
}


/**
 * draws the maximum depth value onto the screen. This is similar to a z-Buffer.
 */
kernel void drawYBufferScreen(global float * screenBuffer, global int* yBuffer, global int* pixelCounter, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    //int x = iGID % width;
	int y = iGID;
	int nextElement = pixelCounter[y];
	if (y==160){
		while (nextElement > 0){
			float currentZ = (float)(yBuffer[(nextElement)*3+1])/1000.0f;
			int currentID = (yBuffer[(nextElement)*3+2]); 
			int x = currentID / 300;
			currentID = currentID % 300;
			nextElement = yBuffer[(nextElement)*3];
			screenBuffer[((int)(currentZ-500)*width)+x] = currentID; 
		}
	}
}

private void putXBuffer(global int * xBuffer, global int * xPixelCount, global int * xBufferPointer, int width, int x, int y, int currentZ, int id){
	// get the next writeable element in the appendBuffer:
	// get the last written element from the pixelCoutner and store the new position:
	//  - xy-location*300+ currentID
	//  - current Z value
	int bufferLoc = atom_inc(xBufferPointer);
	int lastElement = xPixelCount[(y*width)+x];
	xPixelCount[(y*width)+x] = bufferLoc;
	xBuffer[(bufferLoc)*3] = lastElement;
	xBuffer[(bufferLoc)*3+1] = (x*300)+id;
	xBuffer[(bufferLoc)*3+2] = currentZ;
}

/**
 * draws a horizontal line into the appendBuffer
 * we convert the depth of XCat into units of microns by multiplication with 1000
 * int32 ranges from -2 billion to 2 billion (-2 to the power 31 to 2 to the power 31). Hence, we can
 * compute depth values from -2,147 to 2,147 meters. This should be sufficiently accurate for most
 * medical imaging tasks.
 */
private void drawSpanXBuffer(global int * xBuffer, global int* xPixelCount, global int *xBufferPointer, int width, int x1, int z1, int x2, int z2, int y, int id){
	if (x1 == x2){
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
	if ((x2-x1) > width) return;
	for (int i = x1; i <=x2; i++){
		// This is the current value we would like to store in the buffer 
		int currentZ = (int)((z1 - (((float)(i - x1)) * stepInc)));
		putXBuffer(xBuffer, xPixelCount, xBufferPointer, width, i, y, currentZ, id);
	}
}


/**
 * draws the maximum depth value onto the screen. This is similar to a z-Buffer.
 */
kernel void drawYBufferXBuffer(global int* yBuffer, global int* pixelCounter, global int* xBuffer, global int* xBufferPointer, global int* xPixelCount, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    //int x = iGID % width;
    int y = iGID;
    int previousX = -1;
    int previousZ = -1.0f;
    int previousID = -1;
    int lastDrawnX = -1;
    int lastDrawnZ = -1;
	int nextElement = pixelCounter[y];
	while (nextElement > 0){
		int currentZ = (yBuffer[(nextElement)*3+1]);	
		int currentID = (yBuffer[(nextElement)*3+2]); 
		int x = currentID / 300;
		currentID = currentID % 300;
		if (previousID != currentID){
			previousX = -1;
			lastDrawnX = -1;
		}
		if (previousX > 0) {
			if (lastDrawnX == -1){
				drawSpanXBuffer(xBuffer, xPixelCount, xBufferPointer, width, previousX, previousZ, x, currentZ, y, currentID);
				lastDrawnX = x;
				lastDrawnZ = currentZ;
			} else {
				if ((previousX - x > 2)||(x - previousX > 2)){
					drawSpanXBuffer(xBuffer, xPixelCount, xBufferPointer, width, previousX, previousZ, x, currentZ, y, currentID);
					lastDrawnX = x;
					lastDrawnZ = currentZ;
				} else {
					if (previousX != x){
						if (((lastDrawnX-previousX)*(previousX-x))>0){
							drawSpanXBuffer(xBuffer, xPixelCount, xBufferPointer, width, lastDrawnX, lastDrawnZ, x, currentZ, y, currentID);
							lastDrawnX = x;
							lastDrawnZ = currentZ;
						} else {
							drawSpanXBuffer(xBuffer, xPixelCount, xBufferPointer, width, previousX, previousZ, x, currentZ, y, currentID);
							lastDrawnX = x;
							lastDrawnZ = currentZ;
						
						}
					}
				}
			}
		}
		previousX = x;
		previousZ = currentZ;
		previousID = currentID;
		nextElement = yBuffer[(nextElement)*3]; 
	}
}



/**
 * draws the maximum depth value onto the screen. This is similar to a z-Buffer.
 */
kernel void drawXBufferScreen(global float * screenBuffer, global int* xBuffer, global int* xPixelCount, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    //int x = iGID % width;
    int y = iGID;
    for (int x = 0; x < width; x++) {
   		int nextElement = xPixelCount[y*width+x];
  		if (y == 160) {
			while (nextElement > y*width*200){
				float currentZ = (float)(xBuffer[(nextElement)*3+2])/1000.0f;	
				int currentID = xBuffer[(nextElement)*3+1]; 
				int x = currentID / 300;
				currentID = currentID % 300;
				nextElement = xBuffer[(nextElement)*3];
				screenBuffer[(int)(currentZ-500.0f)*width + x] = currentID; 
			}
		
		}
	}
}

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/