/**
 *  appendBuffer.cl
 */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_intel_printf : enable


// Local memory size (Tesla C1060): 16 KB = 16384 B
// Sizeof(int) = 4
// 2 values per element (xy * id, currentZ)
// finally, subtract a constant to reserve some mem for register-swapping 
// Hence, max size is: ~ (16384 / 4 / 2) - 100
// Note that this limit mainly concerns the size of the triangles and the local workgroup size.
// The smaller the triangles we paint, the smaller the local size can be.
#define DEPTH_ACCURACY 0
#define DEPTH_RESOLUTION 0
#define DEPTHSAMPLING 1000.0f
#define LOCAL_LIST 2000

#define EPSILON 0.0f
#define NORMALEPSILON 0.01f
#define FILLEPSILON 0.001f
#define TRIANGLEEPSILON 0.0001f

// u,v in range is must be checked by caller (done in drawTrianglesAppendBufferGlobal)
private float getIndexX(global float * coordinateBuffer, int u, int v, int gridWidth){
	return  coordinateBuffer[((v*gridWidth)+u)*3];
}

private float getIndexY(global float * coordinateBuffer, int u, int v, int gridWidth){
	return  coordinateBuffer[(((v*gridWidth)+u)*3)+1];
}

private float getIndexZ(global float * coordinateBuffer, int u, int v, int gridWidth){
	return  coordinateBuffer[(((v*gridWidth)+u)*3)+2];
}

private void putAppendBufferGlobal(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, int height, int x, int y, float currentZ, int id, int flag){
	//printf("AppBuffIn: %d, %d, %d\n", width, x, y);
	if (x >= width || y >= height || x < 0 || y < 0) { // check high & low limits
		return;
	}
	// TODO: Check for appendBuffer overflow
	// get the next writeable element in the appendBuffer:	
	int appendBufferLocation = atom_inc(&appendBufferPointer[0]);
	// get the last written element from the pixelCounter and store the new position:
	int lastPixelPosition = atom_xchg(&pixelCounter[(y*width)+x], appendBufferLocation);
	flag += 1; // convert sign function (-1,0,1) to positive range (0,1,2)
	// write into the appendBuffer:
	appendBuffer[(appendBufferLocation)*3]   = lastPixelPosition; 			// pointer to next element
	appendBuffer[(appendBufferLocation)*3+1] = -currentZ* DEPTHSAMPLING; 	// current Z value
	appendBuffer[(appendBufferLocation)*3+2] = id*4 + flag;					// current ID	
}

private bool isTrianglePart(float2 v0, float2 v1, float2 p, float x, float y){
	float2 v2 = (float2){x-p.x,y-p.y};
	
	float dot00  = dot(v0, v0);
	float dot01  = dot(v0, v1);
	float dot02  = dot(v0, v2);
	float dot11  = dot(v1, v1);
	float dot12  = dot(v1, v2);
	
	// Compute barycentric coordinates
	float invDenom = 1.0 / ((dot00 * dot11) - (dot01 * dot01));
	float u = ((dot11 * dot02) - (dot01 * dot12)) * invDenom;
	float v = ((dot00 * dot12) - (dot01 * dot02)) * invDenom;
	return (u >= -TRIANGLEEPSILON) && (v >= -TRIANGLEEPSILON) && (u + v <= 1+TRIANGLEEPSILON);
}

private float computeDepth(float2 v0, float2 v1, float2 p, float x, float y, float z1, float z2, float z3){
	float2 v2 = (float2){x-p.x,y-p.y};
	
	float dot00  = dot(v0, v0);
	float dot01  = dot(v0, v1);
	float dot02  = dot(v0, v2);
	float dot11  = dot(v1, v1);
	float dot12  = dot(v1, v2);
	
	// Compute barycentric coordinates
	float invDenom = 1.0 / ((dot00 * dot11) - (dot01 * dot01));
	float u = ((dot11 * dot02) - (dot01 * dot12)) * invDenom;
	float v = ((dot00 * dot12) - (dot01 * dot02)) * invDenom;
	float w = 1-u-v;
	// depth computation using barycentric interpolation
	float depth = u*z2+v*z3+w*z1;

	/*if( isnan(u) || isnan(v) || isnan(w) || depth > 0) {
		// nan values occur if 2 points are exactly the same
		printf("v0: %f, %f\n", v0.x, v0.y);
		printf("v1: %f, %f\n", v1.x, v1.y);
		printf("v2: %f, %f\n", v2.x, v2.y);
		printf("barycentric coordinates: %f, %f, %f\n", u, v, w);
		printf("depth values: %f, %f, %f\n", z1, z2, z3);
		printf("interpolated depth: %f\n", depth);
	}*/
	return depth;
}


private void drawTriangleAppendBufferSimple(global int * appendBuffer, global int * appendBufferPointer, global int * pixelCounter, int width, int height, float x1, 
        float y1, float z1, int idu1, int idv1, float x2, float y2, float z2, int idu2, int idv2, float x3, float y3, float z3, int idu3, int idv3, int id){
	//float8 edge1 = (float8){x1, y1, z1, idu1, x2, y2, z2, idu2};
	//float8 edge2 = (float8){x2, y2, z2, idu2, x3, y3, z3, idu3};
	//float8 edge3 = (float8){x1, y1, z1, idu1, x3, y3, z3, idu3};
	float ymin = fmax(fmin(fmin(y1,y2),y3),0);
	float ymax = fmin(fmax(fmax(y1,y2),y3),height);
	float xmin = fmax(fmin(fmin(x1,x2),x3),0);
	float xmax = fmin(fmax(fmax(x1,x2),x3),width);
	
/*	float A = (z2-z3)/(x2-x3); // numerically unstable: differences of similar numbers, division 
	float B = (z1-z2)/(x1-x2);
	float C = (y2-y3)/(x2-x3);
	float D = (y1-y2)/(x1-x2);
	
	float n_2 = (A - B) / (C - D);
	if (isnan(n_2)) n_2 = 0;
	float n_1 = B - (n_2*D);
	float n_0 = z2 - (x2 * n_1) - (y2 * n_2);*/
	
	float2 v0 = (float2){x2-x1,y2-y1};
	float2 v1 = (float2){x3-x1,y3-y1};
	float2 p = (float2){x1,y1};
	
	for (int j=round(ymin) ; j <= round(ymax); j++){
		for (int i=round(xmin); i<= round(xmax);i++){
			if (isTrianglePart(v0,v1,p,i,j)){
				float currentZ = computeDepth(v0,v1,p,i,j,z1,z2,z3);
				if (currentZ < 0) { // TODO: make independent of camera orientation
					// only put intersection points in front of the camera
					putAppendBufferGlobal(appendBuffer, appendBufferPointer, pixelCounter, width, height, i, j, currentZ, id, idu1);
				} /*else { // currentZ >= 0
					//printf("CurrentZ is %f\n", currentZ);
					//printf("x: %f, %f, %f\n", x1, x2, x3);
					//printf("y: %f, %f, %f\n", y1, y2, y3);
					//printf("z: %f, %f, %f\n", z1, z2, z3);
					//printf("u,v: %d, %d\n", i, j);
				}*/
			}
		}
	}
}

kernel void drawTrianglesAppendBufferGlobal(global float* coordinateBuffer, global int* appendBuffer, global int* appendBufferPointer, global int* pixelCounter, int width, int height, int id, int elementCountU, int elementCountV, int normalsign) {
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= (elementCountU * elementCountV)) {
    	return;
    }
    //printf("Starting iGID: %d\n", iGID);
    // size(coordinateBuffer) == elementCountU * elementCountV (*3) 
	int u = iGID % elementCountU;
	int v = floor(((float) iGID)/elementCountU);
	int u2 = (u + 1) % elementCountU;
	int v2 = (v + 1) % elementCountV;

	if ((u == 0 || u2 == 0)){ // opening in the surface possible
		int ucoord = 0;
		if (u2 == 0) ucoord = u;
		float4 one = (float4){getIndexX(coordinateBuffer, ucoord, v, elementCountU),
			getIndexY(coordinateBuffer, ucoord, v, elementCountU),			
			getIndexZ(coordinateBuffer, ucoord, v, elementCountU), 0.0f};
		float4 two = (float4){getIndexX(coordinateBuffer, ucoord, v2, elementCountU),
			getIndexY(coordinateBuffer, ucoord, v2, elementCountU),
			getIndexZ(coordinateBuffer, ucoord, v2, elementCountU), 0.0f};
		if (distance(one, two) > FILLEPSILON){ // we need to fill the hole
			// hole is filled by triangles between border points and hole center
			float4 center = (float4){0.0f, 0.0f, 0.0f, 0.0f};
			int closingPoints = 8;
			float factor = ((float) elementCountV) / closingPoints;
			for (int i=0; i < closingPoints; i++){
				center += (float4){getIndexX(coordinateBuffer, ucoord, i*factor, elementCountU),
				getIndexY(coordinateBuffer, ucoord, i*factor, elementCountU),
				getIndexZ(coordinateBuffer, ucoord, i*factor, elementCountU), 0.0f};
			}
			center /= closingPoints;
			float4 normal = cross (two-one, center-one);
			int dir = (int)sign(normal.z);
			if (fabs(normal.z)<NORMALEPSILON) dir = 0;
			if (!(u==0)) dir *= -1; // Normal direction depends on the side of the opening, always points outwards of the surface 
			dir *= normalsign; // adapt to spline definition
			drawTriangleAppendBufferSimple(appendBuffer, appendBufferPointer, pixelCounter, width, height, 
			one.x, one.y, one.z, dir, dir,
			two.x, two.y, two.z, dir, dir,
			center.x, center.y, center.z, dir, dir, id);
			//printf("one: %f, %f, %f\n", one.x, one.y, one.z);
			//printf("two: %f, %f, %f\n", two.x, two.y, two.z);
			//printf("center: %f, %f, %f\n", center.x, center.y, center.z);
		}
		if (u2 == 0) return; // don't draw triangles over the coordinate wrap
	}
	
	float4 one = (float4){ getIndexX(coordinateBuffer, u2, v2, elementCountU), 
	getIndexY(coordinateBuffer, u2, v2, elementCountU), 
	getIndexZ(coordinateBuffer, u2, v2, elementCountU), 0.0f};
	float4 two = (float4){ getIndexX(coordinateBuffer, u, v, elementCountU),
	getIndexY(coordinateBuffer, u, v, elementCountU),
	getIndexZ(coordinateBuffer, u, v, elementCountU), 0.0f};
	float4 three = (float4){ getIndexX(coordinateBuffer, u2, v, elementCountU),
	getIndexY(coordinateBuffer, u2, v, elementCountU),
	getIndexZ(coordinateBuffer, u2, v, elementCountU), 0.0f};
	float4 normal = cross (two-one, three-one);
	int dir = (int)sign(normal.z);
	if (fabs(normal.z)<NORMALEPSILON) dir = 0;
	dir*=normalsign;
	// Draw rectangular patch of surface as two triangles
	drawTriangleAppendBufferSimple(appendBuffer, appendBufferPointer, pixelCounter, width, height,
		one.x, one.y, one.z, dir, dir,
		two.x, two.y, two.z, dir, dir,
		three.x, three.y, three.z, dir, dir, id);
	drawTriangleAppendBufferSimple(appendBuffer, appendBufferPointer, pixelCounter, width, height,
			two.x, two.y, two.z, dir, dir,
			getIndexX(coordinateBuffer, u, v2, elementCountU),
			getIndexY(coordinateBuffer, u, v2, elementCountU),
			getIndexZ(coordinateBuffer, u, v2, elementCountU),
			dir,
			dir,
			one.x, one.y, one.z, dir, dir, id);
}


/**
 * draws the maximum depth value onto the screen. This is similar to a z-Buffer.
 */
kernel void drawAppendBufferScreen(global float* screenBuffer, global int* appendBuffer, global int* pixelCounter, int width, int numElements){
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
		float currentElement = (float)(appendBuffer[(nextElement)*3+1]) / DEPTHSAMPLING;
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
		if ((a[swap2*2] < a[child*2]) ||
				 (a[swap2*2] == a[child*2] && a[swap2*2+1] < a[child*2+1])){ // also sort w.r.t. ID
             swap2 = child;
		}
        //(check if right child exists, and if it's bigger than what we're currently swapping with)
		if (child < end && ((a[swap2*2] < a[(child+1)*2]) ||
		      (a[swap2*2] == a[(child+1)*2] && a[swap2*2+1] < a[(child+1)*2+1]))){
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

int getHighestPriorityMaterialFromStack(int * materialStack, int stackPointer, global int* priorities){
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
	return ID;
}

bool contains(int * materialStack, int stackPointer, int ID){
	for (int i = 0; i <= stackPointer; i++){
		if (materialStack[i] == ID){
			return true;
		}
	}	
	return false;
}

int removeAll(int* materialStack, int stackPointer, int ID){
	int del = 0;
	for (int i = 0; i <= stackPointer; i++){
	    // same object
		if (materialStack[i] == ID){
			if (i < stackPointer) {
				for (int k = i; k <stackPointer; k++){
					materialStack[k] = materialStack[k+1];
				}
				i--;
			}
			del++;
		}
	}
	return del;
}

int removeOnce(int* materialStack, int stackPointer, int ID){
	for (int i = 0; i <= stackPointer; i++){
	    // same object
		if (materialStack[i] == ID){
			if (i < stackPointer){
				for (int k = i; k < stackPointer; k++){
					materialStack[k] = materialStack[k+1];
				}
				i--;
			}
			return 1;
		}
	}
	return 0;
}

void resolvePriority(int* rayList, global int* priorities, int length){
	// Assumption: all the elements in the raylist are correct
	// -for each object entrance, there is one exit & inversely
	// -objects are entered first (normal == 0) and left afterwards (normal == 2)
	int materialStack[100];
	int stackPointer = 0;
	materialStack[0] = rayList[1] / 4;
	for (int k=1; k < length;k++){
		int segment = rayList[k*2] - rayList[(k-1)*2]; // convert depth to length values
		int ID = getHighestPriorityMaterialFromStack(materialStack, stackPointer, priorities);
		rayList[(k-1)*2] = segment;
		rayList[(k-1)*2+1] = ID;
		int nextObjectID = rayList[(k*2)+1] / 4;
		int normal = rayList[(k*2)+1] % 4;
		// Normal direction of triangle determines whether the object is entered or left
		if (normal == 0) {
			// Remove from stack
			stackPointer -= removeOnce(materialStack, stackPointer, nextObjectID);
		} else if (normal == 2) {
			// append to Stack
			stackPointer++;
			materialStack[stackPointer] = nextObjectID;
		}
	}
	// Delete last element (should not be drawn anyways)
	rayList[(length-1)*2] = 0;
	rayList[(length-1)*2+1] = 0;
}

/*void resolvePriorityOld(int* rayList, global int* priorities, int length){
	int materialStack [100];
	int stackPointer=0;
	int current=0;
	while (current < length && (rayList[current*2+1] % 4 ==1)){
		current++;
	}
	if (current == length-1) return;
	materialStack[0] = rayList[current*2+1]/4;
	for (int k=current+1; k < length;k++){
		int segment = rayList[k*2] - rayList[(k-1)*2];
		int ID = getHighestPriorityMaterialFromStack(materialStack, stackPointer, priorities);
		rayList[(k-1)*2] = segment;
		rayList[(k-1)*2+1] = ID;
		int nextObjectID = rayList[(k*2)+1] /4;
		int normal = rayList[(k*2)+1] % 4;
		int del = removeAll(materialStack, stackPointer, nextObjectID);
		stackPointer -= del;
		// Append to Stack
		if (del == 0){
			stackPointer++;
			materialStack[stackPointer] = nextObjectID;
		}		
	}
}*/
 
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
	float t = sum/(DEPTHSAMPLING*10.f);
	//if(t < 0.0001f){
	//	return 0.0f;
	//}
	return t;
}

void removePoint(int* data, int* i, int location){
	for (int k = location; k < i[0]-1; k++){
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
		if ((lastID == currentID)&&(abs(data[(l)*2] - data[(l-1)*2]) <= DEPTH_RESOLUTION)){
			if (data[(l)*2+1] % 2 == 0) { // % 2 == 0 ensures that each entry is only marked once
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
void twoSubsequentElementsDeletion(int* data, int* i){
	int lastID = data[1];
	// parse the sorted array
	for (int l = 1; l < i[0]; l++){
		int currentID = data[(l*2)+1];
		// two times the same ID in a row
		if ((lastID %2 == 1) && (lastID == currentID)){
	    	data[(l)*2+1] = -1;
	    	data[(l)*2] = -1;
			//// remove inconsistency candidate flag
			//data[(l-1)*2+1] -= 1; // test without removal -> 3 or more subsequent elements are also deleted
			removePoint(data, i, l);
			if (l < i[0]) {	 
				l--;
			} 
		}
		lastID = data[(l)*2+1];
	}
}

/**
 * Removes inconsistent points from the data. 
 * Each object must be entered and left on the ray equally often.
 * Each object must be entered before it can be left.
 * 
 * @param data the array of intersection points
 * @param i the length of the array. This is changed, if points are removed.
 */
void inconsistentDataCorrection(int* data, int* length){
	int objects[20];
	int objectCount = -1;

	// create a list of objects on the ray.
	for (int i = 0; i < length[0]; i++) {
		int p = data[(i*2)+1]/8;
		bool found = false;
		for (int j=0; j <= objectCount; j++) {
			if (objects[j] == p) {
				found = true;
			}
		}
		if (!found) {
			objectCount++;
			objects[objectCount] = p;
		}
	}
	// resolve intersections along the ray.
	bool rewrite = false;
	for (int objN = 0; objN <= objectCount; objN++){
		int entryCount = 0;
		int undecidedCount = 0;
		int entries[20];
		int undecided[20];
		int leaves[20];
		for (int i=0; i<20;i++){
			entries[i]=-1;
			undecided[i]=-1;
			leaves[i]=-1;
		}
		for (int j = 0; j < length[0]; j++){
			if (data[(j*2)+1]/8 == objects[objN]){
				int normal = (data[(j*2)+1] % 8) / 2;
				if (normal == 2) {
					// object entrance
					entries[entryCount] = j;
					entryCount++;
				} else if (normal == 1) {
					// parallel triangle -> can be used as both
					undecided[undecidedCount] = j;
					undecidedCount++;
				} else if (normal == 0) {
					// object exit
					if (entryCount == 0 && undecidedCount == 0) {
						// there where more exits than entrances -> invalid element
						data[(j*2)+1] = -1;
						rewrite = true;
					} else if (entryCount == 0 && undecidedCount > 0) {
						// undecided element is used as entry
						undecidedCount--;
						data[(undecided[undecidedCount]*2)+1]++; // make element from undecided to entry
						undecided[undecidedCount] = -1;
					} else if (entryCount > 0) {
						// remove entry. Implicit assumption: exit corresponds to last entry
						entryCount--;
						entries[entryCount] = -1;
					}
				}
			}
		}
		
		if (entryCount == 0){
			// If objects are entered & left equally often, everything is fine
			// undecided intersections can be removed
			for(int i=0; i < undecidedCount; i++){
				data[(undecided[i]*2)+1] = -1;
			}
			rewrite = true;
		} else if (entryCount > 0){
			if (undecidedCount == 0) {
				// objects is entered more often than it is left, remove all the objects
				for(int i=0; i < entryCount; i++){
					data[(entries[i]*2)+1] = -1;
				}
				rewrite = true;
			} else {
				// undecided can be used as exits if order is correct
				for(int i=0; i < entryCount; i++){
					for(int j=0; j < undecidedCount; j++){
						if ((entries[i] < undecided[j]) && (undecided[j] != -1)){
							data[(undecided[j]*2)+1] -= 2; // make element from undecided to exit
							undecided[j] = -1; 	// remove this undecided element for other entries
							entries[i] = -1; 	// this entry has found a fitting exit
							j = undecidedCount; // search only 1 exit for the entry. CAREFUL: do not use break; killed nvidia opencl compiler
						}
					}
				}
				// remove objects that have not found a counterpart
				for(int i=0; i < entryCount; i++){
					if (entries[i] != -1){
						data[(entries[i]*2)+1] = -1;
					}
				}
				for(int i=0; i < undecidedCount; i++){
					if (undecided[i] != -1){
						data[(undecided[i]*2)+1] = -1;
					}
				}
				rewrite = true;
			}
		} /*else {
			// entryCount<0 cannot occur, avoided in normal==0
		}*/
	} // end for all objects
	
	// Delete elements if any where marked for deletion
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

// Function to get surface information by computing the smallest depth.
kernel void getSurfaceInformationFromAppendBuffer(global float* surfaceBuffer, global int* appendBuffer, global int* pixelCounter, global float* mu, global int* priorities, int width, int numElements){
	int iGID = get_global_id(0);
	if(iGID >= numElements){
    	return;
    }
    int dataMin = 2147483647;
    int x = iGID % width;
	int y = floor(((float) iGID) / width);
	int nextElement = pixelCounter[(y * width) + x];
	int dataLength = 0;
	int currentDepth = 0;
	while((nextElement > 0) && (dataLength < (LOCAL_LIST - 1) / 2)){
		currentDepth = appendBuffer[(nextElement * 3) + 1];
		nextElement = appendBuffer[nextElement * 3];
		if(dataMin > currentDepth){
			dataMin = currentDepth;
		}
		++dataLength;
	}
	if((dataLength > 0) && (dataLength < (LOCAL_LIST - 1) / 2)){
		surfaceBuffer[(y * width) + x] = dataMin;
	}
	return;
}

kernel void drawAppendBufferScreenMonochromatic(global float * screenBuffer, global int* appendBuffer, global int* pixelCounter, global float * mu, global int* priorities, int width, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	//printf("iGID too big, aborting!");
    	return;
    }
    int data[LOCAL_LIST];
    int x = iGID % width;
	int y = floor(((float)iGID)/width);
	int nextElement = pixelCounter[(y*width)+x];
	int dataLength = 0;
	
	while (nextElement > 0){
		int currentDepth = appendBuffer[(nextElement)*3+1];
		int currentID = appendBuffer[(nextElement)*3+2];
		nextElement = appendBuffer[(nextElement)*3];
		if (dataLength < (LOCAL_LIST-1)/2) {
			data[dataLength*2] = currentDepth;
			// we multiply by two as we want to mark points that created multiple hits.
			data[dataLength*2+1] = currentID * 2;
			dataLength++;
		} else {
			//printf("LOCAL_LIST is too short, aborting!");
			return;
		}
	}
	
	// sort according to depth value.
	heapSort(data, dataLength);
	
	if (dataLength>0){
		twoSubsequentElementsDetection(data, &dataLength);
		twoSubsequentElementsDeletion(data, &dataLength);
		inconsistentDataCorrection(data, &dataLength);	
	}

	for (int l=0;l<dataLength;l++){
		// Remove Flags from IDs
		data[l*2+1] /= 2;
	}
	//if (y==270) {
		// if this is set to i we can see the number of intersections per pixel.
		float paint = 0;
		int paint_sum = (data[0] / 1000.0f)-500; // debug only
		if (dataLength>=1){
			// this will write the intersections array of the current slice to the top of the screen.
			bool paintSortedBuffer = false;
			if (paintSortedBuffer) {
				for (int j = 0; j<dataLength; j++) {
					screenBuffer[((j*2)*width+x)] = ((data[2*j])/DEPTHSAMPLING); //priorities[(data[2*j+1])];//
					screenBuffer[((j*2+1)*width+x)] = (data[2*j+1]);
				}
			}
			// this will draw the contents of the appendbuffer of the current slice onto the screen buffer.
			bool paintIntersection = false;
			if (paintIntersection) {
				for (int j = 0; j<dataLength; j++) {
					float zoom = 1;
					int z = (((((float)data[2*j])/DEPTHSAMPLING)*zoom)-(500*zoom)/*-250*/);
					if ((z>=0) && (z<480))
					screenBuffer[(z*width)+x] = data[2*j+1];
					//screenBuffer[(z*width)+x+1] = j;
				}
			}

			bool drawWithoutPriorities = false;
			if (drawWithoutPriorities){
				int ypaint = paint_sum;
				for (int j = 0; j<dataLength; j++) {
					for (int k = ypaint; k < ((data[2*j]/DEPTHSAMPLING))-500; k++) {
						screenBuffer[(k*width)+x] = data[2*j+1];
						//ypaint++;
					}
					ypaint = (data[2*j]/DEPTHSAMPLING)-500;
				}
			}

			resolvePriority(data, priorities, dataLength);
			
			// this will write the intersections array of the current slice to the top of the screen.
			bool paintPriorityResolvedBuffer = false;
			if (paintPriorityResolvedBuffer) {
				for (int j = 0; j<dataLength; j++){
					screenBuffer[((j*2)*width+x)] = ((data[2*j])/DEPTHSAMPLING);
					screenBuffer[((j*2+1)*width+x)] = (data[2*j+1]);
				}
			}
			
			bool drawWithPriorities = false;
			if (drawWithPriorities) {
				int ypaint = paint_sum;
				for (int j = 0; j<dataLength-1; j++){
					for (int k = 0; k < floor((data[2*j]/DEPTHSAMPLING)); k++) {
						screenBuffer[(ypaint*width)+x] = mu[data[2*j+1]];
						ypaint++;
					}
				}
				paint = ypaint;
			}

			paint = monochromaticAbsorption(data, mu, dataLength-1);
		}
		screenBuffer[(y*width)+x] = paint;
	//}
	return;
}

/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
