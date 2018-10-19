kernel void addValuePI(global float* a, int numElements) {

		// Get index into global data array
        int iGID = get_global_id(0);
        
		
		// Bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements) {
        	return;
        }
        

		// Add value of PI
        a[iGID] = a[iGID] + 3.14f;
}