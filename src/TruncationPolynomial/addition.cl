// first step of sum
kernel void add(global float *grid1, global float *grid2, int const numElements) {

	int iGID = get_global_id(0);
	
	if (iGID >= numElements) {
		return;
	}
	
	grid1[iGID] += grid2[iGID];

}