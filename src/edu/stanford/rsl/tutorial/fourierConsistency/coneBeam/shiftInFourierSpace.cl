kernel void shiftInFourierSpace(global float *real, global float *imag, global float *freqU, global float *freqV, global float *shifts, int const numElementsProj, int const numElementsU, int const numElementsV) {
	int iGIDProj = get_global_id(0);
	int iGIDU = get_global_id(1);
	int iGIDV = get_global_id(2);
	if (iGIDProj >= numElementsProj || iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}

	float angle = freqU[iGIDU]*shifts[2*iGIDProj] + freqV[iGIDV]*shifts[2*iGIDProj + 1];
	float realShift = cos(angle);
	float imagShift = sin(angle);
	float realVal = real[iGIDV*numElementsProj*numElementsU + iGIDU*numElementsProj + iGIDProj];
	float imagVal = imag[iGIDV*numElementsProj*numElementsU + iGIDU*numElementsProj + iGIDProj];
//	float realVal = real[iGIDProj*numElementsU*numElementsV + iGIDU*numElementsV + iGIDV];
//	float imagVal = imag[iGIDProj*numElementsU*numElementsV + iGIDU*numElementsV + iGIDV];
	
	real[iGIDV*numElementsProj*numElementsU + iGIDU*numElementsProj + iGIDProj] = (realShift*realVal) - (imagShift *imagVal);
	imag[iGIDV*numElementsProj*numElementsU + iGIDU*numElementsProj + iGIDProj] = (realShift*imagVal) + (imagShift *realVal);
}
