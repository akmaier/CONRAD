kernel void shiftInFourierSpace2D(global float *real, global float *imag, global float *freqU, global float *freqV, float shiftX, float shiftY, int const numElementsU, int const numElementsV) {
	int iGIDU = get_global_id(0);
	int iGIDV = get_global_id(1);
	if (iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}

	float angle = freqU[iGIDU]*shiftX + freqV[iGIDV]*shiftY;
	float realShift = cos(angle);
	float imagShift = sin(angle);
	float realVal = real[iGIDV*numElementsU + iGIDU];
	float imagVal = imag[iGIDV*numElementsU + iGIDU];
//	float realVal = real[iGIDProj*numElementsU*numElementsV + iGIDU*numElementsV + iGIDV];
//	float imagVal = imag[iGIDProj*numElementsU*numElementsV + iGIDU*numElementsV + iGIDV];
	
	real[iGIDV*numElementsU + iGIDU] = (realShift*realVal) - (imagShift *imagVal);
	imag[iGIDV*numElementsU + iGIDU] = (realShift*imagVal) + (imagShift *realVal);
}