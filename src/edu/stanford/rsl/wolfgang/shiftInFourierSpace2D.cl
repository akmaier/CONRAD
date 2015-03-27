typedef float2 cfloat;

#define I ((cfloat)(0.0, 1.0))

inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// old version
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

// new version
kernel void shift(global float2 *data, global float *freqU, global float *freqV, global float *shifts, int const numProj, int const numElementsU, int const numElementsV) {
	int iGIDU = get_global_id(0);
	int iGIDV = get_global_id(1);
	if (iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)
	for(int iProj = 0; iProj < numProj; iProj++){
	  // hard coded threshold: if shift vector is zero, if case should never be reached
	  if(fabs(shifts[2*iProj]) >  0.1 || fabs(shifts[2*iProj+1]) > 0.1){
	    float angle = freqU[iGIDU]*shifts[2*iProj] + freqV[iGIDV]*shifts[2*iProj+1];
	    float2 cshift = (float2)(cos(angle), sin(angle));
	    float2 orig = data[iGIDV*numElementsU*numProj + iGIDU*numProj + iProj];
	    data[iGIDV*numElementsU*numProj + iGIDU*numProj + iProj] = cmult(cshift, orig);
	  }
	}
}