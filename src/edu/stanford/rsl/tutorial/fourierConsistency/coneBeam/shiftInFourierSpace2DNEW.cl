
typedef float2 cfloat;

#define I ((cfloat)(0.0, 1.0))

inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// new version
kernel void shift(global float *data, global float *freqU, global float *freqV, global float *shifts, int const numProj, int const numElementsU, int const numElementsV) {

	int iGIDU = get_global_id(0);
	int iGIDV = get_global_id(1);
	if (iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	for(int iProj = 0; iProj < numProj; iProj++){
	    float angle = freqU[iGIDU]*shifts[2*iProj] + freqV[iGIDV]*shifts[2*iProj+1];
	    float2 cshift = (float2)(cos(angle), sin(angle));
	    uint globalOffset = (iGIDV*numElementsU*numProj + iGIDU*numProj + iProj)*2;
	    float2 orig = (float2)(data[globalOffset], data[globalOffset + 1]); //vload2(globalOffset, data);
		//vstore2(cmult(cshift, orig), globalOffset, data);
	    float2 res = cmult(cshift, orig);
	    data[globalOffset] = res.x;
	    data[globalOffset + 1] = res.y;
	}

}