
typedef float2 cfloat;

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define I ((cfloat)(0.0, 1.0))

#define cmult(a,b) ((cfloat)(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))

#define cmult4(a,b) ((float8)(cmult((a).s01,(b).s01), cmult((a).s23,(b).s23), cmult((a).s45,(b).s45), cmult((a).s67,(b).s67)))

// new version
/*
kernel void shift(global float *data, global float *freqU, global float *freqV, global float *shifts, int const numProj, int const numElementsU, int const numElementsV) {

	int iGIDU = get_global_id(0);
	int iGIDV = get_global_id(1);
	if (iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	uint globalOffsetMain = mul24(mad24(iGIDV,numElementsU,iGIDU),numProj);
	for(int iProj = 0; iProj < numProj; iProj++){
		int iProjTimes2 = iProj << 1; 
	    float angle = freqU[iGIDU]*shifts[iProjTimes2] + freqV[iGIDV]*shifts[iProjTimes2+1];
	    float2 cshift = (float2)(cos(angle), sin(angle));
		uint globalOffset = (globalOffsetMain + iProj);
	    float2 orig = vload2(globalOffset, data);
		float2 res = cmult(cshift, orig);
		vstore2(res, globalOffset, data);
	    //data[globalOffset] = res.x;
	    //data[globalOffset + 1] = res.y;
	}

}



// new version
kernel void shift(
	global float *data, 
	global float *freqU, 
	global float *freqV, 
	global float *shifts, 
	int const numProj, 
	int const numElementsU, 
	int const numElementsV) 
{

	int iProj = get_global_id(0);
	int iGIDU = get_global_id(1);
	int iGIDV = get_global_id(2);
	if (iProj >= numProj || iGIDU >= numElementsU || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	float2 shiftUV = vload2(iProj, shifts);
	float2 freqUV = (float2)(freqU[iGIDU],freqV[iGIDV]);
	
	float angle = dot(shiftUV,freqUV);
	float2 cshift = (float2)(cos(angle), sin(angle));
	
	uint globalOffset = mad24(mad24(iGIDV, numElementsU, iGIDU),numProj,iProj);
	float2 orig = vload2(globalOffset, data);
	float2 res = cmult(cshift, orig);
	vstore2(res, globalOffset, data);
}
*/

/*
// new version
kernel void shift(
	global float *data, 
	global float *shifts, 
	int const numProj, 
	int const numElementsU, 
	int const numElementsV,
	const float deltaU,
	const float deltaV) 
{

	int iProj = get_global_id(0);
	int iGIDU = get_global_id(1);
	if (iProj >= numProj || iGIDU >= numElementsU) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	uint iGIDVPitch = mul24(numElementsU,numProj);
	uint globalOffsetStatic = mad24(iGIDU, numProj, iProj);
	int iProjTimes2 = iProj << 1; 
	
	float preFreqU = -2.f*M_PI_F*deltaU*deltaU; 
	float preFreqV = -2.f*M_PI_F*deltaV*deltaV;
	float freqU = (iGIDU <= (numElementsU-1)/2) ? iGIDU*preFreqU : (iGIDU-numElementsU)*preFreqU ;
	float preAngle = freqU*shifts[iProjTimes2];
	float preVShift = shifts[iProjTimes2+1];
	
	for(uint iGIDV = 0; iGIDV < numElementsV; iGIDV++){
		uint globalOffset = mad24(iGIDV, iGIDVPitch, globalOffsetStatic);	
		float freqV = (iGIDV <= (numElementsV-1)/2) ? iGIDV*preFreqV : (iGIDV-numElementsV)*preFreqV ;
	    float angle = preAngle + freqV*preVShift;
	    float2 cshift = (float2)(cos(angle), sin(angle));
	    float2 orig = vload2(globalOffset, data);
		float2 res = cmult(cshift, orig);
		vstore2(res, globalOffset, data);
	}

}


kernel void shift(
	__global float *data, 
	__global float *outData,
	__constant float *freqU, 
	__constant float *freqV, 
	__constant float *shifts, 
	int const numProj, 
	int const numElementsU, 
	int const numElementsV) 
{

	int idx = get_global_id(0);
	int nPU = mul24(numProj,numElementsU);
	int n = mul24(nPU, numElementsV);
	
	if (idx >= n) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	int idp = (idx%numProj);
	int idu = (idx%nPU)/numProj;
	int idv = (idx/nPU);
	 
	float2 shiftUV = vload2(idp, shifts);
	float2 freqUV = (float2)(freqU[idu],freqV[idv]);
	float2 orig = vload2(idx, data);
	
	float angle = dot(shiftUV,freqUV);
	float2 cshift = (float2)(native_cos(angle), native_sin(angle));
	
	float2 res = cmult(cshift, orig);
	vstore2(res, idx, outData);
}

*/

// new version
kernel void shift(
	global float *data, 
	global float *freqU, 
	__local float* freqULocal,	
	global float *freqV,
	__local float* freqVLocal,	
	global float *shifts, 
	__local float* shiftsLocal,	
	int const numProj, 
	int const numElementsU, 
	int const numElementsV) 
{

	int iProjLoc = get_local_id(0);
	int iProjGrp = get_group_id(0);
	int locSizex = get_local_size(0);
	int iProj = mad24(iProjGrp,locSizex,iProjLoc);
	
	int iGIDULoc = get_local_id(1);
	int iGIDUGrp = get_group_id(1);
	int locSizey = get_local_size(1);
	int iGIDU = mad24(iGIDUGrp,locSizey,iGIDULoc);
	
	int nrOfLocalThreads = mul24(locSizex,locSizey);
	int nrOfPtsReadInLoops = (numElementsV%nrOfLocalThreads) ? (numElementsV/nrOfLocalThreads + 1) : (numElementsV/nrOfLocalThreads);
	
	for (int i = 0; i < nrOfPtsReadInLoops; i++){
		int linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		int ptsIdx = mad24(iGIDULoc,locSizex,linOffset);
		if (ptsIdx >= numElementsV)
			break;
		freqVLocal[ptsIdx]=freqV[ptsIdx];
	}
	nrOfPtsReadInLoops = (numElementsU%nrOfLocalThreads) ? (numElementsU/nrOfLocalThreads + 1) : (numElementsU/nrOfLocalThreads);
	for (int i = 0; i < nrOfPtsReadInLoops; i++){
		int linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		int ptsIdx = mad24(iGIDULoc,locSizex,linOffset);
		if (ptsIdx >= numElementsU)
			break;
		freqULocal[ptsIdx]=freqU[ptsIdx];
	}
	int numProjTimes2 = numProj*2;
	nrOfPtsReadInLoops = (numProjTimes2%nrOfLocalThreads) ? (numProjTimes2/nrOfLocalThreads + 1) : (numProjTimes2/nrOfLocalThreads);
	for (int i = 0; i < nrOfPtsReadInLoops; i++){
		int linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		int ptsIdx = mad24(iGIDULoc,locSizex,linOffset);
		if (ptsIdx >= numProjTimes2)
			break;
		shiftsLocal[ptsIdx]=shifts[ptsIdx];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // make sure that all points are in local memory
	
	if (iProj >= numProj || iGIDU >= numElementsU) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	uint iGIDVPitch = mul24(numElementsU,numProj);
	uint globalOffsetStatic = mad24(iGIDU, numProj, iProj);
	int iProjTimes2 = iProj << 1; 
	float preAngle = freqULocal[iGIDU]*shiftsLocal[iProjTimes2];
	float preImagShift = shiftsLocal[iProjTimes2+1];
	
	for(uint iGIDV = 0; iGIDV < numElementsV; iGIDV++){
		uint globalOffset = mad24(iGIDV, iGIDVPitch, globalOffsetStatic);	
	    float2 orig = vload2(globalOffset, data);
	    float angle = preAngle + freqVLocal[iGIDV]*preImagShift;
	    float2 cshift = (float2)(native_cos(angle), native_sin(angle));
		float2 res = cmult(cshift, orig);
		vstore2(res, globalOffset, data);
	}
}