/*
 * Copyright (C) 2015 Wolfgang Aichinger, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

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
	__local float* shiftsLocal	
	//uint const numProj, 
	//uint const numElementsU, 
	//uint const numElementsV
	) 
{

	uint iProjLoc = get_local_id(0);
	uint iProjGrp = get_group_id(0);
	uint locSizex = get_local_size(0);
	uint iProj = mad24(iProjGrp,locSizex,iProjLoc);
	
	uint iGIDVLoc = get_local_id(1);
	uint iGIDVGrp = get_group_id(1);
	uint locSizey = get_local_size(1);
	uint iGIDV = mad24(iGIDVGrp,locSizey,iGIDVLoc);
	
	uint nrOfLocalThreads = mul24(locSizex,locSizey);
	uint nrOfPtsReadInLoops = (numElementsV%nrOfLocalThreads) ? (numElementsV/nrOfLocalThreads + 1) : (numElementsV/nrOfLocalThreads);
	
	for (uint i = 0; i < nrOfPtsReadInLoops; i++){
		uint linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		uint ptsIdx = mad24(iGIDVLoc,locSizex,linOffset);
		if (ptsIdx >= numElementsV)
			break;
		freqVLocal[ptsIdx]=freqV[ptsIdx];
	}
	nrOfPtsReadInLoops = (numElementsU%nrOfLocalThreads) ? (numElementsU/nrOfLocalThreads + 1) : (numElementsU/nrOfLocalThreads);
	for (uint i = 0; i < nrOfPtsReadInLoops; i++){
		uint linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		uint ptsIdx = mad24(iGIDVLoc,locSizex,linOffset);
		if (ptsIdx >= numElementsU)
			break;
		freqULocal[ptsIdx]=freqU[ptsIdx];
	}
	uint numProjTimes2 = numProj*2;
	nrOfPtsReadInLoops = (numProjTimes2%nrOfLocalThreads) ? (numProjTimes2/nrOfLocalThreads + 1) : (numProjTimes2/nrOfLocalThreads);
	for (uint i = 0; i < nrOfPtsReadInLoops; i++){
		uint linOffset = mad24(i,nrOfLocalThreads,iProjLoc);
		uint ptsIdx = mad24(iGIDVLoc,locSizex,linOffset);
		if (ptsIdx >= numProjTimes2)
			break;
		shiftsLocal[ptsIdx]=shifts[ptsIdx];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // make sure that all points are in local memory
	
	if (iProj >= numProj || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	uint globalOffsetStatic = mad24(iGIDV, mul24(numElementsU,numProj), iProj);
	uint iProjTimes2 = iProj << 1; 
	float preUShift = shiftsLocal[iProjTimes2];
	float preAngle = freqVLocal[iGIDV]*shiftsLocal[iProjTimes2+1];
	
	#pragma unroll 4
	for(uint iGIDU = 0; iGIDU < numElementsU; iGIDU++){ 
		uint globalOffset = mad24(iGIDU, numProj, globalOffsetStatic);	
	    float2 orig = vload2(globalOffset, data);
		float currAngle = freqULocal[iGIDU]*preUShift;
		/*
		float2 cshift = (0.f,0.f);
		bool uhalf = (iGIDU == numElementsU/2);
		bool vhalf = (iGIDV == numElementsV/2);
		if (uhalf || vhalf)
		{
			if(uhalf){
				cshift*=native_cos(currAngle);
			}
			if(vhalf){
				cshift*=native_cos(preAngle);
			}
		}
		else
		{
			float angle = currAngle+preAngle;
			cshift = (float2)(native_cos(angle), native_sin(angle));
		}
		*/
		float angle = currAngle+preAngle;
		float2 cshift = (float2)(native_cos(angle), native_sin(angle));
		float2 res = cmult(cshift, orig);
		vstore2(res, globalOffset, data);
	}
}

kernel void blackmanVdirection(
	global float *dataIn,
	global float *res,
	__constant float *filt
	) 
{
	uint iProjLoc = get_local_id(0);
	uint iProjGrp = get_group_id(0);
	uint locSizex = get_local_size(0);
	uint iProj = mad24(iProjGrp,locSizex,iProjLoc);
	
	uint iGIDVLoc = get_local_id(1);
	uint iGIDVGrp = get_group_id(1);
	uint locSizey = get_local_size(1);
	uint iGIDV = mad24(iGIDVGrp,locSizey,iGIDVLoc);
	
	if (iProj >= numProj || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	uint iGIDVPitch = mul24(numElementsU,numProj);
	uint globalOffsetStatic = mad24(iGIDV, iGIDVPitch, iProj);
	
	
	#pragma unroll 4
	for(uint iGIDU = 0; iGIDU < numElementsU; iGIDU++){ 
		uint globalOffset = mad24(iGIDU, numProj, globalOffsetStatic);
		
		float2 sum  = (float2)(0.f,0.f);
		for(int k = 0; k < 7; k++)
		{
			float val = filt[k];
			int r = 7-k;
			int deltaV = (iGIDV < r) ? numElementsV-r : -r;
			float2 orig1 = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);
			deltaV = (iGIDV > (numElementsV-r-1)) ? -numElementsV+r : r;
			float2 orig2 = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);
			sum += (float2)((orig1+orig2)*val);
		}
		
		float val = filt[7];
		float2 orig1 = vload2(globalOffset, dataIn);
		sum += (float2)(orig1*val);
		
		//blackman window [10.24 -64 107.52 -64 10.24]
		vstore2(sum, globalOffset, res);
	}
}


/*
kernel void blackmanVdirection(
	global float *dataIn,
	global float *res
	) 
{
	uint iProjLoc = get_local_id(0);
	uint iProjGrp = get_group_id(0);
	uint locSizex = get_local_size(0);
	uint iProj = mad24(iProjGrp,locSizex,iProjLoc);
	
	uint iGIDVLoc = get_local_id(1);
	uint iGIDVGrp = get_group_id(1);
	uint locSizey = get_local_size(1);
	uint iGIDV = mad24(iGIDVGrp,locSizey,iGIDVLoc);
	
	if (iProj >= numProj || iGIDV >= numElementsV) {
		return;
	}
	// for loop over all projections (simple implementation, later vector with shifts and position and iteration over this vector)

	// ((iGIDV*numElementsU+iGIDU) * numProj) + iProj = iGIDV*numElementsU*numProj + iGIDU*numProj + iProj
	uint iGIDVPitch = mul24(numElementsU,numProj);
	uint globalOffsetStatic = mad24(iGIDV, iGIDVPitch, iProj);
	
	#pragma unroll 4
	for(uint iGIDU = 0; iGIDU < numElementsU; iGIDU++){ 
		uint globalOffset = mad24(iGIDU, numProj, globalOffsetStatic);
		
		int deltaV = (iGIDV < 2) ? numElementsV-2 : -2;
		float2 orig = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);
		float2 sum = (orig*10.24f);
		deltaV = (iGIDV < 1) ? numElementsV-1 : -1;
		orig = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);	
		sum += (orig*-64.f);		
		orig = vload2(globalOffset, dataIn);
		sum += (orig*107.52f);
		deltaV = (iGIDV > numElementsV-2) ? -numElementsV+1 : +1;
		orig = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);
		sum += (orig*-64.f);
		deltaV = (iGIDV > numElementsV-3) ? -numElementsV+2 : +2;
		orig = vload2(mad24(deltaV,(int)iGIDVPitch,(int)globalOffset), dataIn);
		sum += (orig*10.24f);
		//[10.24 -64 107.52 -64 10.24]
		vstore2(sum, globalOffset, res);
	}
}
*/