kernel void swapDimensionsFwd(
	__global float *dataIn,
	__global float *dataOut, 
	uint const dim1, 
	uint const dim2, 
	uint const dim3
	) 
{

	uint idx = get_global_id(0);
	
	uint dim1dim2 = mul24(dim1,dim2);
	uint dim3dim1 = mul24(dim3,dim1);
	
	uint inX = idx%dim1;
	uint inY = (idx%dim1dim2)/dim1;
	uint inZ = idx/dim1dim2;
	
	uint idxOut = mad24(inY, dim3dim1, mad24(inX,dim3,inZ));

	if (idx >= mul24(dim1dim2,dim3)) {
		return;
	}

	float2 val = vload2(idx, dataIn);
	vstore2(val, idxOut, dataOut);
}

kernel void swapDimensionsBwd(
	__global float *dataIn,
	__global float *dataOut, 
	uint const dim1, 
	uint const dim2, 
	uint const dim3
	) 
{

	uint idx = get_global_id(0);
	
	uint dim1dim2 = mul24(dim1,dim2);
	uint dim3dim2 = mul24(dim3,dim2);
	
	uint inX = idx%dim1;
	uint inY = (idx%dim1dim2)/dim1;
	uint inZ = idx/dim1dim2;
	
	uint idxOut = mad24(inX, dim3dim2, mad24(inZ,dim2,inY));

	if (idx >= mul24(dim1dim2,dim3)) {
		return;
	}

	float2 val = vload2(idx, dataIn);
	vstore2(val, idxOut, dataOut);
}