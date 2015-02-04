/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

#define LOCAL_GROUP_XDIM 256

kernel void fill(global float *grid, const float value, int const numElements){

	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;	
	
	grid[iGID] = value;

}


// first step of sum
kernel void sum(global const float *grid, global float *result, int const numElements){

	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = 0;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] += grid[offset+i];		
		}
	}
	
}

// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void sum_persist_kernel(
__global const float * x, // input vector
__global float * r, // result vector
uint n_per_group, // elements processed per group
uint n_per_work_item, // elements processed per work item
uint n // input vector size
)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);

    float priv_acc = 0.f; // accumulator in private memory
    __local float lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
	
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset

    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
		// Be wary of out of range offsets, just add 0 if out of range.
		// This code uses conditional expressions rather than ifs for efficiency.
		bool in_range = ( lcl_off < n );
		uint lcl_off2 = ( in_range ) ? lcl_off : 0;
		priv_val = x[lcl_off2]; // read the global memory
		priv_acc += ( in_range ) ? priv_val : 0.f; // accumulate result
    }

    // Store result accumulated so far to local accumulator.
    lcl_acc[lcl_id] = priv_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find the sum of the accumulated products.
    uint dist = LOCAL_GROUP_XDIM; // i.e., get_local_size(0);
    while ( dist > 1 )
    {
		dist >>= 1;
		if ( lcl_id < dist )
		{
			// Private memory accumulator avoids extra local memory read.
			priv_acc += lcl_acc[lcl_id + dist];
			lcl_acc[lcl_id] = priv_acc;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result.
    if ( lcl_id == 0 )
    {
		r[grp_id] = priv_acc;
    }
}


kernel void stddev(global const float *grid, global float *result, float const mean, int const numElements){

	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = 0;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			float temp = (grid[offset+i]-mean);
			result[iGID] += temp*temp;		
		}
	}
	
}

// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void stddev_persist_kernel(
__global const float * x, // input vector
float const mean, // the mean value
__global float * r, // result vector
uint n_per_group, // elements processed per group
uint n_per_work_item, // elements processed per work item
uint n // input vector size
)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);

    float priv_acc = 0.f; // accumulator in private memory
    __local float lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
	
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset

    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
		// Be wary of out of range offsets, just add 0 if out of range.
		// This code uses conditional expressions rather than ifs for efficiency.
		bool in_range = ( lcl_off < n );
		uint lcl_off2 = ( in_range ) ? lcl_off : 0;
		priv_val = x[lcl_off2]-mean; // remove mean
		priv_val *= priv_val; // build square
		priv_acc += ( in_range ) ? priv_val : 0.f; // accumulate result
    }

    // Store result accumulated so far to local accumulator.
    lcl_acc[lcl_id] = priv_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find the sum of the accumulated products.
    uint dist = LOCAL_GROUP_XDIM; // i.e., get_local_size(0);
    while ( dist > 1 )
    {
		dist >>= 1;
		if ( lcl_id < dist )
		{
			// Private memory accumulator avoids extra local memory read.
			priv_acc += lcl_acc[lcl_id + dist];
			lcl_acc[lcl_id] = priv_acc;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result.
    if ( lcl_id == 0 )
    {
		r[grp_id] = priv_acc;
    }
}


kernel void dotProduct(global const float *gridA, global const float *gridB, global float *result, int const numElements){

	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = 0;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] += (gridB[offset+i]*gridA[offset+i]);		
		}
	}
	
}

// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void dot_persist_kernel(
__global const float * x, // input vector
__global const float * y, // input vector
__global float * r, // result vector
uint n_per_group, // elements processed per group
uint n_per_work_item, // elements processed per work item
uint n // input vector size
)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);

    float priv_acc = 0.f; // accumulator in private memory
	
    __local float lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
	
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset

    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
		// Be wary of out of range offsets, just add 0 if out of range.
		// This code uses conditional expressions rather than ifs for efficiency.
		bool in_range = ( lcl_off < n );
		uint lcl_off2 = ( in_range ) ? lcl_off : 0;
		priv_val = x[lcl_off2] * y[lcl_off2]; // multiply elements
		priv_acc += ( in_range ) ? priv_val : 0.f; // accumulate result
    }

    // Store result accumulated so far to local accumulator.
    lcl_acc[lcl_id] = priv_acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find the sum of the accumulated products.
    uint dist = LOCAL_GROUP_XDIM; // i.e., get_local_size(0);
    while ( dist > 1 )
    {
		dist >>= 1;
		if ( lcl_id < dist )
		{
			// Private memory accumulator avoids extra local memory read.
			priv_acc += lcl_acc[lcl_id + dist];
			lcl_acc[lcl_id] = priv_acc;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }
	
    // Store the result.
    if ( lcl_id == 0 )
    {
		r[grp_id] = priv_acc;
    }
}

kernel void weightedSSD(global const float *gridA, global const float *gridB, const float weightB, global float *result, int const numElements){

	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = 0;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			float diff = ((weightB*gridB[offset+i]-gridA[offset+i]));
			result[iGID] += diff*diff;
		}
	}
}


// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void weightedSSD_persist_kernel(
__global float * signal1, // input vector
__global float * signal2, // input vector
float weightY, // the weight for second array
float addY, // the offset for second array
__global float * r, // result vector
uint n_per_group, // elements processed per group
uint n_per_work_item, // elements processed per work item
uint numElements // input vector size
)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);
	
    float priv_acc = 0.f; // accumulator in private memory
    __local float lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
	
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset

	
    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
		// Be wary of out of range offsets, just add 0 if out of range.
		// This code uses conditional expressions rather than ifs for efficiency.
		bool in_range = ( lcl_off < numElements);
		uint lcl_off2 = ( in_range ) ? lcl_off : 0;
		priv_val = (signal1[lcl_off2] - (weightY*signal2[lcl_off2]+addY)); // multiply elements
		priv_acc += ( in_range ) ? (priv_val*priv_val) : 0.f; // accumulate result
    }

	
    // Store result accumulated so far to local accumulator.
    lcl_acc[lcl_id] = priv_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
	

    // Find the sum of the accumulated products.
    uint dist = LOCAL_GROUP_XDIM; // i.e., get_local_size(0);
    while ( dist > 1 )
    {
		dist >>= 1;
		if ( lcl_id < dist )
		{
			// Private memory accumulator avoids extra local memory read.
			priv_acc += lcl_acc[lcl_id + dist];
			lcl_acc[lcl_id] = priv_acc;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result.
    if ( lcl_id == 0 )
    {
		r[grp_id] = priv_acc;
    }
}


// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void weightedDotProduct_persist_kernel(
__global float * signal1, // input vector
__global float * signal2, // input vector
float weightY, // the weight for second array
float addY, // the offset for second array
__global float * r, // result vector
uint n_per_group, // elements processed per group
uint n_per_work_item, // elements processed per work item
uint numElements // input vector size
)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);
	
    float priv_acc = 0.f; // accumulator in private memory
    __local float lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
	
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset

	
    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
		// Be wary of out of range offsets, just add 0 if out of range.
		// This code uses conditional expressions rather than ifs for efficiency.
		bool in_range = ( lcl_off < numElements);
		uint lcl_off2 = ( in_range ) ? lcl_off : 0;
		priv_val = (signal1[lcl_off2] * (weightY*signal2[lcl_off2]+addY)); // multiply elements
		priv_acc += ( in_range ) ? priv_val : 0.f; // accumulate result
    }

	
    // Store result accumulated so far to local accumulator.
    lcl_acc[lcl_id] = priv_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
	

    // Find the sum of the accumulated products.
    uint dist = LOCAL_GROUP_XDIM; // i.e., get_local_size(0);
    while ( dist > 1 )
    {
		dist >>= 1;
		if ( lcl_id < dist )
		{
			// Private memory accumulator avoids extra local memory read.
			priv_acc += lcl_acc[lcl_id + dist];
			lcl_acc[lcl_id] = priv_acc;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the result.
    if ( lcl_id == 0 )
    {
		r[grp_id] = priv_acc;
    }
}


// numElements must always be an even value
kernel void maximum(global const float *grid, global float *result, int const numElements){
	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = -99999999;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] = fmax(grid[offset+i], result[iGID]);		
		}
	}
}

// numElements must always be an odd value
kernel void minimum(global const float *grid, global float *result, int const numElements){
	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = 99999999;
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] = fmin(grid[offset+i], result[iGID]);		
		}
	}
}



kernel void addBy(global float *grid1,  global const float *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] += grid2[iGID]; 

}

kernel void addByVal(global float *grid,  const float value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] += value; 

}


kernel void subtractBy(global float *grid1,  global const float *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] -= grid2[iGID]; 

}

kernel void subtractByVal(global float *grid,  const float value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] -= value; 

}


kernel void multiplyBy(global float *grid1,  global const float *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] *= grid2[iGID]; 

}

kernel void multiplyByVal(global float *grid,  const float value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] *= value; 

}


kernel void divideBy(global float *grid1,  global const float *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] /= grid2[iGID]; 

}

kernel void divideByVal(global float *grid,  const float value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID]/= value; 

}

kernel void copy(global float *grid1,  global const float *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = grid2[iGID]; 

}

kernel void absolute(global float *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = fabs(grid1[iGID]); 

}

kernel void power(global float *grid1, const float exponent, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = pow(grid1[iGID],exponent); 

}

kernel void logarithm2(global float *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = log2(grid1[iGID]); 

}

kernel void logarithm10(global float *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = log10(grid1[iGID]); 

}

kernel void logarithm(global float *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = log(grid1[iGID]); 

}

kernel void exponent(global float *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = exp(grid1[iGID]); 

}

kernel void minimalValue(global float *grid1, float value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = fmax(grid1[iGID], value); 

}

