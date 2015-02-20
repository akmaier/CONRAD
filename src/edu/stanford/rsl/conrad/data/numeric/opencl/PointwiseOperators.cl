/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/


// This is a template for furhter kernels; rename input, output, and cache
kernel void kernelTemplate(global float *input, global float *output, const uint num_elements, local float *cache)
{
    // begin header of eadch kernel
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    
    const uint local_size = get_local_size(0);
    const uint global_size = get_global_size(0);
    
    const uint group_id = get_group_id(0);
    // end header
    
    if (global_id >= num_elements) return; // be careful with this line if you have barriers in your code!
    
    // write your code here
    // ...
}


// ---------------------------------------------
/* Only or testing and benchmarking the significant difference between local and global memory! */
kernel void sumGlobalMemory(global const float *grid, global float *result, int const num_elements)
{
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);
    int work_per_thread = (num_elements/global_size) + 1;
    
    if(global_id >= num_elements) return;
    
    int offset = 0;
    offset = global_id * work_per_thread;
    
    result[global_id] = 0;
    for(int i = 0; i<work_per_thread; ++i)
    {
        if(offset + i < num_elements)
        {
            result[global_id] += grid[offset+i];
        }
    }
}
// ---------------------------------------------


/* Sum of all entries in grid stored in result */
kernel void sum(global float *grid, global float *result, const unsigned int num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}


/* standard deviation of all entries in grid stored in result */
kernel void stddev(global float *grid, global float *result, float const mean, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? pow( (grid[global_id] - mean) , 2) : 0.0f; // sorry for this monster line
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}


/* result = dot(gridA, gridB) */
kernel void dotProduct(global const float *gridA, global const float *gridB, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? (gridA[global_id] * gridB[global_id]) : 0.0f; // sorry for this monster line, too
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}


kernel void dotProduct1(global const float *gridA, global const float *gridB, global float *result, int const numElements)
{
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


/* result = max(grid) */
kernel void maximum(global const float *grid, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] = fmax(cache[local_id], cache[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}


/* result = min(grid) */
kernel void minimum(global const float *grid, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] = fmin(cache[local_id], cache[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}


/* Fill the grid with value */
kernel void fill(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements) return;
    
    grid[global_id] = value;
}



/* gridA = gridA + gridB */
kernel void addBy(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] += gridB[global_id];
}


/* gridA = gridA + value */
kernel void addByVal(global float *grid,  const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] += value;
    
}

/* gridA = gridA - gridB */
kernel void subtractBy(global float *gridA, const global float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] -= gridB[global_id];
}


/* grid = grid - value */
kernel void subtractByVal(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] -= value;
}


/* gridA = gridA * gridB */
kernel void multiplyBy(global float *gridA, global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] *= gridB[global_id];
}


/* grid = grid * value */
kernel void multiplyByVal(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] *= value;
}


/* gridA = gridA / gridB */
kernel void divideBy(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] /= gridB[global_id];
}


/* grid = grid / value */
kernel void divideByVal(global float *grid,  const float value, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id]/= value;
}


/* gridA = gridB */
kernel void copyGrid(global float *gridA,  global const float *gridB, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] = gridB[global_id];
}

/* grid = abs(grid) */
kernel void absolute(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = fabs(grid[global_id]);
}


/* grid = pow(grid, exponent) */
kernel void power(global float *grid, const float exponent, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = pow(grid[global_id], exponent);
}


/* grid = log2(grid) */
kernel void logarithm2(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = log2(grid[global_id]);
}


/* grid = log10(grid) */
kernel void logarithm10(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = log10(grid[global_id]);
}


/* grid = log(grid) */
kernel void logarithm(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = log(grid[global_id]);
}


/* grid = exp(grid) */
kernel void exponential(global float *gridA, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)	
        return;
    
    gridA[global_id] = exp(gridA[global_id]); 
}


kernel void minimalValue(global float *gridA, float value, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)	
        return;
    
    gridA[global_id] = fmax(gridA[global_id], value); 
}





// The following kernels are not supported by the new OpenCLGridOperators class! Please fix this.

#define LOCAL_GROUP_XDIM 256 // Comment by Michael Dorner: Why forcing the group size to 256? Each kernel has an optimal work group size (here called localSize), especially if the kernel uses local memory, what you should do, because local memory is a way faster AND barriers have to be synced only in their own group and not globally!

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



