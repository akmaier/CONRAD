/*
 * Copyright (C) 2015 - Katrin Mentl, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/


typedef float2 cfloat;

#define LOCAL_GROUP_XDIM 256
#define I ((cfloat)(0.0, 1.0))

#define cmult(a,b) ((cfloat)(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))
#define cmultConj(a,b) ((cfloat)(mad((a).y, (b).y, (a).x * (b).x), mad(-(a).y, (b).x, (a).x * (b).y)))
#define cmultConjReal(a,b) (mad((a).y, (b).y, (a).x * (b).x))
#define squaredLength(a) (mad((a).x, (a).x, (a).y * (a).y))


// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void sumEnergyGradient(
            __global const float * x, // input vector
            __global const float * shiftedX, //gradient input vector, added
            __global const float * freqUAxis,
            __global const float * freqVAxis,
            __global const float * freqPAxis, 
			__global const float * mask, // mask over proj and u, 1 for values to be summed, 0 else
            __global float * r, // result vector
            uint n_per_group, // elements processed per group
            uint n_per_work_item, // elements processed per work item
			uint n0, // dim proj
			uint n01, // numElementsMask
			uint n, // numElements, add comma
			uint grad_idx,
			float angleInc
			)
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);
    
	float2 priv_acc = (float2)(0.f,0.f); // accumulator in private memory
    __local float2 lcl_acc[LOCAL_GROUP_XDIM]; // accumulators in local memory
    
    uint grp_off = mul24(n_per_group, grp_id); // group offset
    uint lcl_off = grp_off + lcl_id; // local offset
    
    //compute ext_p and even from grad_idx
    uint ext_p = grad_idx;
    //bool even = (grad_idx%2==0);
    
    // Accumulate products over n_per_work_item elements.
    float priv_val = 0.f;
    for ( uint i = 0; i < n_per_work_item; i++, lcl_off += LOCAL_GROUP_XDIM)
    {
    	//get the indices for u, v, proj
    	/*uint v = lcl_off/(n011);
		uint u = (lcl_off%(n011))/n0;
		uint proj = (lcl_off%(n011))%n0;*/
		
		//different approach to indices computation
		uint proj = lcl_off%n0;
		uint v = lcl_off/n01;
		uint maskIdx = (lcl_off%n01);
		uint u = maskIdx/n0;
		
		//compute shift in freqP axis using ext_p (equals FFT in p-direction)
		float ext_p_shift_angle = freqPAxis[proj]*ext_p*angleInc;
		float2 ext_p_shift = (float2)(-native_sin(ext_p_shift_angle),native_cos(ext_p_shift_angle));

		
		//compute index for position corresponding to (ext_p, u, v)
		uint index = lcl_off + ext_p - proj;
		float2 shifted_val = vload2(index, shiftedX);
		
		float2 val = cmult(shifted_val, ext_p_shift);
		
        bool in_range = (lcl_off < n && mask[maskIdx]);
		
		//derive for u direction and for v direction simultaneously
        priv_acc += ( in_range ) ? ((float2)(freqUAxis[u],freqVAxis[v]))*cmultConjReal(vload2(lcl_off, x), val) : ((float2)(0.f,0.f));
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
        vstore2(priv_acc,grp_id,r);
    }
}