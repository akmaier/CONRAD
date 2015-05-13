/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/


typedef float2 cfloat;

#define LOCAL_GROUP_XDIM 256
#define I ((cfloat)(0.0, 1.0))

inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline cfloat add(cfloat a, cfloat b){
  return (cfloat)(a.x + b.x, a.y + b.y);
}
inline float getMagnitude(cfloat a){
  return sqrt(a.x*a.x + a.y*a.y);
}

/*float2 fft(__private const float * x,__private const float * dftMat, uint proj, uint u, uint v,uint  n0, uint n1,uint n2){
  float2 value = (float2)(0,0);
  for(int k = 0; k < n0; k++){
    int offSetA = (proj * n0 + k)*2;
    float2 elementA = (float2)(dftMat[offSetA], dftMat[offSetA + 1]);
    
    int offSetB = (v*n0*n1 + u*n1 + k)*2;
    float2 elementB = (float2)(x[offSetB], x[offSetB + 1]);
    value = add(value, cmult(elementA, elementB));
  }
  
  return value;
  
}
*/

// Kernel for part 1 of dot product, version 3.
// Please see http://developer.amd.com/community/blog/2012/07/05/efficient-dot-product-implementation-using-persistent-threads/ for detailed explanation
__kernel __attribute__((reqd_work_group_size(LOCAL_GROUP_XDIM, 1, 1)))
void sumFFTEnergy(
                        __global const float * x, // input vector
			__global const float * dftMat, // dft matrix
			__global const float * mask, // mask over proj and u, 1 for values to be summed, 0 else
                        __global float * r, // result vector
                        uint n_per_group, // elements processed per group
                        uint n_per_work_item, // elements processed per work item
                        uint n0, // dim proj
			uint n1, // dim u
			uint n2  // dim v 
                        )
{
    // uint id = get_global_id(0); // unused here
    uint lcl_id = get_local_id(0);
    uint grp_id = get_group_id(0);
    uint n = n0*n1*n2;
    uint n01 = n0*n1;
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
        //uint lcl_off2 = ( in_range ) ? lcl_off : 0;
	if(in_range){
	  
	  uint v = lcl_off/(n01);
	  //uint u = (lcl_off - v*n01)/n0;
	  //uint proj = lcl_off-(v*n01+u*n0);
	  uint u = (lcl_off%(n01))/n0;
	  uint proj = (lcl_off%(n01))%n0;
	  
	  if(mask[u*n0 + proj]){
	     //****************************************************************************************
		//****************************************************************************************
		// HIER MUESSTE UNSERE FFT AUSWERTUNG FUER lcl_off2 ERFOLGEN ABER NUR WENN lcl_off2 in der 
		// MASKE 1 ist. DANACH DEN MAGNITUDENWERT IN priv_val SCHREIBEN
		// DIE FFT KANN MAN AUCH IN EINE ANDERE METHODE AUSLAGERN
		float2 value = (float2)(0,0);
		for(int k = 0; k < n0; k++){
		  int offSetA = (proj * n0 + k)*2;
		  float2 elementA = (float2)(dftMat[offSetA], dftMat[offSetA + 1]);
    	  
    	  // ERROR HAPPENS SOMEWHERE HERE
		  int offSetB = (v*n0*n1 + u*n0 + k)*2;
		  float2 elementB = (float2)(x[offSetB], x[offSetB + 1]);
		  value = add(value, cmult(elementA, elementB));
		}
	     
		float pixEnergy = getMagnitude( value);
		
		priv_val = pixEnergy; // read the global memory
		//****************************************************************************************
		//****************************************************************************************
		//****************************************************************************************		
	    
	  }
	  else{
	    priv_val = 0;
	  }
	 
	  
	}
	else{
	  priv_val = 0;
	}
	
			
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