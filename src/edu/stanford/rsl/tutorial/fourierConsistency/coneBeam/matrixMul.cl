/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */

typedef float2 cfloat;

#define I ((cfloat)(0.0, 1.0))

inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline cfloat add(cfloat a, cfloat b){
  return (cfloat)(a.x + b.x, a.y + b.y);
}



// OpenCL Kernel
kernel void matrixMul(global float* C, 
          global float* A, 
          global float* B, 
          int wA, int hA, int wB, int hB)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   if (tx >= wB || ty >= hA ){
     return;
  }
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * wB + tx] = value;
}


kernel void complexMatrixMul(global float* C, global float* A, global float* B,
			int wA, int hA, int wB, int hB)
{
  int tx = get_global_id(0);
  int ty = get_global_id(1);
  
  if(tx >= wB || ty >= hA){
    return;
  }
  float2 value = (float2) (0,0);
  for(int k = 0; k < wA; ++k){
    int offSetA = (ty * wA + k)*2;
    float2 elementA = (float2)(A[offSetA], A[offSetA + 1]);
    int offSetB = (k*wB + tx)*2;
    float2 elementB = (float2)(B[offSetB], B[offSetB + 1]);
    value = add(value, cmult(elementA, elementB));
  }
  int offSetC = (ty*wB + tx)*2;
  C[offSetC] = value.x;
  C[offSetC +1 ] = value.y;
}

kernel void dftMatrixMul(global float* C, global float* A, global float* B,
	int wA, int hA, int dimB0, int dimB1, int dimB2)
{
  int proj = get_global_id(0);
  int u = get_global_id(1);
  int v = get_global_id(2);
  
  if(proj >= dimB0 || u >= dimB1 || v >= dimB2){
    return;
  }
  
  float2 value = (float2)(0,0);
  for(int k = 0; k < dimB0; k++){
    int offSetA = (proj * wA + k)*2;
    float2 elementA = (float2)(A[offSetA], A[offSetA + 1]);
    
    int offSetB = (v*dimB0*dimB1 + u*dimB0 + k)*2;
    float2 elementB = (float2)(B[offSetB], B[offSetB + 1]);
    value = add(value, cmult(elementA, elementB));
  }
  
  int offSetC = (v*dimB0*dimB1 + u *dimB0 + proj)*2;
  C[offSetC] = value.x;
  C[offSetC +1 ] = value.y;
}