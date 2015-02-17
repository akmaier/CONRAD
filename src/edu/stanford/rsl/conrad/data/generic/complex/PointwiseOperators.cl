/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

//2 component vector to hold the real and imaginary parts of a complex number:
typedef float2 cfloat;

#define I ((cfloat)(0.0, 1.0))


/*
 * Return Real (Imaginary) component of complex number:
 */
inline float  real(cfloat a){
     return a.x;
}
inline float  imag(cfloat a){
     return a.y;
}

/*
 * Get the modulus of a complex number (its length):
 */
inline float cmod(cfloat a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

inline cfloat cfmin(cfloat a, cfloat b){
	return (cmod(a) < cmod(b)) ? a : b;
}
inline cfloat cfmax(cfloat a, cfloat b){
	return (cmod(a) > cmod(b)) ? a : b;
}

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline float carg(cfloat a){
    if(a.x > 0){
        return atan(a.y / a.x);

    }else if(a.x < 0 && a.y >= 0){
        return atan(a.y / a.x) + M_PI_F;

    }else if(a.x < 0 && a.y < 0){
        return atan(a.y / a.x) - M_PI_F;

    }else if(a.x == 0 && a.y > 0){
        return M_PI_F/2;

    }else if(a.x == 0 && a.y < 0){
        return -M_PI_F/2;

    }else{
        return 0;
    }
}

/*
 * Multiply two complex numbers:
 *
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 */
inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


/*
 * Divide two complex numbers:
 *
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 * 
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 * 
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag 
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 * 
 */
inline cfloat cdiv(cfloat a, cfloat b){
    return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}


/*
 *  Square root of complex number.
 *  Although a complex number has two square roots, numerically we will
 *  only determine one of them -the principal square root, see wikipedia
 *  for more info: 
 *  http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
 inline cfloat csqrt(cfloat a){
     return (cfloat)( sqrt(cmod(a)) * cos(carg(a)/2),  sqrt(cmod(a)) * sin(carg(a)/2));
 }
 
  inline cfloat cToPolar(cfloat a){
     return (cfloat)(cmod(a),carg(a));
 }
 
   inline cfloat cToCartesian(cfloat a){
     return (cfloat)(a.x*cos(a.y),a.x*sin(a.y));
 }
 

kernel void fill(global float2 *grid, global const float2* value, int const numElements){

	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;	
	
	grid[iGID] = value[0];

}


// first step of sum
kernel void sum(global const float2 *grid, global float2 *result, int const numElements){

	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = 0;
	offset = iGID*workPerThread;	
	
	result[iGID] = (0,0);
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] += grid[offset+i];		
		}
	}
}


// numElements must always be an even value
kernel void maximum(global const float2 *grid, global float2 *result, int const numElements){
	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = iGID*workPerThread;	
	
	result[iGID] = (0,0);
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] = cfmax(grid[offset+i], result[iGID]);		
		}
	}
}

// numElements must always be an even value
kernel void minimum(global const float2 *grid, global float2 *result, int const numElements){
	int iGID = get_global_id(0);
	int totalThreads = get_global_size(0);
	int workPerThread = (numElements/totalThreads)+1;	
		
	int offset = iGID*workPerThread;	
	
	result[iGID] = (INFINITY,INFINITY);
	for(int i = 0; i<workPerThread; ++i)
	{
		if(offset +i < numElements)
		{
			result[iGID] = cfmin(grid[offset+i], result[iGID]);		
		}
	}
}

kernel void addBy(global float2 *grid1,  global const float2 *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] += grid2[iGID]; 

}

kernel void addByVal(global float2 *grid, global const float2 *value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] += value[0];
}


kernel void subtractBy(global float2 *grid1,  global const float2 *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;

	grid1[iGID] -= grid2[iGID];

}

kernel void subtractByVal(global float2 *grid, global const float2 *value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] -= value[0];

}


kernel void multiplyBy(global float2 *grid1,  global const float2 *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = cmult(grid1[iGID],grid2[iGID]);

}

kernel void multiplyByVal(global float2 *grid,  global const float2* value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] = cmult(grid[iGID],value[0]); 

}


kernel void divideBy(global float2 *grid1,  global const float2 *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = cdiv(grid1[iGID],grid2[iGID]); 

}

kernel void divideByVal(global float2 *grid,  global const float2* value, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID] = cdiv(grid[iGID],value[0]); 

}

kernel void copy(global float2 *grid1,  global const float2 *grid2, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID] = grid2[iGID]; 

}

kernel void absolute(global float2 *grid1, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid1[iGID].x = cmod(grid1[iGID]);
	grid1[iGID].y = 0.f;

}

kernel void power(global float2 *grid1, global const float2* exponent, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	float2 polar = cToPolar(grid1[iGID]);
	grid1[iGID] = cToCartesian((pow(polar.x,exponent[0].x), polar.y*exponent[0].x));

}

kernel void conj(global float2 *grid, int const numElements )
{
	int iGID = get_global_id(0);
	
	if(iGID >= numElements)	
		return;
		
	grid[iGID].y = -grid[iGID].y;

}