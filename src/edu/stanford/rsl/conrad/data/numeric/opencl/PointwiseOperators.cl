/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

#define LOCAL_GROUP_XDIM 256

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

/* result = l1(grid) */
kernel void normL1(global const float *grid, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] = fabs(cache[local_id]) + fabs(cache[local_id + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) result[group_id] = cache[0];
}

/* result = negative elements of grid */
kernel void countNegativeElements(global const float *grid, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = 0.0f;
    
    if( (global_id < num_elements) && (grid[global_id] < 0.0f) && !isnan(grid[global_id]) && !isinf(grid[global_id]) ){
    	cache[local_id] = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int s = local_size >> 1; s > 0; s >>= 1) {
        if (local_id < s) {
            cache[local_id] += cache[local_id+s];
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

/* Fill the grids invalid values with value */
kernel void fillInvalidValues(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements) return;
    
    if(isnan(grid[global_id])) grid[global_id] = value;
    if(isinf(grid[global_id])) grid[global_id] = value;
}

/* gridA = gridA + gridB */
kernel void addBy(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    gridA[global_id] += gridB[global_id];
}

/* save add method of gridA = gridA + gridB */
kernel void addBySave(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    if(isnan(gridA[global_id]) || isnan(gridB[global_id]) || isinf(gridA[global_id]) || isinf(gridB[global_id]))
    	gridA[global_id] = 0;
    else
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

/* grid = grid - value */
kernel void addBySaveVal(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    if(isnan(grid[global_id]) || isinf(grid[global_id]))
    	grid[global_id] = 0;
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

/* save subtract method of gridA = gridA + gridB */
kernel void subtractBySave(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    if(isnan(gridA[global_id]) || isnan(gridB[global_id]) || isinf(gridA[global_id]) || isinf(gridB[global_id]))
    	gridA[global_id] = 0;
    else
    	gridA[global_id] -= gridB[global_id];
}

/* grid = grid - value */
kernel void subtractBySaveVal(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    if(isnan(grid[global_id]) || isnan(grid[global_id]))
    	grid[global_id] = 0;
    else
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

/* gridA = gridA * gridB */
kernel void multiplyBySave(global float *gridA, global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    if(isnan(gridA[global_id]) || isinf(gridA[global_id]) || isnan(gridB[global_id]) || isinf(gridB[global_id]))
		gridA[global_id] = 0;
	else
    	gridA[global_id] *= gridB[global_id];
}

/* grid = grid * value */
kernel void multiplyBySaveVal(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    if(isnan(grid[global_id]) || isinf(grid[global_id]))
		grid[global_id] = 0;
	else
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

/* gridA = gridA / gridB */
kernel void divideBySave(global float *gridA,  global const float *gridB, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
	
    if(isnan(gridA[global_id]) || isnan(gridB[global_id]) || isinf(gridA[global_id]) || isinf(gridB[global_id]))
		gridA[global_id] = 0;
	else
   		gridA[global_id] /= gridB[global_id];
}

/* grid = grid / value */
kernel void divideBySaveVal(global float *grid,  const float value, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    if(isnan(grid[global_id]) || isnan(grid[global_id]))
		grid[global_id] = 0;
    else
    	grid[global_id] /= value;
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


/* grid = exp(grid) -- element-wise natural exponential (used by GridOperators.exp) */
kernel void exponential(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);

    if(global_id >= num_elements)
        return;

    grid[global_id] = exp(grid[global_id]);
}


/* grid = log(grid) -- element-wise natural logarithm (used by GridOperators.log) */
kernel void logarithm(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);

    if(global_id >= num_elements)
        return;

    grid[global_id] = log(grid[global_id]);
}


/* ------------------------------------------------------------------------- */
/* GPU random number generation for noise simulation.                        */
/* Philox4x32-10 counter-based RNG (Salmon et al., SC'11 / Random123):       */
/* stateless -- output depends only on (counter, key), so it is fully        */
/* reproducible and needs no per-thread state buffers. Each work item derives */
/* its stream from key=(seed, WELL) and counter=(global_id, draw, 0, 0).      */
/* ------------------------------------------------------------------------- */
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

static uint4 philox4x32_10(uint4 ctr, uint2 key)
{
    for (int i = 0; i < 10; i++) {
        uint hi0 = mul_hi(PHILOX_M0, ctr.x); uint lo0 = PHILOX_M0 * ctr.x;
        uint hi1 = mul_hi(PHILOX_M1, ctr.z); uint lo1 = PHILOX_M1 * ctr.z;
        ctr = (uint4)(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += PHILOX_W0; key.y += PHILOX_W1;
    }
    return ctr;
}

/* uniform float in (0,1) from a 32-bit word (24-bit mantissa, never 0 or 1) */
static float u01(uint x) { return ((x >> 8) + 0.5f) * (1.0f / 16777216.0f); }

/* one Poisson(lambda) draw for work item gid, seed; matches CONRAD's CPU
 * StatisticsUtil: Knuth's product method for small lambda, Hoermann's PTRS
 * transformed rejection (JSCS 1993) for large lambda. Exact across the range. */
static int poisson_sample(float lambda, uint gid, uint seed)
{
    uint2 key = (uint2)(seed, 0x9E3779B9u);
    uint draw = 0u;
    if (lambda <= 0.0f) return 0;
    if (lambda < 30.0f) {                    /* Knuth: multiply uniforms */
        float L = exp(-lambda);
        float p = 1.0f;
        int k = 0;
        do {
            uint4 r = philox4x32_10((uint4)(gid, draw++, 0u, 0u), key);
            p *= u01(r.x);
            k++;
        } while (p > L);
        return k - 1;
    }
    /* PTRS transformed rejection for large lambda */
    float smu   = sqrt(lambda);
    float b     = 0.931f + 2.53f * smu;
    float a     = -0.059f + 0.02483f * b;
    float inv_a = 1.1239f + 1.1328f / (b - 3.4f);
    float vr    = 0.9277f - 3.6224f / (b - 2.0f);
    float loglam = log(lambda);
    for (int iter = 0; iter < 128; iter++) {
        uint4 r = philox4x32_10((uint4)(gid, draw++, 0u, 0u), key);
        float U = u01(r.x) - 0.5f;
        float V = u01(r.y);
        float us = 0.5f - fabs(U);
        float kf = floor((2.0f * a / us + b) * U + lambda + 0.43f);
        if (us >= 0.07f && V <= vr) return (int) kf;
        if (kf < 0.0f) continue;
        if (us < 0.013f && V > us) continue;
        if (log(V * inv_a / (a / (us * us) + b)) <= kf * loglam - lambda - lgamma(kf + 1.0f))
            return (int) kf;
    }
    return (int) lambda;                      /* fallback (should not occur) */
}

/* grid = Poisson(grid) : replace each element (its mean lambda) by a draw */
kernel void poisson(global float *grid, uint const seed, int const num_elements)
{
    int gid = get_global_id(0);
    if (gid >= num_elements)
        return;
    grid[gid] = (float) poisson_sample(grid[gid], (uint) gid, seed);
}

/* grid = N(0,1) : fill each element with a standard normal draw (Box-Muller) */
kernel void standardNormal(global float *grid, uint const seed, int const num_elements)
{
    int gid = get_global_id(0);
    if (gid >= num_elements)
        return;
    uint4 r = philox4x32_10((uint4)((uint) gid, 0u, 0u, 0u), (uint2)(seed, 0x85EBCA6Bu));
    float u1 = u01(r.x), u2 = u01(r.y);
    grid[gid] = sqrt(-2.0f * log(u1)) * cospi(2.0f * u2);
}


/* grid = pow(grid, exponent) */
kernel void power(global float *grid, const float exponent, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = pow(grid[global_id], exponent);
}

/* Remove negative elements */
kernel void minimalValue(global float *grid, const float value, int const num_elements)
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements) return;
    
    if(grid[global_id] < 0)
    	grid[global_id] = value;
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

/* subtraction of two grids with offsets - used for gradients*/
kernel void gradient(global float *result, const global float *gridA,const global float *gridB, 
					 const int xOffset,const int yOffset,const int zOffset, 
					 const int sizeX, const int sizeY, const int sizeZ, 
					 const int offsetleft)
{

	int x = get_global_id(0) + xOffset;
	int y = get_global_id(1) + yOffset;
	int z = get_global_id(2) + zOffset;

	if ( x >= (sizeX+xOffset) ||  y >=  (sizeY+yOffset)||  z >= (sizeZ+zOffset)) 
	{
		return;
	}
	
	int idxbuffer = 0;
	int idx = 0;
	int xIdx = (x >= sizeX || x < 0) ? fmin(fmax(0.0f, x), sizeX-1) : x;
	int yIdx = (y >= sizeY || y < 0) ? fmin(fmax(0.0f, y), sizeY-1) : y;
	int zIdx = (z >= sizeZ || z < 0) ? fmin(fmax(0.0f, z), sizeZ-1) : z;
	
	idx = mad24((z-zOffset),mul24(sizeX,sizeY),mad24((y-yOffset),sizeX,(x-xOffset)));
	idxbuffer = mad24(zIdx,mul24(sizeX,sizeY),mad24(yIdx,sizeX,xIdx));
	
	if(offsetleft == 1) 
		result[idx] = gridA[idxbuffer] - gridB[idx];
	else 
		result[idx] = gridA[idx] - gridB[idxbuffer];

}

/* divergence of X*/
kernel void divergencex(global float *grid, global float *gridBuf, const int sizeX, const int sizeY, const int sizeZ, const int offsetValue)
{
	const int e = get_global_id(0);
	const int f = get_global_id(1);
	
	if (e >= sizeX || f >= sizeY) 
	{
		return;
	}
	int xIdx = 0;
	int xIdxBuf = 0;
	int sizeXY = mul24(sizeY,sizeX);
	
	xIdx = mad24(f,sizeXY,mul24(e,sizeX));
	
	grid[xIdx] = gridBuf[xIdx];
		
	xIdx 	+= (sizeX-1);
	xIdxBuf = xIdx - 1;
		
	grid[xIdx] = -gridBuf[xIdxBuf];

}

/* divergence of Y*/
kernel void divergencey(global float *grid, global float *gridBuf, const int sizeX, const int sizeY, const int sizeZ, const int offsetValue)
{
	const int e = get_global_id(0);
	const int f = get_global_id(1);
	
	if (e >= sizeX || f >= sizeY) 
	{
		return;
	}
	
	int yIdx = 0;
	int yIdxBuf = 0;

	int sizeXY = mul24(sizeY,sizeX);
	
	yIdx = mad24(f,sizeXY,e);
		
	grid[yIdx] = gridBuf[yIdx];
	int tmp = yIdx;
	yIdx 	= mad24(sizeY-1,sizeX,yIdx);
	yIdxBuf = mad24(sizeY-2,sizeX,tmp);
	grid[yIdx] = -gridBuf[yIdxBuf];

}

/* divergence of Z*/
kernel void divergencez(global float *grid, global float *gridBuf, const int sizeX, const int sizeY, const int sizeZ, const int offsetValue)
{
	const int e = get_global_id(0);
	const int f = get_global_id(1);
	
	if (e >= sizeX || f >= sizeY) 
	{
		return;
	}
	
	int zIdx = 0;
	int zIdxBuf = 0;
	int sizeXY = mul24(sizeY,sizeX);
	
	zIdx = mad24(f,sizeX,e);

	grid[zIdx] = gridBuf[zIdx];
		
	int tmp = zIdx;
	zIdx 	= mad24(sizeZ-1,sizeXY,zIdx);
	zIdxBuf = mad24(sizeZ-2,sizeXY,tmp);
	grid[zIdx] = -gridBuf[zIdxBuf];

}