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