/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

//#define LOCAL_GROUP_XDIM 256

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
    
    cache[local_id] = (global_id < num_elements) ? pow( (grid[global_id] - mean), 2) : 0.0f; // sorry for this monster line
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


/* result = max(grid) */
kernel void maximum(global const float *grid, global float *result, int const num_elements, local float *cache)
{
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    const uint group_id = get_group_id(0);
    const uint local_size = get_local_size(0);
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : -MAXFLOAT;
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
    
    cache[local_id] = (global_id < num_elements) ? grid[global_id] : MAXFLOAT;
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


/* grid = exp(grid) */
kernel void expontial(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = exp(grid[global_id]);
}


/* grid = log(grid) */
kernel void logarithm(global float *grid, int const num_elements )
{
    int global_id = get_global_id(0);
    
    if(global_id >= num_elements)
        return;
    
    grid[global_id] = log(grid[global_id]);
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


