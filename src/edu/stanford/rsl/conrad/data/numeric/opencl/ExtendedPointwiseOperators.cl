
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