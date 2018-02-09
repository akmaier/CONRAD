/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
 #define POS(x, y) ((x) + ((y) * width))
 
kernel void kernel3x3Filter2D(global float * image, global float * out, int width, int height, int kernelWidth, global float* kernelArray)
{
	int gidX = get_global_id(0);
	int gidY = get_global_id(1);
	
	// Make sure that coordinates are in range
	if (gidX < kernelWidth/2 || gidY < kernelWidth/2 ||
		gidX > width-kernelWidth/2-1 || gidY > height-kernelWidth/2-1)
	{
		return;
	}
	
	// sum up over kernel
	float sum = 0.0;
	int count = 0;
	for (int i = gidX - kernelWidth/2; i <= gidX + kernelWidth/2; i++) {
		for (int j = gidY - kernelWidth/2; j <= gidY + kernelWidth/2; j++) {
		
			sum += image[POS(i, j)] * kernelArray[count++];
		
		}
	}
	out[POS(gidX, gidY)] = sum; 
	return;
}

