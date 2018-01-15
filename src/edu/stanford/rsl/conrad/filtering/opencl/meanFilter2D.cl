/*
 * Copyright (C) 2017 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
 #define POS(x, y) ((x) + ((y) * width))
 
kernel void meanFilter2D(global float * image, global float * out, int width, int height, int kernelWidth, int kernelHeight, float factor)
{
	int gidX = get_global_id(0);
	int gidY = get_global_id(1);
	
	// Make sure that coordinates are in range
	if (gidX < kernelWidth/2 || gidY < kernelHeight/2 ||
		gidX > width-kernelWidth/2-1 || gidY > height-kernelHeight/2-1)
	{
		return;
	}
	
	// sum up over kernel
	float sum = 0.0;
	for (int i = gidX - kernelWidth/2; i <= gidX + kernelWidth/2; i++) {
		for (int j = gidY - kernelHeight/2; j <= gidY + kernelHeight/2; j++) {
		
			sum += image[POS(i, j)];
		
		}
	}
	out[POS(gidX, gidY)] = factor * sum; 
	return;
}

