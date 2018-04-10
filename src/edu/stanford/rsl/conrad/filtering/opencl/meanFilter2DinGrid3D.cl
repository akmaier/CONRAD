/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
 #define POS(x, y, z) ((x) + ((y) * width) + ((z) * width * height))
 
kernel void meanFilter2DinGrid3D(global float * image, global float * out, int width, int height, int depth, int kernelWidth, int kernelHeight, float factor)
{
	int gidX = get_global_id(0);
	int gidY = get_global_id(1);
	int gidZ = get_global_id(2);
	
	// Make sure that coordinates are in range
	if (gidX < kernelWidth/2 || gidY < kernelHeight/2 || gidZ < 0 ||
		gidX > width-kernelWidth/2-1 || gidY > height-kernelHeight/2-1 || gidZ > depth-1)
	{
		return;
	}
	
	// sum up over kernel
	double sum = 0.0;
	for (int i = gidX - kernelWidth/2; i <= gidX + kernelWidth/2; i++) {
		for (int j = gidY - kernelHeight/2; j <= gidY + kernelHeight/2; j++) {
		
			sum += image[POS(i, j, gidZ)];
		
		}
	}

	out[POS(gidX, gidY, gidZ)] = factor * sum; 
}

