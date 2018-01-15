/*
 * Copyright (C) 2017 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
 #define POS(x, y) ((x) + ((y) * width))
 
kernel void binarization2D(global float * image, global float * out, int width, int height, float thres)
{
	int gidX = get_global_id(0);
	int gidY = get_global_id(1);
	
	// Make sure that coordinates are in range
	if (gidX < 0 || gidY < 0 ||
		gidX > width-1 || gidY > height-1)
	{
		return;
	}

	out[POS(gidX, gidY)] = (image[POS(gidX, gidY)] >= thres) ? 1.0 : 0.0;
	return;
}

