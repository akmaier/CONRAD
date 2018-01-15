/*
 * Copyright (C) 2018 Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
 #define POS(x, y, z) ((x) + ((y) * width) + ((z) * width * height))
 
 kernel void meanFilter6Surrounding3D(global float * image, global float * out, global float * computeValue, int width, int height, int depth)
{
	int gidX = get_global_id(0);
	int gidY = get_global_id(1);
	int gidZ = get_global_id(2);
	
	// Make sure that coordinates are in range
	if (gidX < 0 || gidY < 0 || gidZ < 0 ||
		gidX >= width || gidY >= height || gidZ >= depth)
	{
		return;
	}
	
	if (computeValue[POS(gidX, gidY, gidZ)] == 0 || gidX == 0 || gidY == 0 || gidZ == 0 ||
		gidX == width-1 || gidY == height-1 || gidZ == depth-1)
	{
		out[POS(gidX, gidY, gidZ)] = image[POS(gidX, gidY, gidZ)];
	}
	else
	{
		out[POS(gidX, gidY, gidZ)] = 1.0/6.0 * image[POS(gidX-1, gidY, gidZ)] + 1.0/6.0 * image[POS(gidX+1, gidY, gidZ)]
									 + 1.0/6.0 * image[POS(gidX, gidY-1, gidZ)] + 1.0/6.0 * image[POS(gidX, gidY+1, gidZ)]
									 + 1.0/6.0 * image[POS(gidX, gidY, gidZ-1)] + 1.0/6.0 * image[POS(gidX, gidY, gidZ+1)]; 
	}
	
}
