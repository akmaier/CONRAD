/*
 * Copyright (C) 2014 Benedikt Lorch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
#define POS(x, y, z) ((x) + ((y) * width) + ((z) * width * height))

// Function declarations
float geomClose(float3 point1, float3 point2, float sigma_spatial);
float photomClose(float one, float two, float sigma_photo);
float computeKernel(global float* image, int width, int height, int depth, int x, int y, int z, global float * geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo, global float * jbfTemplate, int jbf);
kernel void jointBilateralFilter3D(global float * image, global float * out, int width, int height, int depth, global float * geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo, global float * jbfTemplate, int jbf);
kernel void bilateralFilter3D(global float * image, global float * out, int width, int height, int depth, global float* geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo);


kernel void bilateralFilter3D(global float * image, global float * out, int width, int height, int depth, global float* geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo)
{
	jointBilateralFilter3D(image, out, width, height, depth, geometricKernel, kernelWidth, sigma_spatial, sigma_photo, 0, 0);
}


kernel void jointBilateralFilter3D(global float * image, global float * out, int width, int height, int depth, global float * geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo, global float * jbfTemplate, int jbf)
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

	out[POS(gidX, gidY, gidZ)] = computeKernel(image, width, height, depth, gidX, gidY, gidZ, geometricKernel, kernelWidth, sigma_spatial, sigma_photo, jbfTemplate, jbf);
}


float computeKernel(global float* image, int width, int height, int depth, int x, int y, int z, global float* geometricKernel, int kernelWidth, float sigma_spatial, float sigma_photo, global float * jbfTemplate, int jbf)
{
	// Boundaries: Weight of voxels beyond boundaries will be 0

	float sumWeight = 0;
	float sumFilter = 0;
	
	float that = image[POS(x, y, z)];

	for (int k = (z - kernelWidth/2); k < (z + kernelWidth/2 + 1); k++) {
		if (k < 0 || k >= depth) { continue; }
	
		for (int j = (y - kernelWidth/2); j < (y + kernelWidth/2 + 1); j++) {
			if (j < 0 || j >= height) { continue; }
		
			for (int i = (x - kernelWidth/2); i < (x + kernelWidth/2 + 1); i++) {
				if (i < 0 || i > width) { continue; }

				// Value of current pixel
				float currentVal = image[POS(i, j, k)];

				// Compute photometric distance of current pixel to the center pixel (that)
				float photometric = (1 == jbf) ?
						photomClose(jbfTemplate[POS(x, y, z)], jbfTemplate[POS(i, j, k)], sigma_photo) :
						photomClose(that, currentVal, sigma_photo);

				// Lookup geometric distance from current pixel to center pixel (that)
				//float geometric = geomClose((float3){x, y, z}, (float3){i, j, k}, sigma_spatial);
				float geometric = geometricKernel[i - (x - kernelWidth/2) + kernelWidth * (j - (y - kernelWidth/2)) + kernelWidth * kernelWidth * (k - (z - kernelWidth/2))];

				// Multiply both distances as weight factor
				float currentWeight = geometric * photometric;

				sumWeight += currentWeight;
				sumFilter += currentWeight * currentVal;
			}
		}
	}

	if (0 == sumWeight) {
		return that;
	}
	

	return (float) (sumFilter / sumWeight);
}


float geomClose(float3 point1, float3 point2, float sigma_spatial){
	float square = distance(point1,point2)/sigma_spatial;
	return exp(-.5f*(square*square));
}

float photomClose(float one, float two, float sigma_photo){
	float square = (one-two)/sigma_photo;
	return exp(-.5f*(square*square));
}
