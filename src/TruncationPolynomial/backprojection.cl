kernel void backproject(global float *sinogram, global float *out, int const numElementsX, int const numElementsY, int const sinogramWidth) {

	int iGIDx = get_global_id(0);
	int iGIDy = get_global_id(1);
	
	if (iGIDx >= numElementsX || iGIDy >= numElementsY) {
		return;
	}

	for (int thetaGrad = 0; thetaGrad < 180; thetaGrad++) {
		float theta = (thetaGrad)* M_PI_F/180;
		float s = ((iGIDx - 0.5f*numElementsX) * cos(theta) + (iGIDy - 0.5f*numElementsY) * sin(theta)) + 0.5f * sinogramWidth;
		int sFloor = floor(s);
		int sCeil = ceil(s);
		float value = (s -  floor(s)) * sinogram[thetaGrad*sinogramWidth + sCeil] + (ceil(s) - s) * sinogram[thetaGrad*sinogramWidth + sFloor];
		out[iGIDy*numElementsX + iGIDx] += value;
	}

}