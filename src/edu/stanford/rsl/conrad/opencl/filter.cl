/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

float geomClose(float2 point1, float2 point2, float sigma_spatial){
	float square = distance(point1,point2)/sigma_spatial;
	return exp(-.5f*(square*square));
}

float photomClose(float one, float two, float sigma_photo){
	float square = (one-two)/sigma_photo;
	return exp(-.5f*(square*square));
}

float gauss(float value, float sigma){
	float square = (value)/sigma;
	return exp(-.5f*(square*square));
}

float stddev2D (global float * image, int x, int y, float * kern, int halfWidth, int halfHeight, int iwidth, int iheight, float mean){
	float sumWeight = 0;
	float sumFilter = 0;
	int xstep = halfWidth*2+1;
	for (int j=0; j<halfHeight*2+1;j++){
		for (int i=0; i<xstep;i++){
			int nx =x-halfWidth+i;
			int ny =y-halfHeight+j;
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					sumWeight+=kern[(j*xstep)+i];
					float value =image[nx+(ny*iwidth)]-mean;
					sumFilter+=kern[(j*xstep)+i] * value * value;
				}
			}
		}
	}
	if (fabs(sumWeight) < 0.0001f) sumWeight = 1.0f;
	return sqrt(sumFilter)/sumWeight;
}

float convolution2D (global float * image, int x, int y, float * kern, int halfWidth, int halfHeight, int iwidth, int iheight){
	float sumWeight = 0;
	float sumFilter = 0;
	int xstep = halfWidth*2+1;
	for (int j=0; j<halfHeight*2+1;j++){
		for (int i=0; i<xstep;i++){
			int nx =x-halfWidth+i;
			int ny =y-halfHeight+j;
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					sumWeight+=kern[(j*xstep)+i];
					sumFilter+=kern[(j*xstep)+i] * image[nx+(ny*iwidth)];
				}
			}
		}
	}
	if (fabs(sumWeight) < 0.0001f) sumWeight = 1.0f;
	return sumFilter/sumWeight;
}

kernel void bilateralFilter(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_spatial, float sigma_photo){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	int width = (int)(sigma_spatial*2);

	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float two = template[nx+(ny*iwidth)];
					float photom = photomClose(one, two, sigma_photo);
					float weight = photom * geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	float value = sumFilter/sumWeight;
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}

kernel void bilateralFilterTwoSigma(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_spatial_large, float sigma_spatial_small, float sigma_photo, float i0){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	

	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float gaussKernel [9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

	float gaussValue = (convolution2D(template, x,y, gaussKernel, 1,1, iwidth, iheight)/i0);
	float sigma_spatial =(gaussValue *sigma_spatial_small) + ((1-gaussValue)*sigma_spatial_large);
	
	int width = (int)(sigma_spatial*2);
	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float two = template[nx+(ny*iwidth)];
					float photom = photomClose(one, two, sigma_photo);
					float weight = photom * geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	float value = sumFilter/sumWeight;
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}



kernel void bilateralFilterRangeAdaptive(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_geom, float sigma_photo, float i0, float i1){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	

	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float gaussKernel [49] = {1, 1, 1, 1,1, 1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1, 1, 1, 1, 1,1, 1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1};
	
	float meanTemplate = (convolution2D(template, x,y, gaussKernel, 3,3, iwidth, iheight));
	//float sigmaTemplate = stddev2D(template, x,y, gaussKernel, 3,3, iwidth, iheight, meanTemplate)/i0;
	
	//float meanImage = (convolution2D(image, x,y, gaussKernel, 3,3, iwidth, iheight));
	//float sigmaImage = stddev2D(image, x,y, gaussKernel, 3,3, iwidth, iheight, meanImage)/i1;
	
	//float sigma_spatial = (sigma_geom* sigmaImage) / (sigmaTemplate);
	
	float sigma_spatial = sigma_geom;
	
	int width = (int)(sigma_spatial*2);
	if (width > 16) width = 16;
	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float two = template[nx+(ny*iwidth)];
					// Note sigma_photo = (1 - I_2 / I_1);
					
					float photom = photomClose(one, two, meanTemplate * sigma_photo);
					float weight = photom * geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	float value = sumFilter/sumWeight;
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}

kernel void bilateralFilterNoiseAdaptivePathLength(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_geom, float sigma_photo, float i0, float i1){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	

	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float gaussKernel [49] = {1, 1, 1, 1,1, 1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,
			1, 1, 1, 1,1, 1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1};

	float meanTemplate = (convolution2D(template, x,y, gaussKernel, 3,3, iwidth, iheight));
	//float sigmaTemplate = stddev2D(template, x,y, gaussKernel, 3,3, iwidth, iheight, meanTemplate)/i0;
	
	float meanImage = (convolution2D(image, x,y, gaussKernel, 3,3, iwidth, iheight));
	
	float sigma_path = (i1) / (meanImage);
	
	//float sigmaImage = stddev2D(image, x,y, gaussKernel, 3,3, iwidth, iheight, meanImage)/i1;
	
	//float sigma_adaptive = sigma_geom * (sigmaImage / sigmaTemplate);
	 
	//float sigma_spatial = sigma_adaptive * sigma_path;
	float sigma_spatial = sigma_geom * sigma_path;
	
	
	int width = (int)(sigma_spatial*2);
	if (width > 16) width = 16;
	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float two = template[nx+(ny*iwidth)];
					// Note sigma_photo = (1 - I_2 / I_1);
					
					float photom = photomClose(one, two, meanTemplate * sigma_photo);
					float weight = photom * geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	float value = sumFilter/sumWeight;
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}


kernel void bilateralFilterPathLengthAdaptive(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_geom, float sigma_photo, float i0, float i1){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	
	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float gaussKernel [25] = {1, 1, 1, 1,1, 1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1,1, 1, 1, 1,1};
	
	float meanImage = (convolution2D(image, x,y, gaussKernel, 2,2, iwidth, iheight));
	
	float sigma_path = (i1) / (meanImage);
	
	float sigma_spatial = sigma_geom * sigma_path;
	
	
	int width = (int)(sigma_spatial*2);
	if (width > 16) width = 16;
	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float two = template[nx+(ny*iwidth)];
					float photom = photomClose(one, two, sigma_photo);
					float weight = photom * geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	float value = sumFilter/sumWeight;
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}


kernel void laplacianFilter(global float * template, global float * image, global float * out, int iwidth, int iheight, float sigma_spatial, float sigma_photo){
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);

	int gloSizex = get_global_size(0);
	int gloSizey = get_global_size(1);

	unsigned int x = gidx*locSizex+lidx;
	unsigned int y = gidy*locSizey+lidy;

	if (x >= iwidth || y >= iheight)
		return;

	int width = (int)(sigma_spatial*2);

	float sumWeight = 0.f;
	float sumFilter = 0.f;

	float one = template[x+(y*iwidth)];
	float2 current = (float2){x,y};

	float laplace [9] = {-1, 2, -1,-2, 4, -2, -1, 2, -1};

	float lpValue = convolution2D(template, x,y, laplace, 1,1, iwidth, iheight);
	float gaussian = gauss(lpValue, sigma_photo);
	//gaussian = 0;
	
	for (int i=0; i<width*2+1;i++){
		for (int j=0; j<width*2+1;j++){
			int nx =x-width+i;
			int ny =y-width+j;
			float2 n = (float2){nx,ny};
			if(nx >= 0 && nx < iwidth){
				if (ny >= 0 && ny < iheight){
					float weight = geomClose(n,current,sigma_spatial);
					float value = image[nx+(ny*iwidth)];
					if(!isnan(weight) 
							&& !isinf(weight)
							&& !isnan(value)
							&& !isinf(value)){
						sumWeight+=weight;
						sumFilter+=weight*value;
					}
				}
			}
		}
	}
	// interpolate according to edge value
	float value = (gaussian*sumFilter/sumWeight)+((1.0f-gaussian)*image[x+(y*iwidth)]);
	if (isnan(value)) value = image[x+(y*iwidth)];
	out[x+(y*iwidth)]=value;

}