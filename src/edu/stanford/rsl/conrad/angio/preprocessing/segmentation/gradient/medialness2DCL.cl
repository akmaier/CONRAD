
#define SAMPLE_SIZE 20

typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void medialnessKernel(
		__global TvoxelValue* medialnessResponse,
		// the volume origin
		__constant Tcoord_dev* gVolumeSize,
		// the voxel size
		__constant Tcoord_dev* gVoxelElementSize,
		// the positive line samples
		__constant Tcoord_dev* gSamplesPos,
		// the negative line samples
		__constant Tcoord_dev* gSamplesNeg,
		// the number of samples
		int nrSamples,
		//__read_only image3d_t gTex3D_D,
		__read_only image3d_t gTexDerivatives,
		// the number of derivative scales
		int nrDerivatives
		)
{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	int x = mad24(gidx,locSizex,lidx);
    int y = mad24(gidy,locSizey,lidy);
	
	unsigned int yStride = gVolumeSize[0];
		
	if (x >= gVolumeSize[0] || y >= gVolumeSize[1])
	{
		return;
	}
	
	float medialness = 0.0f;
	
	// Loop over 180 degrees from 0 to pi
	for(int ang = 0; ang < 8; ang++)
	{
		float alpha = M_PI_4*ang;
		float uAlphaX = cos(alpha);
		float uAlphaY = sin(alpha);
		float uAlphaOrthX = uAlphaY;
		float uAlphaOrthY = -uAlphaX;
		
		float xCoord = x + 0.5f*uAlphaX;
		float yCoord = y + 0.5f*uAlphaY;
		// loop over all scales
		for(int sc = 0; sc < nrDerivatives; sc++)
		{
			float gPosValBuffer[SAMPLE_SIZE];
			float gPosMaxBuffer[SAMPLE_SIZE];
			float normPos = 1;
			float gNegValBuffer[SAMPLE_SIZE];
			float gNegMaxBuffer[SAMPLE_SIZE];
			float normNeg = 1;
			for(int i = 0; i < nrSamples; i++)
			{
				// calculate Edge response for positive direction
				float evalXpos = xCoord + uAlphaOrthX*gSamplesPos[i];
				float evalYpos = yCoord + uAlphaOrthY*gSamplesPos[i];
				gPosValBuffer[i]  = uAlphaOrthX*read_imagef(gTexDerivatives, sampler, (float4)(evalXpos+0.5f, evalYpos+0.5f, 2*sc  , 0)).x;
				gPosValBuffer[i] += uAlphaOrthY*read_imagef(gTexDerivatives, sampler, (float4)(evalXpos+0.5f, evalYpos+0.5f, 2*sc+1, 0)).x;
				gPosValBuffer[i] *= (-1);
				normPos = max(-gPosValBuffer[i], normPos);
				
				// calculate Edge response for negative direction
				float evalXneg = xCoord + uAlphaOrthX*gSamplesNeg[i];
				float evalYneg = yCoord + uAlphaOrthY*gSamplesNeg[i];
				gNegValBuffer[i]  = uAlphaOrthX*read_imagef(gTexDerivatives, sampler, (float4)(evalXneg+0.5f, evalYneg+0.5f, 2*sc  , 0)).x;
				gNegValBuffer[i] += uAlphaOrthY*read_imagef(gTexDerivatives, sampler, (float4)(evalXneg+0.5f, evalYneg+0.5f, 2*sc+1, 0)).x;
				//gNegValBuffer[i] *= (-1);
				normNeg = max(-gNegValBuffer[i], normNeg);
				
				if(i==0)
				{
					gPosMaxBuffer[i] = 0.0f;
					gNegMaxBuffer[i] = 0.0f;
				}else{
					gPosMaxBuffer[i] = max(gPosValBuffer[i],gPosMaxBuffer[i-1]);
					gNegMaxBuffer[i] = max(gNegValBuffer[i],gNegMaxBuffer[i-1]);
				}				
			}
			// now we can calculate the medialness at this scale
			for(int i = 0; i < nrSamples; i++)
			{
				float medCurrent = 0.0f;
				medCurrent += max(-gPosValBuffer[i] - gPosMaxBuffer[i], 0.0f) / normPos;
				medCurrent += max(-gNegValBuffer[i] - gNegMaxBuffer[i], 0.0f) / normNeg;
				medCurrent /= 2.0f;
				medialness = max(medCurrent, medialness);
			}		
		}			
	}
		
	unsigned long idx = y*yStride + x;		

	medialnessResponse[idx] = medialness;
	
	return;
}



/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
