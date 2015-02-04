	
	constant float oneOverSix = 1.0f / 6.0f;
	constant float twoOverThree = 2.0f / 3.0f;
	
	
		
	private void getWeights(float t, float* revan){
		float index = floor(t);
		float alpha = t - index;
		float oneAlpha = 1.0f - alpha;
		float oneAlphaSquare = oneAlpha * oneAlpha;
		float alphaSquare = alpha * alpha;
		
		revan[0] = oneOverSix * oneAlphaSquare * oneAlpha;
		revan[1] = twoOverThree - 0.5f * alphaSquare *(2.0f-alpha);
		revan[2] = twoOverThree - 0.5f * oneAlphaSquare * (2.0f-oneAlpha);
		revan[3] = oneOverSix * alphaSquare * alpha;
		
	}
	
	private float4 getWeights4(float t){
		float4 revan;
		float index = floor(t);
		float alpha = t - index;
		float oneAlpha = 1.0f - alpha;
		float oneAlphaSquare = oneAlpha * oneAlpha;
		float alphaSquare = alpha * alpha;
		
		revan.x = oneOverSix * oneAlphaSquare * oneAlpha;
		revan.y = twoOverThree - 0.5f * alphaSquare *(2.0f-alpha);
		revan.z = twoOverThree - 0.5f * oneAlphaSquare * (2.0f-oneAlpha);
		revan.w = oneOverSix * alphaSquare * alpha;
		return revan;
	}
	
	kernel void evaluate(global const float* controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnots, int numControlPoints, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internal = ((samplingValues[iGID] * numKnots) -3.0f);
		float step = internal - floor(internal);
		float weights [4]; 
		getWeights(step, weights);

		// controlPoint coordinate: (i*vPoints)+j)
		for (int j = 0; j < 3; j++)
					outputBuffer[(iGID*3)+(j)] = 0;
		
		for (int i=0; i < 4; i++){
			int loc = (internal+i < 0)? 0: (internal+i>=numControlPoints)? numControlPoints-1 : (int) (internal+i);
			for (int j = 0; j < 3; j++){
				outputBuffer[(iGID*3)+(j)] += weights[i] * controlPoints[(loc*3)+j];
			} 
		}
	}
	
	kernel void evaluate2D(global const float* controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*2)] * numKnotsU) -3.0f);
		float internalV = ((samplingValues[(iGID*2)+1] * numKnotsV) -3.0f);
		float stepU = internalU - floor(internalU);
		float stepV = internalV - floor(internalV);
		int numControlPointsU = numKnotsU - 4;
		int numControlPointsV = numKnotsV - 4;
		
		float weightsU [4];
		getWeights(stepU, weightsU);
		
		float weightsV [4];
		getWeights(stepV, weightsV);
		
		// controlPoint coordinate: (i*vPoints)+j)
		for (int j = 0; j < 3; j++){
			outputBuffer[(iGID*3)+j] = 0;
		}
					
		for (int k=0; k < 4; k++){
			int locU = (internalU+k < 0)? 0: (internalU+k>=numControlPointsU)? numControlPointsU-1 : (int) (internalU+k);
			for (int i=0; i < 4; i++){	
				int locV = (internalV+i < 0)? 0: (internalV+i>=numControlPointsV)? numControlPointsV-1 : (int) (internalV+i);
				for (int j = 0; j < 3; j++) {
					outputBuffer[(iGID*3)+j] += weightsV[i] * weightsU[k] * controlPoints[(((locU*numControlPointsV)+locV)*3)+j];
					//outputBuffer[(iGID*3)+j] = weightsU[k];// weightsV[i]* controlPoints[(((locU*numControlPointsV)+locV)*3)+j];
				}
			}
		}
	}
	
	kernel void evaluate3D(global const float* controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int timeSplines, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*3)] * numKnotsU) -3.0f);
		float internalV = ((samplingValues[(iGID*3)+1] * numKnotsV) -3.0f);
		float internalTime = ((samplingValues[(iGID*3)+2] * timeSplines) -3.0f);
		float stepU = internalU - floor(internalU);
		float stepV = internalV - floor(internalV);
		float stepT = internalTime - floor(internalTime);
		int numControlPointsU = numKnotsU - 4;
		int numControlPointsV = numKnotsV - 4;
		int numControlPointsT = timeSplines - 4;
		
		float4 weightsU = getWeights4(stepU);
		float4 weightsV = getWeights4(stepV);
		float4 weightsT = getWeights4(stepT);
		
		// controlPoint coordinate: ((i)*vPoints)+j)+(t*vPoints*uPoints)
		outputBuffer[(iGID*3)] = 0.0f;
		outputBuffer[(iGID*3)+1] = 0.0f;
		outputBuffer[(iGID*3)+2] = 0.0f;
		for (int t = 0; t < 4; t++){
			float tweight = weightsT.x;
			if (t == 1) tweight = weightsT.y;
			if (t == 2) tweight = weightsT.z;
			if (t == 3) tweight = weightsT.w; 
			int locT = (internalTime+t < 0)? 0: (internalTime+t>=numControlPointsT)? numControlPointsT-1 : (int) (internalTime+t);
			for (int k=0; k < 4; k++){
				float uweight = weightsU.x;
				if (k == 1) uweight = weightsU.y;
				if (k == 2) uweight = weightsU.z;
				if (k == 3) uweight = weightsU.w; 
				int locU = (internalU+k < 0)? 0: (internalU+k>=numControlPointsU)? numControlPointsU-1 : (int) (internalU+k);
				for (int i=0; i < 4; i++){
					float vweight = weightsV.x;
					if (i == 1) vweight = weightsV.y;
					if (i == 2) vweight = weightsV.z;
					if (i == 3) vweight = weightsV.w; 			
					int locV = (internalV+i < 0)? 0: (internalV+i>=numControlPointsV)? numControlPointsV-1 : (int) (internalV+i);
					int coord = (((locU*numControlPointsV)+locV+(locT*numControlPointsV*numControlPointsU))*3);
					
					float weight = tweight * uweight * vweight;
					outputBuffer[(iGID*3)] += weight * controlPoints[coord];
					outputBuffer[(iGID*3)+1] += weight * controlPoints[coord+1];
					outputBuffer[(iGID*3)+2] += weight * controlPoints[coord+2];
				}
			}
		}
	}
	
		

	constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	constant sampler_t interpSample = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	
	kernel void evaluate3DTexture(read_only image3d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int timeSplines, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*3)] * numKnotsU) -3.0f);
		float internalV = ((samplingValues[(iGID*3)+1] * numKnotsV) -3.0f);
		float internalTime = ((samplingValues[(iGID*3)+2] * timeSplines) -3.0f);
		float stepU = internalU - floor(internalU);
		float stepV = internalV - floor(internalV);
		float stepT = internalTime - floor(internalTime);
		int numControlPointsU = numKnotsU - 4;
		int numControlPointsV = numKnotsV - 4;
		int numControlPointsT = timeSplines - 4;
		
		float4 weightsU = getWeights4(stepU);
		float4 weightsV = getWeights4(stepV);
		float4 weightsT = getWeights4(stepT);
		
		// controlPoint coordinate: ((i)*vPoints)+j)+(t*vPoints*uPoints)
		outputBuffer[(iGID*3)] = 0.0f;
		outputBuffer[(iGID*3)+1] = 0.0f;
		outputBuffer[(iGID*3)+2] = 0.0f;
		for (int t = 0; t < 4; t++){
			float tweight = weightsT.x;
			if (t == 1) tweight = weightsT.y;
			if (t == 2) tweight = weightsT.z;
			if (t == 3) tweight = weightsT.w; 
			int locT = (internalTime+t < 0)? 0: (internalTime+t>=numControlPointsT)? numControlPointsT-1 : (int) (internalTime+t);
			for (int k=0; k < 4; k++){
				float uweight = weightsU.x;
				if (k == 1) uweight = weightsU.y;
				if (k == 2) uweight = weightsU.z;
				if (k == 3) uweight = weightsU.w; 
				int locU = (internalU+k < 0)? 0: (internalU+k>=numControlPointsU)? numControlPointsU-1 : (int) (internalU+k);
				for (int i=0; i < 4; i++){
					float vweight = weightsV.x;
					if (i == 1) vweight = weightsV.y;
					if (i == 2) vweight = weightsV.z;
					if (i == 3) vweight = weightsV.w; 			
					int locV = (internalV+i < 0)? 0: (internalV+i>=numControlPointsV)? numControlPointsV-1 : (int) (internalV+i);
					//int coord = (((locU*numControlPointsV)+locV+(locT*numControlPointsV*numControlPointsU))*3);
					float4 cp = read_imagef(controlPoints, sampler, (int4){locV,locU,locT,0});
					float weight = tweight * uweight * vweight;
					outputBuffer[(iGID*3)] += weight * cp.x;
					outputBuffer[(iGID*3)+1] += weight * cp.y;
					outputBuffer[(iGID*3)+2] += weight * cp.z;
				}
			}
		}
	}
	
	
	kernel void evaluate3DTextureIntermediate(read_only image3d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int timeSplines, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*3)] * numKnotsU) -2.0f);
		float internalV = ((samplingValues[(iGID*3)+1] * numKnotsV) -2.0f);
		float internalT = ((samplingValues[(iGID*3)+2] * timeSplines) -3.0f);
		
		int numControlPointsT = timeSplines - 4;
		
		float4 coordGrid = (float4){internalV, internalU, internalT, 0.0f};
		float4 index = floor (coordGrid);
		float4 fraction = coordGrid - index;
		
		
		// inline formulation (error 3.2E-3)
		float4 oneFrac= 1.0f-fraction;
		float4 oneFrac2 = oneFrac * oneFrac;
		float4 fraction2 = fraction * fraction;
		
		float4 w0 =1.0f/6.0f * oneFrac2 * oneFrac;
		float4 w1 =2.0f/3.0f - 0.5f * fraction2 *(2.0f-fraction);
		float4 w2 =2.0f/3.0f - 0.5f * oneFrac2 * (2.0f-oneFrac);
		float4 w3 =1.0f/6.0f * fraction2 * fraction;
		
		
		float4 g0 = w0 + w1;
		float4 g1 = w2 + w3;
		float4 h0 = (w1 / g0) -0.5f + index;
		float4 h1 = (w3 / g1) + 1.5f + index;
		
		// controlPoint coordinate: (i*vPoints)+j)
		for (int j = 0; j < 3; j++)
					outputBuffer[(iGID*3)+(j)] = 0;
		
		for (int t = 0; t < 4; t++){
			float tweight = w0.z;
			if (t == 1) tweight = w1.z;
			if (t == 2) tweight = w2.z;
			if (t == 3) tweight = w3.z; 
			int locT = (internalT+t < 0)? 0: (internalT+t>=numControlPointsT)? numControlPointsT-1 : (int) (internalT+t);
		
			float4 tex00 = read_imagef(controlPoints, interpSample, (float4){h0.x, h0.y, locT+0.5f, 0.0f});
			float4 tex10 = read_imagef(controlPoints, interpSample, (float4){h1.x, h0.y, locT+0.5f, 0.0f});
			float4 tex01 = read_imagef(controlPoints, interpSample, (float4){h0.x, h1.y, locT+0.5f, 0.0f});
			float4 tex11 = read_imagef(controlPoints, interpSample, (float4){h1.x, h1.y, locT+0.5f, 0.0f});
		
			tex00 = mix (tex00,tex01,g1.y);
			tex10 = mix (tex10,tex11,g1.y);
		
			float4 cp = mix (tex00, tex10, g1.x);
		
		
			outputBuffer[(iGID*3)+(0)] += tweight * cp.x; 
			outputBuffer[(iGID*3)+(1)] += tweight * cp.y;
			outputBuffer[(iGID*3)+(2)] += tweight * cp.z;
		}
		
	}
	
	kernel void evaluate3DTextureInterp(read_only image3d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int timeSplines, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*3)] * numKnotsU) -2.0f);
		float internalV = ((samplingValues[(iGID*3)+1] * numKnotsV) -2.0f);
		float internalT = ((samplingValues[(iGID*3)+2] * timeSplines) -2.0f);
		
		
		float4 coordGrid = (float4){internalV, internalU, internalT, 0.0f};
		float4 index = floor (coordGrid);
		float4 fraction = coordGrid - index;
		
		
		// inline formulation (error 3.2E-3)
		float4 oneFrac= 1.0f-fraction;
		float4 oneFrac2 = oneFrac * oneFrac;
		float4 fraction2 = fraction * fraction;
		
		float4 w0 =1.0f/6.0f * oneFrac2 * oneFrac;
		float4 w1 =2.0f/3.0f - 0.5f * fraction2 *(2.0f-fraction);
		float4 w2 =2.0f/3.0f - 0.5f * oneFrac2 * (2.0f-oneFrac);
		float4 w3 =1.0f/6.0f * fraction2 * fraction;
		
		
		float4 g0 = w0 + w1;
		float4 g1 = w2 + w3;
		float4 h0 = (w1 / g0) -0.5f + index;
		float4 h1 = (w3 / g1) + 1.5f + index;
		
		float4 tex000 = read_imagef(controlPoints, interpSample, (float4){h0.x, h0.y, h0.z, 0.5f});
		float4 tex100 = read_imagef(controlPoints, interpSample, (float4){h1.x, h0.y, h0.z, 0.5f});
		float4 tex010 = read_imagef(controlPoints, interpSample, (float4){h0.x, h1.y, h0.z, 0.5f});
		float4 tex110 = read_imagef(controlPoints, interpSample, (float4){h1.x, h1.y, h0.z, 0.5f});
		
		float4 tex001 = read_imagef(controlPoints, interpSample, (float4){h0.x, h0.y, h1.z, 0.5f});
		float4 tex101 = read_imagef(controlPoints, interpSample, (float4){h1.x, h0.y, h1.z, 0.5f});
		float4 tex011 = read_imagef(controlPoints, interpSample, (float4){h0.x, h1.y, h1.z, 0.5f});
		float4 tex111 = read_imagef(controlPoints, interpSample, (float4){h1.x, h1.y, h1.z, 0.5f});
		
		float4 tex0t0 = mix (tex000,tex010,g1.y);
		float4 tex1t0 = mix (tex100,tex110,g1.y);
		float4 tex0t1 = mix (tex001,tex011,g1.y);
		float4 tex1t1 = mix (tex101,tex111,g1.y);
		
		
		float4 text0 = mix (tex0t0, tex1t0, g1.x);
		float4 text1 = mix (tex0t1, tex1t1, g1.x);
		
		float4 cp = mix(text0,text1, g1.z);
		
		outputBuffer[(iGID*3)+(0)] = cp.x; 
		outputBuffer[(iGID*3)+(1)] = cp.y;
		outputBuffer[(iGID*3)+(2)] = cp.z;
		
		
	}
	
	
	kernel void evaluate2DTexture(read_only image2d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*2)] * numKnotsU) -3.0f);
		float internalV = ((samplingValues[(iGID*2)+1] * numKnotsV) -3.0f);
		float stepU = internalU - floor(internalU);
		float stepV = internalV - floor(internalV);
		int numControlPointsU = numKnotsU - 4;
		int numControlPointsV = numKnotsV - 4;
		
		float weightsU [4];
		getWeights(stepU, weightsU);
		float weightsV [4];
		getWeights(stepV, weightsV);
		
		// controlPoint coordinate: (i*vPoints)+j)
		for (int j = 0; j < 3; j++)
					outputBuffer[(iGID*3)+(j)] = 0;
		
		for (int k=0; k < 4; k++){
			int locU = (internalU+k < 0)? 0: (internalU+k>=numControlPointsU)? numControlPointsU-1 : (int) (internalU+k);
			for (int i=0; i < 4; i++){	
				int locV = (internalV+i < 0)? 0: (internalV+i>=numControlPointsV)? numControlPointsV-1 : (int) (internalV+i);
				float4 cp = read_imagef(controlPoints, sampler, (int2){locV,locU});
				outputBuffer[(iGID*3)+(0)] += weightsV[i] * weightsU[k] * cp.x; 
				outputBuffer[(iGID*3)+(1)] += weightsV[i] * weightsU[k] * cp.y;
				outputBuffer[(iGID*3)+(2)] += weightsV[i] * weightsU[k] * cp.z;
				
			}
		}
	}
	
	kernel void evaluate2DTextureInterpIntermediate(read_only image2d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
		float internalU = ((samplingValues[(iGID*2)] * numKnotsU) -3.0f);
		float internalV = ((samplingValues[(iGID*2)+1] * numKnotsV) -2.0f);
		float stepU = internalU - floor(internalU);
		float stepV = internalV - floor(internalV);
		int numControlPointsU = numKnotsU - 4;
		
		float weightsU [4];
		getWeights(stepU, weightsU);
		float weightsV [4]; 
		getWeights(stepV, weightsV);
		
		// controlPoint coordinate: (i*vPoints)+j)
		for (int j = 0; j < 3; j++)
					outputBuffer[(iGID*3)+(j)] = 0;
					
		for (int k=0; k < 4; k++){
			int locU = (internalU+k < 0)? 0: (internalU+k>=numControlPointsU)? numControlPointsU-1 : (int) (internalU+k);
			int locV = floor(internalV);// Border behavior is done by clamping of the texture.
			float g0 = weightsV[0] + weightsV[1];
			float g1 = weightsV[2] + weightsV[3];
			float h0 = (weightsV[1] / g0) -0.5f;
			float h1 = (weightsV[3] / g1) +1.5f; 
			float4 cp1 = read_imagef(controlPoints, interpSample, (float2){locV+h0,locU+0.5f});
			float4 cp2 = read_imagef(controlPoints, interpSample, (float2){locV+h1,locU+0.5f});
			float4 cp3 = mix(cp1,cp2,g1);// Equivalent to cp3 = g0 * cp1 + g1 * cp2;
			outputBuffer[(iGID*3)+(0)] += weightsU[k] * cp3.x; 
			outputBuffer[(iGID*3)+(1)] += weightsU[k] * cp3.y;
			outputBuffer[(iGID*3)+(2)] += weightsU[k] * cp3.z;
		}
        
       
				
		
	}

	kernel void evaluate2DTextureInterp(read_only image2d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
        //float internalU = samplingValues[(iGID*2)];
		//float internalV = samplingValues[(iGID*2)+1];
		float internalU = ((samplingValues[(iGID*2)] * numKnotsU) -2.0f);
		float internalV = ((samplingValues[(iGID*2)+1] * numKnotsV) -2.0f);
		
		float2 coordGrid = (float2){internalV, internalU};
		float2 index = floor (coordGrid);
		float2 fraction = coordGrid - index;
		
		float weightsX [4];
		getWeights(fraction.x, weightsX);
		float weightsY [4]; 
		getWeights(fraction.y, weightsY);
		
		float2 w0 = (float2){weightsX[0], weightsY[0]};
		float2 w1 = (float2){weightsX[1], weightsY[1]};
		float2 w2 = (float2){weightsX[2], weightsY[2]};
		float2 w3 = (float2){weightsX[3], weightsY[3]};
		
		// inline formulation (error 3.2E-3)
		//float2 oneFrac= 1.0-fraction;
		//float2 oneFrac2 = oneFrac * oneFrac;
		//float2 fraction2 = fraction * fraction;
		
		//float2 w0 =1.0/6.0 * oneFrac2 * oneFrac;
		//float2 w1 =2.0/3.0 - 0.5 * fraction2 *(2.0-fraction);
		//float2 w2 =2.0/3.0 - 0.5 * oneFrac2 * (2.0-oneFrac);
		//float2 w3 =1.0/6.0 * fraction2 * fraction;
		
		
		float2 g0 = w0 + w1;
		float2 g1 = w2 + w3;
		float2 h0 = (w1 / g0) -0.5f + index;
		float2 h1 = (w3 / g1) + 1.5f + index;
		
		float4 tex00 = read_imagef(controlPoints, interpSample, (float2){h0.x, h0.y});
		float4 tex10 = read_imagef(controlPoints, interpSample, (float2){h1.x, h0.y});
		float4 tex01 = read_imagef(controlPoints, interpSample, (float2){h0.x, h1.y});
		float4 tex11 = read_imagef(controlPoints, interpSample, (float2){h1.x, h1.y});
		
		tex00 = mix (tex00,tex01,g1.y);
		tex10 = mix (tex10,tex11,g1.y);
		
		float4 cp = mix (tex00, tex10, g1.x);
		
		outputBuffer[(iGID*3)+(0)] = cp.x; 
		outputBuffer[(iGID*3)+(1)] = cp.y;
		outputBuffer[(iGID*3)+(2)] = cp.z;
				
		
	}



	// Example function to evaluate a texture at its integer control points
	// Note that the float texture must be RGBA in order to make this work.
    kernel void readTexture(read_only image2d_t controlPoints, global float* outputBuffer, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        float4 cp = read_imagef(controlPoints, sampler, (int2){iGID,0});
		outputBuffer[(iGID*3)+(0)] = cp.x; 
		outputBuffer[(iGID*3)+(1)] = cp.y;
		outputBuffer[(iGID*3)+(2)] = cp.z;
	}
	
	// Example function to evaluate a texture at its float control points. returns the same as the above example
	// Note that the float texture must be RGBA in order to make this work.
    kernel void readTextureInterp(read_only image2d_t controlPoints, global float* outputBuffer, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        float4 cp = read_imagef(controlPoints, interpSample, (float2){iGID+0.5f,0.5f});
		outputBuffer[(iGID*3)+(0)] = cp.x; 
		outputBuffer[(iGID*3)+(1)] = cp.y;
		outputBuffer[(iGID*3)+(2)] = cp.z;
	}

	// 
	kernel void evalTexture(read_only image2d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnots, int numControlPoints, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
        float internal = ((samplingValues[iGID] * numKnots) -3.0f);
		float step = internal - floor(internal);

		float weights [4]; 
		getWeights(step, weights);
		
		for (int i=0; i < 4; i++){
			int loc = (internal+i < 0)? 0: (internal+i>=numControlPoints)? numControlPoints-1 : (int) (internal+i);
			float4 cp = read_imagef(controlPoints, sampler, (int2){loc,0});
			outputBuffer[(iGID*3)+(0)] += weights[i] * cp.x; 
			outputBuffer[(iGID*3)+(1)] += weights[i] * cp.y;
			outputBuffer[(iGID*3)+(2)] += weights[i] * cp.z;
		}
	}
	
	kernel void evalTextureInterp(read_only image2d_t controlPoints, global const float * samplingValues, global float* outputBuffer, int numKnots, int numControlPoints, int numElements) {
		// get index into global data array
        int iGID = get_global_id(0);
		// bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }
        
        float internal = ((samplingValues[iGID] * numKnots) -2.0f);
		float step = internal - floor(internal);

		
		float weights [4]; 
		getWeights(step, weights);
		
		float g0 = weights[0] + weights[1];
		float g1 = weights[2] + weights[3];
		float h0 = (weights[1] / g0) -0.5f;
		float h1 = (weights[3] / g1) +1.5f;
		
		int loc = floor(internal);// Border behavior is done by clamping of the texture. (internal < 0)? -1: (int) (internal);
		
		float4 cp1 = read_imagef(controlPoints, interpSample, (float2){loc+h0,0});
		float4 cp2 = read_imagef(controlPoints, interpSample, (float2){loc+h1,0});
		float4 cp3 = mix(cp1,cp2,g1);// Equivalent to cp3 = g0 * cp1 + g1 * cp2;
		outputBuffer[(iGID*3)+(0)] = cp3.x; 
		outputBuffer[(iGID*3)+(1)] = cp3.y;
		outputBuffer[(iGID*3)+(2)] = cp3.z;
	}

    // OpenCL Kernel Function for element by element vector addition
    kernel void ComputeWeights(global float* c, int numElements) {

        // get index into global data array
        int iGID = get_global_id(0);

        // bound check, equivalent to the limit on a 'for' loop
        if (iGID >= numElements)  {
            return;
        }

		float weights [4];
		getWeights(0.2, weights);
        // add the vector elements
        c[iGID] = weights[iGID%4];
    }
    
    /*
     * Copyright (C) 2010-2014 Andreas Maier
     * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
    */