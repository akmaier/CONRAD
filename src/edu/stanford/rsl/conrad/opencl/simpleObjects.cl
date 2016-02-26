/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

//#pragma OPENCL EXTENSION cl_khr_fp64: enable


kernel void applyTransform(global float* outputBuffer, global float* matrix, int numElements){
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
	float4 point = (float4){outputBuffer[(iGID*3)], outputBuffer[(iGID*3)+1], outputBuffer[(iGID*3)+2], 1.0f};
	float4 row1 = (float4){matrix[0],matrix[1],matrix[2],matrix[3]};
    float4 row2 = (float4){matrix[4],matrix[5],matrix[6],matrix[7]};
    float4 row3 = (float4){matrix[8],matrix[9],matrix[10],matrix[11]};
    float4 row4 = (float4){matrix[12],matrix[13],matrix[14],matrix[15]};
    float h = dot(row4,point);
    outputBuffer[(iGID*3)] = (dot(row1,point) / h);
	outputBuffer[(iGID*3)+1] = (dot(row2,point) / h);
	outputBuffer[(iGID*3)+2] = (dot(row3,point) /h);
}

kernel void evaluateCylinder(global const float* parameter, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV) {
	int numElements = numKnotsU*numKnotsV;
	

	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    //printf("Starting iGID: %d\n", iGID);
    // size(coordinateBuffer) == elementCountU * elementCountV (*3) 
    // -> assumes elementCountU == elementCountV (uniformly sampled splines)
    int gridWidth = (int)sqrt((float)numElements); 

	
	float u = (samplingValues[(iGID*2)] * numKnotsU);
	float v = (samplingValues[(iGID*2)+1] * numKnotsV);
	
	
	float angle = (float) v/(numKnotsV -1) *2*M_PI_F;

	
	outputBuffer[(iGID*3)] = (float) (((-parameter[0] + parameter[0])/2)+0.5*(parameter[0]-(-parameter[0]))*(float) cos(angle));
	outputBuffer[(iGID*3)+1] = (float) (((-parameter[1]+ parameter[1])/2)+0.5*(parameter[1]-(-parameter[1]))*sin(angle));
	outputBuffer[(iGID*3)+2] = (float) (0.5*parameter[2]-   (0.5*parameter[2]-(-0.5*parameter[2])) *  ((float) u/(numKnotsU-1)) );
	
}



kernel void evaluateSphere(global const float* parameter, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV) {
	
	//parameter[0] - parameter[2] represent surfaceOrigin x,y,z values
	//parameter[3] = radius
	
	int numElements = numKnotsU*numKnotsV;
	

	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }

	float u = (samplingValues[(iGID*2)] * numKnotsU);
	float v = (samplingValues[(iGID*2)+1] * numKnotsV);
	
	//float angle = ((float) u)/(numKnotsU-1) *2*M_PI_F;

	//float curveDistance = (float) (( (float) parameter[3]*2.0)/(numKnotsV-1));
	//float scalingCurve = (float) sqrt(parameter[3]*parameter[3] - (parameter[3]-v*curveDistance)*(parameter[3]-v*curveDistance));
	//float scalingCurve = (float) sqrt(2*((float) parameter[3])*v*curveDistance - (v*curveDistance)*(v*curveDistance)); // numerically not better than above
	//float scalingCurve = parameter[3] * cos(asin(1-v*curveDistance/parameter[3]));

	
	//outputBuffer[(iGID*3)] = (float) (parameter[0]+scalingCurve*cos(angle));
	//outputBuffer[(iGID*3)+1] = (float) (parameter[1]+scalingCurve*sin(angle));
	//outputBuffer[(iGID*3)+2] = (float) (parameter[2]-parameter[3]+v*curveDistance);
	
	
	float angle1 = (float) v/(numKnotsV) *2*M_PI_F;
	float angle2 = (float) u/(numKnotsU -1) *M_PI_F;
	
	float helper = parameter[0]*sin(angle2);
	
	outputBuffer[(iGID*3)] = (float) helper*cos(angle1);
	outputBuffer[(iGID*3)+1] = (float) helper*sin(angle1);
	outputBuffer[(iGID*3)+2] = (float) parameter[0]*cos(angle2);
	
	//outputBuffer[(iGID*3)] = 2*parameter[3]*v*curveDistance;
	//outputBuffer[(iGID*3)+1] = 1-v*curveDistance/parameter[3];
	//outputBuffer[(iGID*3)+2] = scalingCurve;

}

/************************************************/

/*
// method1:
kernel void evaluateBox(global const float* parameter, global const float* samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV){
	int numElements = numKnotsU * numKnotsV;
	
	int iGID = get_global_id(0);
	if (iGID >= (int)numElements/4){
		return;
	}
	
	float u = (samplingValues[(iGID*2)] * numKnotsU);
	float v = (samplingValues[(iGID*2)+1] * numKnotsV);
	if (v >= (float)(numKnotsV/4)){
		return;
	}
	
	float stepV1 = (float)(parameter[3] - parameter[0])/(numKnotsV/4 - 1);
	float stepV2 = (float)(parameter[4] - parameter[1])/(numKnotsV/4 - 1);
	float stepU = (float)(parameter[5] - parameter[2])/(numKnotsU - 1);
	
	int offset = (int)3*numElements/4;
	
	// edge e01,...,e09
	outputBuffer[(iGID*3)] = (float)(parameter[0] + v*stepV1);
	outputBuffer[(iGID*3)+1] = (float)parameter[1];
	outputBuffer[(iGID*3)+2] = (float)(parameter[2] + u*stepU);
	// edge e05,...,e11
	outputBuffer[(iGID*3) + offset] = (float)parameter[3];
	outputBuffer[(iGID*3) + offset + 1] = (float)(parameter[1] + v*stepV2);
	outputBuffer[(iGID*3) + offset + 2] = (float)(parameter[2] + u*stepU);
	// edge e06,...,e12
	outputBuffer[(iGID*3) + offset*2] = (float)(parameter[0] + v*stepV1);
	outputBuffer[(iGID*3) + offset*2 + 1] = (float)parameter[4];
	outputBuffer[(iGID*3) + offset*2 + 2] = (float)(parameter[2] + u*stepU);
	// edge e02,...,e10
	outputBuffer[(iGID*3) + offset*3] = (float)parameter[0];
	outputBuffer[(iGID*3) + offset*3 + 1] = (float)(parameter[1] + v*stepV2);
	outputBuffer[(iGID*3) + offset*3 + 2] = (float)(parameter[2] + u*stepU);
	
}*/

// method2:
kernel void evaluateBox(global const float* parameter, global const float* samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV){
	int numElements = numKnotsU * numKnotsV;
	
	int iGID = get_global_id(0);
	if (iGID >= numElements){
		return;
	}
	float u = (samplingValues[(iGID*2)]);// * numKnotsU);
	float v = (samplingValues[(iGID*2)+1]);// * numKnotsV);


	
	float4 firstTop;
	float4 secondTop;
	float4 firstBottom;
	float4 secondBottom;
	
	if (v <= 0.25){
		
		firstTop = (float4){parameter[0], parameter[1], parameter[5], 0};
		secondTop = (float4){parameter[3], parameter[1], parameter[5], 0};

		firstBottom = (float4){parameter[0], parameter[1], parameter[2], 0};
		secondBottom = (float4){parameter[3], parameter[1], parameter[2], 0};
	
	} else if(v <= 0.5){
		v -= 0.25;

		firstTop = (float4){parameter[3], parameter[1], parameter[5], 0};
		secondTop = (float4){parameter[3], parameter[4], parameter[5], 0};

		firstBottom = (float4){parameter[3], parameter[1], parameter[2], 0};
		secondBottom = (float4){parameter[3], parameter[4], parameter[2], 0};
	
	}else if(v <= 0.75){
		v -= 0.5;
		
		firstTop = (float4){parameter[3], parameter[4], parameter[5], 0};
		secondTop = (float4){parameter[0], parameter[4], parameter[5], 0};
		
		firstBottom = (float4){parameter[3], parameter[4], parameter[2], 0};
		secondBottom = (float4){parameter[0], parameter[4], parameter[2], 0};
		
	}else{
		v -= 0.75;

		firstTop = (float4){parameter[0], parameter[4], parameter[5], 0};
		secondTop = (float4){parameter[0], parameter[1], parameter[5], 0};

		firstBottom = (float4){parameter[0], parameter[4], parameter[2], 0};
		secondBottom = (float4){parameter[0], parameter[1], parameter[2], 0};
	
	}
		
	float4 pointTop = mix(firstTop,secondTop, (float4)(4.0*v));
	float4 pointBottom = mix(firstBottom,secondBottom, (float4)(4.0*v));
	float4 point = mix(pointTop, pointBottom, (float4)(1.0-u));
	
	
	outputBuffer[(iGID*3)] = point.x;  //(float)(pointBottom[0] * (1.0 - u) + pointTop[0] * u);
	outputBuffer[(iGID*3)+1] = point.y;  //(float)(pointBottom[1] * (1.0 - u) + pointTop[1] * u);
	outputBuffer[(iGID*3)+2] = point.z;  //(float)(pointBottom[2] * (1.0 - u) + pointTop[2] * u);
}


/**********************************************/

kernel void evaluateEllipsoid(global const float* parameter, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV) {
	int numElements = numKnotsU*numKnotsV;
	
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    
    int gridWidth = (int)sqrt((float)numElements); 

	
	float u = (samplingValues[(iGID*2)] * numKnotsU);
	float v = (samplingValues[(iGID*2)+1] * numKnotsV);
	
	
	float angle1 = (float) v/(numKnotsV) *2*M_PI_F;
	float angle2 = (float) u/(numKnotsU -1) *M_PI_F;
	
	outputBuffer[(iGID*3)] = (float) (((-parameter[0] + parameter[0])/2)+0.5*(parameter[0]-(-parameter[0]))*sin(angle2)*cos(angle1));
	outputBuffer[(iGID*3)+1] = (float) (((-parameter[1]+ parameter[1])/2)+0.5*(parameter[1]-(-parameter[1]))*sin(angle2)*sin(angle1));
	outputBuffer[(iGID*3)+2] = (float) (((-parameter[2] + parameter[2])/2)+0.5*(parameter[2]-(-parameter[2]))*cos(angle2));
	
}

/**********************************************/

kernel void evaluateCone(global const float* parameter, global const float * samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV) {
	int numElements = numKnotsU*numKnotsV;
	
	int iGID = get_global_id(0);
	// bound check, equivalent to the limit on a 'for' loop
    if (iGID >= numElements)  {
    	return;
    }
    
    int gridWidth = (int)sqrt((float)numElements); 
    
    //float u = (samplingValues[(iGID*2)]);
	//float v = (samplingValues[(iGID*2)+1]);

	float u = (samplingValues[(iGID*2)] * numKnotsU);
	float v = (samplingValues[(iGID*2)+1] * numKnotsV);
	
	float height = u*(parameter[2])/(float)(numKnotsU - 1);
	//float factor = (float) u/(numKnotsU - 1);
	float angle = (float) v/(numKnotsV) *2*M_PI_F;
	
	float coorZ = -parameter[2] + height;
	//float coorZ = mix(-parameter[2], 0, factor);
	
	outputBuffer[(iGID*3)] = (float)(coorZ*parameter[3]*cos(angle));
	outputBuffer[(iGID*3)+1] = (float)(coorZ*parameter[4]*sin(angle));
	outputBuffer[(iGID*3)+2] = coorZ;

}

/**********************************************/

kernel void evaluatePyramid(global const float* parameter, global const float* samplingValues, global float* outputBuffer, int numKnotsU, int numKnotsV){
	
	int numElements = numKnotsU*numKnotsV;
	int iGID = get_global_id(0);
	if (iGID >= numElements){
		return;
	}
	
	int gridWidth = (int)sqrt((float)numElements);
	
	float u = samplingValues[(iGID*2)];//*numKnotsU;
	float v = (samplingValues[(iGID*2 + 1)]*numKnotsV)/(numKnotsV);
	
	float factor =  u*numKnotsU/((float)(numKnotsU-1));
	float absCoorZ = -((-parameter[2] + factor*(parameter[2])))/parameter[2];
	//float absCoorZ = -(-parameter[2] + u*(parameter[2]));
	

	float4 first;
	float4 second;
	float4 point;
	
	if (v <= 0.25){
		first = (float4){-parameter[0]*parameter[3]*absCoorZ, -parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		second = (float4){parameter[0]*parameter[3]*absCoorZ, -parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		
	} else if (v <= 0.5){
		v -= 0.25;
		first = (float4){parameter[0]*parameter[3]*absCoorZ, -parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		second = (float4){parameter[0]*parameter[3]*absCoorZ, parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		
	} else if (v <= 0.75){
		v -= 0.5;
		first = (float4){parameter[0]*parameter[3]*absCoorZ, parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		second = (float4){-parameter[0]*parameter[3]*absCoorZ, parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		
	} else {
		v -= 0.75;
		first = (float4){-parameter[0]*parameter[3]*absCoorZ, parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		second = (float4){-parameter[0]*parameter[3]*absCoorZ, -parameter[1]*parameter[4]*absCoorZ, -absCoorZ, 0};
		
	}
	
	point = mix(first, second, (float4)(4.0*v*numKnotsV/(float)(numKnotsV)));

	outputBuffer[(iGID*3)] = point.x;
	outputBuffer[(iGID*3)+1] = point.y;
	outputBuffer[(iGID*3)+2] = -absCoorZ*parameter[2];

}




