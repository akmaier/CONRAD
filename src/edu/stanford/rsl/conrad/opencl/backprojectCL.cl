typedef float TvoxelValue;
typedef float Tcoord_dev;
typedef float TdetValue;

// Volume texture
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

//__constant__ int gVolStride[2];  

// System geometry related
//__constant__ Tcoord_dev gProjMatrix[12];  


/* --------------------------------------------------------------------------
 *
 *
 *    Voxel-based BP algorithm implementation in OpenCL kernel programming
 *
 *
 * -------------------------------------------------------------------------- */

__kernel void backprojectKernel(
		__global TvoxelValue* pVolume,
		int sizeX,
		int sizeY,
		int recoSizeZ,
		int offset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__constant Tcoord_dev* gProjMatrix,
		float projMultiplier)

{
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
	
	if (x >= sizeX || y >= sizeY)
		return;
		
	unsigned int zStride = sizeX*sizeY;
	unsigned int yStride = sizeX;
		
	//int idx = vdx*projStride+udx;
    //int z = (int) blockIdx.z;

	// x and y will be constant in this thread;
    float xcoord = (x * voxelSpacingX) - offsetX;
    float ycoord = (y * voxelSpacingY) - offsetY;

	// precompute everything except z from the matrix multiplication
	float precomputeR = gProjMatrix[3]  + ycoord * gProjMatrix[1] + xcoord * gProjMatrix[0];
	float precomputeS = gProjMatrix[7]  + ycoord * gProjMatrix[5] + xcoord * gProjMatrix[4];
    float precomputeT = gProjMatrix[11] + ycoord * gProjMatrix[9] + xcoord * gProjMatrix[8];

	for (unsigned int z = 0; z < recoSizeZ; z++){

		float zcoord = ((z) * voxelSpacingZ) - offsetZ;

		// compute homogeneous coordinates
		float r = zcoord * gProjMatrix[2]  + precomputeR;
		float s = zcoord * gProjMatrix[6]  + precomputeS;
		float t = zcoord * gProjMatrix[10] + precomputeT;

		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;

		//float proj_val = tex2D(gTex2D, fu, fv);   // <--- visible error!
		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + offset, fv + 0.5f)).x;
		//float proj_val = tex2D(gTex2D, fu+0.5f + offset, fv+0.5f);   // <--- correct for non-padded detector
		//float proj_val = tex2D(gTex2D, fu+1.5, fv+1.5);

		// compute volume index for x,y,z
		unsigned long idx = z*zStride + y*yStride + x;
    

		// Distance weighting by 1/t^2
		pVolume[idx] += proj_val * denom * denom * projMultiplier;
		//pVolume[idx] += proj_val;
	}
	
    return;
}

/* --------------------------------------------------------------------------
 *
 *
 *    Voxel-based BP algorithm implementation in OpenCL kernel programming
 *    WITH DEFORMATION!
 *
 *
 * -------------------------------------------------------------------------- */
float4 mls_deformation(float x, float y, float z, global float* p, global float* q, int noPoints)
{
	
	// q* and p*
	float qStar0 = 0;
	float qStar1 = 0;
	float qStar2 = 0;
	float pStar0 = 0;
	float pStar1 = 0;
	float pStar2 = 0;
	float wSum = 0;
	
	for (unsigned int idx = 0; idx < noPoints; idx++) {
		float w = 1/((p[idx*3] - x) * (p[idx*3] - x) + (p[idx*3+1] - y) * (p[idx*3+1] - y) + (p[idx*3+2] - z) * (p[idx*3+2] - z));
		wSum += w;
		qStar0 += q[idx*3] * w;
		qStar1 += q[idx*3+1] * w;
		qStar2 += q[idx*3+2] * w;
		pStar0 += p[idx*3] * w;
		pStar1 += p[idx*3+1] * w;
		pStar2 += p[idx*3+2] * w;
	}
	qStar0 /= wSum;
	qStar1 /= wSum;
	qStar2 /= wSum;
	pStar0 /= wSum;
	pStar1 /= wSum;
	pStar2 /= wSum;
		
	
	// PQ'
	float PQtrans0 = 0;
	float PQtrans1 = 0;
	float PQtrans2 = 0;
	float PQtrans3 = 0;
	float PQtrans4 = 0;
	float PQtrans5 = 0;
	float PQtrans6 = 0;
	float PQtrans7 = 0;
	float PQtrans8 = 0;
	for (unsigned int idx = 0; idx < noPoints; idx++) {
		float w = 1/((p[idx*3] - x) * (p[idx*3] - x) + (p[idx*3+1] - y) * (p[idx*3+1] - y) + (p[idx*3+2] - z) * (p[idx*3+2] - z));
		
		float p_min_pStar0 = p[idx*3] - pStar0;
		float p_min_pStar1 = p[idx*3+1] - pStar1;
		float p_min_pStar2 = p[idx*3+2] - pStar2;
		
		float q_min_qStar0 = q[idx*3] - qStar0;
		float q_min_qStar1 = q[idx*3+1] - qStar1;
		float q_min_qStar2 = q[idx*3+2] - qStar2;
		
		PQtrans0 += w*p_min_pStar0*q_min_qStar0;
		PQtrans3 += w*p_min_pStar0*q_min_qStar1;
		PQtrans6 += w*p_min_pStar0*q_min_qStar2;
		
		PQtrans1 += w*p_min_pStar1*q_min_qStar0;
		PQtrans4 += w*p_min_pStar1*q_min_qStar1;
		PQtrans7 += w*p_min_pStar1*q_min_qStar2;
		
		PQtrans2 += w*p_min_pStar2*q_min_qStar0;
		PQtrans5 += w*p_min_pStar2*q_min_qStar1;
		PQtrans8 += w*p_min_pStar2*q_min_qStar2;
		
	}

	// svd
	int n = 3;

	double tol = 0.000000000000001f;		
	int loopmax = 100 * n;
	int loopcount = 0;

	float U0 = 1;
	float U1 = 0;
	float U2 = 0;
	float U3 = 0;
	float U4 = 1;
	float U5 = 0;
	float U6 = 0;
	float U7 = 0;
	float U8 = 1;

	float V0 = 1;
	float V1 = 0;
	float V2 = 0;
	float V3 = 0;
	float V4 = 1;
	float V5 = 0;
	float V6 = 0;
	float V7 = 0;
	float V8 = 1;

	float S0 = PQtrans0;
	float S1 = PQtrans3;
	float S2 = PQtrans6;
	float S3 = PQtrans1;
	float S4 = PQtrans4;
	float S5 = PQtrans7;
	float S6 = PQtrans2;
	float S7 = PQtrans5;
	float S8 = PQtrans8;

	float err = 10000;

	while (err > tol && loopcount < loopmax) {

		// qr //////////////////////////
		int m = 3;

		float R0 = S0;
		float R1 = S3;
		float R2 = S6;
		float R3 = S1;
		float R4 = S4;
		float R5 = S7;
		float R6 = S2;
		float R7 = S5;
		float R8 = S8;

		float Q0 = 1;
		float Q1 = 0;
		float Q2 = 0;
		float Q3 = 0;
		float Q4 = 1;
		float Q5 = 0;
		float Q6 = 0;
		float Q7 = 0;
		float Q8 = 1;

		// k = 0;
		float x0 = R0;
		float x1 = R1;
		float x2 = R2;

		float norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);
		float g = -sign(x0) * norm_x;
		x0 = x0 - g;

		// Orthogonal transformation matrix that eliminates one element
		// below the diagonal of the matrix it is post-multiplying:
		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);

		if (norm_x != 0) {

			x0 = x0/norm_x;
			x1 = x1/norm_x;
			x2 = x2/norm_x;

			float u1 = 2 * (R0*x0 + R1*x1 + R2*x2);
			float u2 = 2 * (R3*x0 + R4*x1 + R5*x2);
			float u3 = 2 * (R6*x0 + R7*x1 + R8*x2);

			R0 = R0 - x0*u1;
			R3 = R3 - x0*u2;
			R6 = R6 - x0*u3;
			R1 = R1 - x1*u1;
			R4 = R4 - x1*u2;
			R7 = R7 - x1*u3;
			R2 = R2 - x2*u1;
			R5 = R5 - x2*u2;
			R8 = R8 - x2*u3;

			float Q0tmp = Q0 - 2*(Q0*x0*x0 + Q3*x1*x0 + Q6*x2*x0);
			float Q3tmp = Q3 - 2*(Q0*x0*x1 + Q3*x1*x1 + Q6*x2*x1);
			float Q6tmp = Q6 - 2*(Q0*x0*x2 + Q3*x1*x2 + Q6*x2*x2);
			float Q1tmp = Q1 - 2*(Q1*x0*x0 + Q4*x1*x0 + Q7*x2*x0);
			float Q4tmp = Q4 - 2*(Q1*x0*x1 + Q4*x1*x1 + Q7*x2*x1);
			float Q7tmp = Q7 - 2*(Q1*x0*x2 + Q4*x1*x2 + Q7*x2*x2);
			float Q2tmp = Q2 - 2*(Q2*x0*x0 + Q5*x1*x0 + Q8*x2*x0);
			float Q5tmp = Q5 - 2*(Q2*x0*x1 + Q5*x1*x1 + Q8*x2*x1);
			float Q8tmp = Q8 - 2*(Q2*x0*x2 + Q5*x1*x2 + Q8*x2*x2);

			Q0 = Q0tmp;
			Q1 = Q1tmp;
			Q2 = Q2tmp;
			Q3 = Q3tmp;
			Q4 = Q4tmp;
			Q5 = Q5tmp;
			Q6 = Q6tmp;
			Q7 = Q7tmp;
			Q8 = Q8tmp;
		}

		// k = 1;
		x0 = 0;
		x1 = R4;
		x2 = R5;

		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);
		g = -sign(x1) * norm_x;
		x1 = x1 - g;

		// Orthogonal transformation matrix that eliminates one element
		// below the diagonal of the matrix it is post-multiplying:
		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);

		if (norm_x != 0) {

			x0 = x0/norm_x;
			x1 = x1/norm_x;
			x2 = x2/norm_x;

			float u1 = 2 * (R0*x0 + R1*x1 + R2*x2);
			float u2 = 2 * (R3*x0 + R4*x1 + R5*x2);
			float u3 = 2 * (R6*x0 + R7*x1 + R8*x2);

			R0 = R0 - x0*u1;
			R3 = R3 - x0*u2;
			R6 = R6 - x0*u3;
			R1 = R1 - x1*u1;
			R4 = R4 - x1*u2;
			R7 = R7 - x1*u3;
			R2 = R2 - x2*u1;
			R5 = R5 - x2*u2;
			R8 = R8 - x2*u3;

			float Q0tmp = Q0 - 2*(Q0*x0*x0 + Q3*x1*x0 + Q6*x2*x0);
			float Q3tmp = Q3 - 2*(Q0*x0*x1 + Q3*x1*x1 + Q6*x2*x1);
			float Q6tmp = Q6 - 2*(Q0*x0*x2 + Q3*x1*x2 + Q6*x2*x2);
			float Q1tmp = Q1 - 2*(Q1*x0*x0 + Q4*x1*x0 + Q7*x2*x0);
			float Q4tmp = Q4 - 2*(Q1*x0*x1 + Q4*x1*x1 + Q7*x2*x1);
			float Q7tmp = Q7 - 2*(Q1*x0*x2 + Q4*x1*x2 + Q7*x2*x2);
			float Q2tmp = Q2 - 2*(Q2*x0*x0 + Q5*x1*x0 + Q8*x2*x0);
			float Q5tmp = Q5 - 2*(Q2*x0*x1 + Q5*x1*x1 + Q8*x2*x1);
			float Q8tmp = Q8 - 2*(Q2*x0*x2 + Q5*x1*x2 + Q8*x2*x2);

			Q0 = Q0tmp;
			Q1 = Q1tmp;
			Q2 = Q2tmp;
			Q3 = Q3tmp;
			Q4 = Q4tmp;
			Q5 = Q5tmp;
			Q6 = Q6tmp;
			Q7 = Q7tmp;
			Q8 = Q8tmp;
		}

		//////////////////////////////////////

		S0 = R0;
		S1 = R1;
		S2 = R2;
		S3 = R3;
		S4 = R4;
		S5 = R5;
		S6 = R6;
		S7 = R7;
		S8 = R8;

		float U0tmp = U0 * Q0 + U3 * Q1 + U6 * Q2;
		float U1tmp = U1 * Q0 + U4 * Q1 + U7 * Q2;
		float U2tmp = U2 * Q0 + U5 * Q1 + U8 * Q2;
		float U3tmp = U0 * Q3 + U3 * Q4 + U6 * Q5;
		float U4tmp = U1 * Q3 + U4 * Q4 + U7 * Q5;
		float U5tmp = U2 * Q3 + U5 * Q4 + U8 * Q5;
		float U6tmp = U0 * Q6 + U3 * Q7 + U6 * Q8;
		float U7tmp = U1 * Q6 + U4 * Q7 + U7 * Q8;
		float U8tmp = U2 * Q6 + U5 * Q7 + U8 * Q8;
		U0 = U0tmp;
		U1 = U1tmp;
		U2 = U2tmp;
		U3 = U3tmp;
		U4 = U4tmp;
		U5 = U5tmp;
		U6 = U6tmp;
		U7 = U7tmp;
		U8 = U8tmp;

		// qr //////////////////////////
		R0 = S0;
		R1 = S3;
		R2 = S6;
		R3 = S1;
		R4 = S4;
		R5 = S7;
		R6 = S2;
		R7 = S5;
		R8 = S8;

		Q0 = 1;
		Q1 = 0;
		Q2 = 0;
		Q3 = 0;
		Q4 = 1;
		Q5 = 0;
		Q6 = 0;
		Q7 = 0;
		Q8 = 1;

		// k = 0;
		x0 = R0;
		x1 = R1;
		x2 = R2;

		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);
		g = -sign(x0) * norm_x;
		x0 = x0 - g;

		// Orthogonal transformation matrix that eliminates one element
		// below the diagonal of the matrix it is post-multiplying:
		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);

		if (norm_x != 0) {

			x0 = x0/norm_x;
			x1 = x1/norm_x;
			x2 = x2/norm_x;

			float u1 = 2 * (R0*x0 + R1*x1 + R2*x2);
			float u2 = 2 * (R3*x0 + R4*x1 + R5*x2);
			float u3 = 2 * (R6*x0 + R7*x1 + R8*x2);

			R0 = R0 - x0*u1;
			R3 = R3 - x0*u2;
			R6 = R6 - x0*u3;
			R1 = R1 - x1*u1;
			R4 = R4 - x1*u2;
			R7 = R7 - x1*u3;
			R2 = R2 - x2*u1;
			R5 = R5 - x2*u2;
			R8 = R8 - x2*u3;

			float Q0tmp = Q0 - 2*(Q0*x0*x0 + Q3*x1*x0 + Q6*x2*x0);
			float Q3tmp = Q3 - 2*(Q0*x0*x1 + Q3*x1*x1 + Q6*x2*x1);
			float Q6tmp = Q6 - 2*(Q0*x0*x2 + Q3*x1*x2 + Q6*x2*x2);
			float Q1tmp = Q1 - 2*(Q1*x0*x0 + Q4*x1*x0 + Q7*x2*x0);
			float Q4tmp = Q4 - 2*(Q1*x0*x1 + Q4*x1*x1 + Q7*x2*x1);
			float Q7tmp = Q7 - 2*(Q1*x0*x2 + Q4*x1*x2 + Q7*x2*x2);
			float Q2tmp = Q2 - 2*(Q2*x0*x0 + Q5*x1*x0 + Q8*x2*x0);
			float Q5tmp = Q5 - 2*(Q2*x0*x1 + Q5*x1*x1 + Q8*x2*x1);
			float Q8tmp = Q8 - 2*(Q2*x0*x2 + Q5*x1*x2 + Q8*x2*x2);

			Q0 = Q0tmp;
			Q1 = Q1tmp;
			Q2 = Q2tmp;
			Q3 = Q3tmp;
			Q4 = Q4tmp;
			Q5 = Q5tmp;
			Q6 = Q6tmp;
			Q7 = Q7tmp;
			Q8 = Q8tmp;
		}

		// k = 1;
		x0 = 0;
		x1 = R4;
		x2 = R5;

		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);
		g = -sign(x1) * norm_x;
		x1 = x1 - g;

		// Orthogonal transformation matrix that eliminates one element
		// below the diagonal of the matrix it is post-multiplying:
		norm_x = sqrt(x0*x0 + x1*x1 + x2*x2);

		if (norm_x != 0) {

			x0 = x0/norm_x;
			x1 = x1/norm_x;
			x2 = x2/norm_x;

			float u1 = 2 * (R0*x0 + R1*x1 + R2*x2);
			float u2 = 2 * (R3*x0 + R4*x1 + R5*x2);
			float u3 = 2 * (R6*x0 + R7*x1 + R8*x2);

			R0 = R0 - x0*u1;
			R3 = R3 - x0*u2;
			R6 = R6 - x0*u3;
			R1 = R1 - x1*u1;
			R4 = R4 - x1*u2;
			R7 = R7 - x1*u3;
			R2 = R2 - x2*u1;
			R5 = R5 - x2*u2;
			R8 = R8 - x2*u3;

			float Q0tmp = Q0 - 2*(Q0*x0*x0 + Q3*x1*x0 + Q6*x2*x0);
			float Q3tmp = Q3 - 2*(Q0*x0*x1 + Q3*x1*x1 + Q6*x2*x1);
			float Q6tmp = Q6 - 2*(Q0*x0*x2 + Q3*x1*x2 + Q6*x2*x2);
			float Q1tmp = Q1 - 2*(Q1*x0*x0 + Q4*x1*x0 + Q7*x2*x0);
			float Q4tmp = Q4 - 2*(Q1*x0*x1 + Q4*x1*x1 + Q7*x2*x1);
			float Q7tmp = Q7 - 2*(Q1*x0*x2 + Q4*x1*x2 + Q7*x2*x2);
			float Q2tmp = Q2 - 2*(Q2*x0*x0 + Q5*x1*x0 + Q8*x2*x0);
			float Q5tmp = Q5 - 2*(Q2*x0*x1 + Q5*x1*x1 + Q8*x2*x1);
			float Q8tmp = Q8 - 2*(Q2*x0*x2 + Q5*x1*x2 + Q8*x2*x2);

			Q0 = Q0tmp;
			Q1 = Q1tmp;
			Q2 = Q2tmp;
			Q3 = Q3tmp;
			Q4 = Q4tmp;
			Q5 = Q5tmp;
			Q6 = Q6tmp;
			Q7 = Q7tmp;
			Q8 = Q8tmp;
		}

		//////////////////////////////////////

		S0 = R0;
		S1 = R1;
		S2 = R2;
		S3 = R3;
		S4 = R4;
		S5 = R5;
		S6 = R6;
		S7 = R7;
		S8 = R8;

		float V0tmp = V0 * Q0 + V3 * Q1 + V6 * Q2;
		float V1tmp = V1 * Q0 + V4 * Q1 + V7 * Q2;
		float V2tmp = V2 * Q0 + V5 * Q1 + V8 * Q2;
		float V3tmp = V0 * Q3 + V3 * Q4 + V6 * Q5;
		float V4tmp = V1 * Q3 + V4 * Q4 + V7 * Q5;
		float V5tmp = V2 * Q3 + V5 * Q4 + V8 * Q5;
		float V6tmp = V0 * Q6 + V3 * Q7 + V6 * Q8;
		float V7tmp = V1 * Q6 + V4 * Q7 + V7 * Q8;
		float V8tmp = V2 * Q6 + V5 * Q7 + V8 * Q8;
		V0 = V0tmp;
		V1 = V1tmp;
		V2 = V2tmp;
		V3 = V3tmp;
		V4 = V4tmp;
		V5 = V5tmp;
		V6 = V6tmp;
		V7 = V7tmp;
		V8 = V8tmp;

		// exit when we get "close"
		float E = sqrt(S3*S3 + S6*S6 + S7*S7);
		float F = sqrt(S0*S0 + S4*S4 + S8*S8);
		if (F == 0) {
			F = 1;
		}
		err = E/F;
		loopcount = loopcount + 1;
	}

	// fix the signs in S
	float SS0 = S0;
	float SS1 = S4;
	float SS2 = S8;

	if (SS0 < -0.000000001) {
		U0 = -U0;
		U1 = -U1;
		U2 = -U2;
	}

	if (SS1 < -0.000000001) {
		U3 = -U3;
		U4 = -U4;
		U5 = -U5;
	}

	if (SS2 < -0.000000001) {
		U6 = -U6;
		U7 = -U7;
		U8 = -U8;
	}

	// transform input coords
	float x_min_pStar = x - pStar0;
	float y_min_pStar = y - pStar1;
	float z_min_pStar = z - pStar2;

	float VUtrans0 = V0*U0 + V3*U3 + V6*U6;
	float VUtrans1 = V1*U0 + V4*U3 + V7*U6;
	float VUtrans2 = V2*U0 + V5*U3 + V8*U6;
	float VUtrans3 = V0*U1 + V3*U4 + V6*U7;
	float VUtrans4 = V1*U1 + V4*U4 + V7*U7;
	float VUtrans5 = V2*U1 + V5*U4 + V8*U7;
	float VUtrans6 = V0*U2 + V3*U5 + V6*U8;
	float VUtrans7 = V1*U2 + V4*U5 + V7*U8;
	float VUtrans8 = V2*U2 + V5*U5 + V8*U8;

	if (VUtrans0*VUtrans4*VUtrans8 + VUtrans3*VUtrans7*VUtrans2 + VUtrans6*VUtrans1*VUtrans5 - VUtrans2*VUtrans4*VUtrans6 - VUtrans5*VUtrans7*VUtrans0 - VUtrans8*VUtrans1*VUtrans3 <= 0)
	{
		V6 = -V6;
		V7 = -V7;
		V8 = -V8;
		VUtrans0 = V0*U0 + V3*U3 + V6*U6;
		VUtrans1 = V1*U0 + V4*U3 + V7*U6;
		VUtrans2 = V2*U0 + V5*U3 + V8*U6;
		VUtrans3 = V0*U1 + V3*U4 + V6*U7;
		VUtrans4 = V1*U1 + V4*U4 + V7*U7;
		VUtrans5 = V2*U1 + V5*U4 + V8*U7;
		VUtrans6 = V0*U2 + V3*U5 + V6*U8;
		VUtrans7 = V1*U2 + V4*U5 + V7*U8;
		VUtrans8 = V2*U2 + V5*U5 + V8*U8;
	}

	float x_transformed = VUtrans0*x_min_pStar + VUtrans3*y_min_pStar + VUtrans6*z_min_pStar + qStar0;
	float y_transformed = VUtrans1*x_min_pStar + VUtrans4*y_min_pStar + VUtrans7*z_min_pStar + qStar1;
	float z_transformed = VUtrans2*x_min_pStar + VUtrans5*y_min_pStar + VUtrans8*z_min_pStar + qStar2;
	
	float4 v_transformed = (float4)(x_transformed, y_transformed, z_transformed, 0.0f);
	return v_transformed;

}


__kernel void backprojectKernelDeformed(
		__global TvoxelValue* pVolume,
		int sizeX,
		int sizeY,
		int recoSizeZ,
		int offset,
		float voxelSpacingX,
		float voxelSpacingY,
		float voxelSpacingZ,
		float offsetX,
		float offsetY, 
		float offsetZ,
		__read_only image2d_t gTex2D,
		__constant Tcoord_dev* gProjMatrix,
		float projMultiplier,
		global float* p,
		global float* q,
		int noPoints)
{
	
	int gidx = get_group_id(0);
	int gidy = get_group_id(1);
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	
	int locSizex = get_local_size(0);
	int locSizey = get_local_size(1);
	
	unsigned int x = gidx*locSizex+lidx;
    unsigned int y = gidy*locSizey+lidy;
	
	if (x >= sizeX || y >= sizeY)
		return;
		
	unsigned int zStride = sizeX*sizeY;
	unsigned int yStride = sizeX;

	for (unsigned int z = 0; z < recoSizeZ; z++){

		float4 coords_new = mls_deformation((x * voxelSpacingX) - offsetX, (y * voxelSpacingY) - offsetY, (z * voxelSpacingZ) - offsetZ, p, q, noPoints);
		float xcoord = coords_new.x;
		float ycoord = coords_new.y;
		float zcoord = coords_new.z;
		
		// precompute everything except z from the matrix multiplication
		float precomputeR = gProjMatrix[3]  + ycoord * gProjMatrix[1] + xcoord * gProjMatrix[0];
		float precomputeS = gProjMatrix[7]  + ycoord * gProjMatrix[5] + xcoord * gProjMatrix[4];
	    float precomputeT = gProjMatrix[11] + ycoord * gProjMatrix[9] + xcoord * gProjMatrix[8];

		// compute homogeneous coordinates
		float r = zcoord * gProjMatrix[2]  + precomputeR;
		float s = zcoord * gProjMatrix[6]  + precomputeS;
		float t = zcoord * gProjMatrix[10] + precomputeT;
		
		// compute projection coordinates 
		float denom = 1.0f / t;
		float fu = r * denom;
		float fv = s * denom;
		
		float proj_val = read_imagef(gTex2D, sampler, (float2)(fu + 0.5f + offset, fv + 0.5f)).x;
		
		// compute volume index for x,y,z
		unsigned long idx = z*zStride + y*yStride + x;
    
		// Distance weighting by 1/t^2
		pVolume[idx] += proj_val * denom * denom * projMultiplier;
		
	}
	
    return;
}



/*
 * Copyright (C) 2010-2021 Martin Berger, Jennifer Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/