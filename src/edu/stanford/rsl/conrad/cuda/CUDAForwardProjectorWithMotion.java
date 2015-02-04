package edu.stanford.rsl.conrad.cuda;

//TODO: Use our own matrices instead of Jama.Matrix

import java.util.ArrayList;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY3D_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY3D;
import jcuda.driver.CUaddress_mode;
import jcuda.driver.CUarray;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUdevprop;
import jcuda.driver.CUfilter_mode;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.dim3;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.ProjectionTableFileTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.XmlUtils;



/**
 * Forward projection expects input of a volumetric phantom scaled to mass density. Projection result {@latex.inline $p(\\vec{x})$} is then the accumulated mass along the ray {@latex.inline $\\vec{x}$} which consists of the line segments {@latex.inline $x_i$} in {@latex.inline $[\\textnormal{cm}]$} with the mass densities {@latex.inline $\\mu_i$} in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^3}]$}.
 * The actual projection is then computed as:<br>
 * {@latex.inline $$p(\\vec{x}) = \\sum_{i} x_i \\cdot \\mu_i$$}<BR>
 * The projection values are then returned in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^2}]$}
 * @author akmaier, Choi, Martin
 *
 */
public class CUDAForwardProjectorWithMotion implements GUIConfigurable, Citeable {

	static int bpBlockSize[] = {16, 16};

	private static boolean debug = true;
	/**
	 * The CUDA module containing the kernel
	 */
	private CUmodule module = null;
	/**
	 * The handle for the CUDA function of the kernel that is to be called
	 */
	private CUfunction function = null;
	/**
	 * The 3D volume texture reference
	 */
	private CUtexref gTex3D = null;
	private CUarray gVolume = null;

	private CUdeviceptr gVolumeEdgeMaxPoint = null;
	private CUdeviceptr gVolumeEdgeMinPoint = null;
	private CUdeviceptr gInvARmatrix = null;
	private CUdeviceptr gSrcPoint = null;
	private CUdeviceptr gProjection = null;
	/**
	 * the context
	 */
	private CUcontext cuCtx = null;

	private boolean initialized = false;
	private boolean largeVolumeMode = false;
	private int nSteps;
	private int currentStep = 0;
	private float [] voxelSize = null;
	private float [] volumeSize = null;
	private boolean configured = false;
	private float [] volumeEdgeMinPoint = null;
	private float [] volumeEdgeMaxPoint = null;
	// buffer for 3D volume:
	private float [] h_volume;
	private int subVolumeZ;

	private ImagePlus tex3D = null;

	/**
	 * Gets the volume to project
	 * @return the volume as image plus
	 */
	public ImagePlus getTex3D() {
		return tex3D;
	}

	/**
	 * Sets the volume to project
	 * @param tex3d
	 */
	public void setTex3D(ImagePlus tex3d) {
		tex3D = tex3d;
	}
	
	/**
	 * The XML filename where the rigid motion parameters are stored
	 */
	private String rotationTranslationFilename = null;

	private float [] projection;
	private int width;
	private int height;
	private Trajectory geometry;

	/**
	 * 
	 * 	
	 *  Method: computeCanonicalProjectionMatrix<br>
	 *  Author: Sungwon Yoon<br>
	 *  Description:<br>
	 *  <pre> 
	 *         W -> W projection matrix = [ AR   t ]
	 *         C -> C projection matrix = T0 * [AR  t] * T4
	 *                 
	 *                                 [ [ du(0)  dv(0) ]^-1   -0.5 ]
	 *                where      T0 =  [ [ du(1)  dv(1) ]      -0.5 ]  
	 *                                 [     0      0            1  ]      ,
	 *
	 *                                 [  dx   0    0   -(L-1)/2*dx  ]
	 *                           T4 =  [  0   dy    0   -(M-1)/2*dy  ]
	 *                                 [  0    0   dz   -(N-1)/2*dz  ]
	 *                                 [  0    0    0         1      ]
	 *  </pre>
	 *
	 *  C -> C projection matrix can be written as
	 *  <pre>
	 *        C -> C projection matrix = [ T0 * AR * T4(1:3,1:3)    T0 * (AR * T4(1:3,4) + t) ] 
	 *  
	 *  Therefore, the new invARmatrix = T4(1:3,1:3)^-1 * (AR)^-1 * T0^-1
	 *                                   [ 1/dx    0      0   ]             [du(0)  dv(0)  0]
	 *                                 = [  0     1/dy    0   ] * (AR)^-1 * [du(1)  dv(1)  0]
	 *                                   [  0      0     1/dz ]             [  0      0    1]
	 *
	 *            and the new srcPoint = -T4(1:3,1:3)^-1 * T4(1:3,4) - T4(1:3,1:3)^-1 * (AR)^-1 * t
	 *                                     [[ -0.5 * (L-1) ]     [ 1/dx    0      0   ]                 ]
	 *                                 = - [[ -0.5 * (M-1) ]  +  [  0     1/dy    0   ] * srcPoint^{W}  ]
	 *                                     [[ -0.5 * (N-1) ]     [  0      0     1/dz ]                 ]
	 *  </pre>  
	 *  
	 *  <BR><BR>
	 *  This implementation is consistent with Andreas Keil's Projection class and Benni's inversion.
	 *                                   
	 * @param canonicalProjMatrix is filled with a 3x4 projection matrix in this canonical format
	 * @param invARmatrix is filled with the inverse of AR in canonical format
	 * @param srcPoint is filled with the 3x1 source point in canonical format 
	 * @param projectionMatrix the Matrix on which the conversion is based.
	 */

	public void computeCanonicalProjectionMatrix(float [] canonicalProjMatrix, float [] invARmatrix, float [] srcPoint, Jama.Matrix projectionMatrix){

		double [] du = {Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX(), 0};
		double [] dv = {0, Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY()};


		// Assumed the vectors have only one non-zero element
		// use 
		du[0] = 1;
		dv[1] = 1;



		/*  ---------------------------------------------------
		 *
		 *  Canonical projection matrix used for BP computation
		 *
		 *  ---------------------------------------------------
		 */
		// T0 matrix
		Jama.Matrix T0 = new Jama.Matrix (3,3);
		double denom = 1.0f / (du[0]*dv[1] - dv[0]*du[1]);
		T0.set(0,0, denom * dv[1]);
		T0.set(0,1,-denom * dv[0]);
		T0.set(1,0, -denom * du[1]);
		T0.set(1,1, denom * du[0]);
		T0.set(0,2, -0.5);
		T0.set(1,2, -0.5);
		T0.set(2,2, 1.0);
		//T0.print(NumberFormat.getInstance(), 8);

		// T4 matrix 
		Jama.Matrix T4 = new Jama.Matrix(4,4);
		T4.set(0,0, voxelSize[0]);
		T4.set(1,1, voxelSize[1]);
		T4.set(2,2, voxelSize[2]);
		for (int k=0; k<3; ++k){
			T4.set(k,3, -0.5 * (volumeSize[k] - 1.0) * voxelSize[k]);
		}
		T4.set(3,3, 1.0);

		// New projection matrix in Canonical coord sys
		Jama.Matrix tmpMatrix;
		Jama.Matrix newProjMat;
		tmpMatrix = projectionMatrix.times(T4);
		newProjMat = T0.times(tmpMatrix);
		for (int r=0; r<3; ++r) {
			for (int c=0; c<4; ++c) {
				// 3x3 matrix 1st row (indices 0-3), 2nd row (indices 4-7),
				//   and 3rd row (indices 8-11)
				canonicalProjMatrix[4*r + c] = (float) newProjMat.get(r,c);
			}
		}

		/*  -----------------------------------------------------------
		 *
		 *  Canonical inverse projection matrix used for FP computation
		 *
		 *  -----------------------------------------------------------
		 */
		// Inverse of T0 matrix
		//T0.inverse().print(NumberFormat.getInstance(), 8);
		T0.set(0,0, du[0]);
		T0.set(0,1, dv[0]);
		T0.set(1,0, du[1]);
		T0.set(1,1, dv[1]);
		T0.set(0,2, 0);//0.5 * (du[0]+dv[0]));
		T0.set(1,2, 0);//0.5 * (du[1]+dv[1]));
		//T0.print(NumberFormat.getInstance(), 8);

		// Inverse scaling by dx, dy, dz
		Jama.Matrix invVoxelScale = new Jama.Matrix(3,3);
		invVoxelScale.set(0,0, 1.0/voxelSize[0]);
		invVoxelScale.set(1,1, 1.0/voxelSize[1]);
		invVoxelScale.set(2,2, 1.0/voxelSize[2]);

		// New invARmatrix in the Canonical coord sys
		Jama.Matrix invARmatrixMat = projectionMatrix.getMatrix(0, 2, 0, 2).inverse();
		tmpMatrix = invARmatrixMat.times(T0);    // invARmatrix_ * T0^{-1}
		Jama.Matrix invAR = invVoxelScale.times(tmpMatrix);     // invVoxelScale * (invARmatrix_ * T0)
		for (int r=0; r<3; ++r) {
			for (int c=0; c<3; ++c) {
				// 3x3 matrix 1st row (indices 0-2), 2nd row (indices 3-5),
				//   and 3rd row (indices 6-8)
				invARmatrix[3*r + c] = (float) invAR.get(r,c);

			}
		}
		//invAR.print(NumberFormat.getInstance(), 6);
		// New srcPoint in the Canonical coord sys

		Jama.Matrix srcPtW = computeSrcPt(projectionMatrix, invARmatrixMat);
		srcPoint[0] = (float) -(-0.5 * (volumeSize[0] -1.0) + invVoxelScale.get(0,0) * srcPtW.get(0, 0)); 
		srcPoint[1] = (float) -(-0.5 * (volumeSize[1] -1.0) + invVoxelScale.get(1,1) * srcPtW.get(1, 0)); 
		srcPoint[2] = (float) -(-0.5 * (volumeSize[2] -1.0) + invVoxelScale.get(2,2) * srcPtW.get(2, 0)); 


	}

	/**
	 * computes the location of the Source Point given a 3x4 projection matrix and and inverted 3x3 AR Projection matrix.
	 * Used in computeCanonicalProjectionMatrix
	 * 
	 * @param projectionMatrix the original projection matrix
	 * @param invertedProjMatrix the inverted AR projection matrix
	 * @return the source point
	 */
	private Jama.Matrix computeSrcPt(Jama.Matrix projectionMatrix, Jama.Matrix invertedProjMatrix) {
		Jama.Matrix at = projectionMatrix.getMatrix(0, 2, 3, 3);
		at.times(-1.0);
		return invertedProjMatrix.times(at);
	}

	/**
	 * Initiates communication with the CUDA card.
	 */
	private void init(){
		if (!initialized) {
			largeVolumeMode = false;
			// Initialize the JCudaDriver. Note that this has to be done from 
			// the same thread that will later use the JCudaDriver API.
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			CUdevice dev = new CUdevice();
			JCudaDriver.cuDeviceGet(dev, 0);
			cuCtx = new CUcontext();
			JCudaDriver.cuCtxCreate(cuCtx, 0, dev);
			// check space on device:
			int [] memory = new int [1];
			int [] total = new int [1]; 
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			JCudaDriver.cuMemGetInfo(memory, total);
			int availableMemory = (int) (CUDAUtil.correctMemoryValue(memory[0]) / ((long) 1024 * 1024));
			int requiredMemory = (int)(((
					((double) volumeSize[0]) * volumeSize[1] * ((double) volumeSize[2]) * Sizeof.FLOAT) 
					+ (((double) height) * width * Sizeof.FLOAT)) 
					/ (1024.0 * 1024));
			if (debug) {
				System.out.println("Total available Memory on CUDA card:" + availableMemory);
				System.out.println("Required Memory on CUDA card:" + requiredMemory);
			}
			if (requiredMemory > availableMemory){
				nSteps = CUDAUtil.iDivUp (requiredMemory, (int)(availableMemory));
				if (debug) System.out.println("Switching to large volume mode with nSteps = " + nSteps);
				largeVolumeMode = true;
			}
			if (debug) {
				CUdevprop prop = new CUdevprop();
				JCudaDriver.cuDeviceGetProperties(prop, dev);
				System.out.println(prop.toFormattedString());
			}
			// Load the CUBIN file containing the kernel
			module = new CUmodule();
			JCudaDriver.cuModuleLoad(module, "projectWithCuda.sm_10.cubin");

			// Obtain a function pointer to the kernel function. This function
			// will later be called. 
			// 
			function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module, "_Z13projectKernelPfjf");


			int memorysize = (int) (volumeSize[0] * volumeSize[1] * volumeSize[2] * Sizeof.FLOAT);
			if (largeVolumeMode){
				subVolumeZ = CUDAUtil.iDivUp((int) volumeSize[2], nSteps);
				if(debug) System.out.println("SubVolumeZ: " + subVolumeZ);
				h_volume = new float[(int) (volumeSize[0] * volumeSize[1] * subVolumeZ)];
				memorysize = (int) (volumeSize[0] * volumeSize[1] * subVolumeZ * Sizeof.FLOAT);
				if(debug)System.out.println("Memory: " + memorysize);
			} else {
				h_volume = new float[(int) (volumeSize[0] * volumeSize[1] * volumeSize[2])];
				subVolumeZ = (int) volumeSize[2];
				nSteps = 1;
			}

			copyVolumeToCard();

			gTex3D = new CUtexref();
			// Obtain the 3D texture reference for the volume data from 
			// the module, set its parameters and assign the 3D volume 
			// data array as its reference.
			JCudaDriver.cuModuleGetTexRef(gTex3D, module, "gTex3D");
			JCudaDriver.cuTexRefSetFilterMode(gTex3D,
					CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
			JCudaDriver.cuTexRefSetAddressMode(gTex3D, 0,
					CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
			JCudaDriver.cuTexRefSetAddressMode(gTex3D, 1,
					CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
			JCudaDriver.cuTexRefSetFormat(gTex3D,
					CUarray_format.CU_AD_FORMAT_FLOAT, 1);
			JCudaDriver.cuTexRefSetFlags(gTex3D,
					JCudaDriver.CU_TRSF_READ_AS_INTEGER);
			JCudaDriver.cuTexRefSetArray(gTex3D, gVolume,
					JCudaDriver.CU_TRSA_OVERRIDE_FORMAT);


			CUDAUtil.copyFloatArrayToDevice(voxelSize, module, "gVoxelElementSize");

			// copy volume to device
			gProjection = new CUdeviceptr();
			JCudaDriver.cuMemAlloc(gProjection, width * height * Sizeof.FLOAT);


			initialized = true;
		}

	}


	/**
	 * Load the inverted projection matrix for the current projection and resets the projection data.
	 * @param projectionNumber
	 */
	private void prepareProjection(int projectionNumber){
		float [] cann = new float[3*4];
		float [] invAR = new float[3*3];
		float [] srcP = new float[3];
		
		// load the motion transforms
		// load data from XML file
		SimpleMatrix[] motion = readInMotionMatrices();		
		
		SimpleMatrix projMat = geometry.getProjectionMatrix(projectionNumber).computeP();
		double [][] mat = new double [3][4];
		
		SimpleOperators.multiplyMatrixProd(projMat, motion[projectionNumber]).copyTo(mat); // apply motion
//		projMat.copyTo(mat);// No motion applied
		
		computeCanonicalProjectionMatrix(cann, invAR, srcP, new Jama.Matrix(mat));
		if (gInvARmatrix == null){
			gInvARmatrix = CUDAUtil.copyFloatArrayToDevice(invAR, module, "gInvARmatrix");
			gSrcPoint = CUDAUtil.copyFloatArrayToDevice(srcP, module, "gSrcPoint");
		} else {
			CUDAUtil.updateFloatArrayOnDevice(gInvARmatrix, invAR, module);
			CUDAUtil.updateFloatArrayOnDevice(gSrcPoint, srcP, module);
		}
		// reset Projection Data
		JCudaDriver.cuMemsetD32(gProjection, 0, width * height);
	}

	public SimpleMatrix[] readInMotionMatrices() {

		SimpleMatrix[] motion = new SimpleMatrix[geometry.getNumProjectionMatrices()];
		// load data from XML file
		ArrayList<double[][][]> RotTrans = null;
		try {
			if (rotationTranslationFilename == null)
				rotationTranslationFilename = FileUtil.myFileChoose(".xml", false);
			
			RotTrans = (ArrayList<double[][][]>)XmlUtils.importFromXML(rotationTranslationFilename);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for (int i = 0; i < geometry.getNumProjectionMatrices(); i++) {
			motion[i] = new SimpleMatrix(4,4);
			SimpleMatrix rotation = new SimpleMatrix(RotTrans.get(0)[i]);
			SimpleMatrix translation = new SimpleMatrix(RotTrans.get(1)[i]);
//			motion[i].setSubMatrixValue(0, 0, rotation.transposed());
//			motion[i].setSubMatrixValue(0, 3, translation.multipliedBy(-1));
			motion[i].setSubMatrixValue(0, 0, rotation); //Correct
			motion[i].setSubMatrixValue(0, 3, translation); //Correct
			motion[i].setRowValue(3, new SimpleVector(0,0,0,1));
		}
		
		return motion;
		
	}
	

	/**
	 * Main method checks different computation methods and compares to a reference implementation.
	 * The P matrix
	 * [[   -1.957647643397161     -1.5159990314154403    2.6654943287669416E-5    284.3151786700395 ]; 
	 *  [ -0.22689199744970734     0.21980776263540414      -2.4371669460715766   219.19724835256451 ];
	 *  [-9.088284790690905E-4    8.644729632667306E-4    1.7220251080057994E-5                  1.0 ]]
	 * 
	 * should yield the following source point in texture coordinates
	 * 1402.969301
	 * -851.170185
	 * 228.742784
	 * 
	 * and this inverse (up to a scalar factor):
	 * -0.563161 	-0.006972 	-985.825785
	 * -0.592038 	 0.008988 	1273.025240
	 * -0.000967 	-0.819165 	206.591022
	 * 
	 * This part should be moved to a test-case some time later.
	 * 
	 * @param args
	 */
	public static void main(String[] args){
		Configuration.loadConfiguration();
		/*
		double [] matrix = {-1.957647643397161,    -1.5159990314154403,    2.6654943287669416E-5,   284.3151786700395,
				-0.22689199744970734,    0.21980776263540414,    -2.4371669460715766,   219.19724835256451,
				-9.088284790690905E-4,    8.644729632667306E-4,    1.7220251080057994E-5,    1.0 };
		*/
		Projection proj = new Projection();
		proj.setPMatrixSerialization("[[-1.957647643397161    -1.5159990314154403    2.6654943287669416E-5   284.3151786700395]; " +
				"[-0.22689199744970734    0.21980776263540414    -2.4371669460715766   219.19724835256451 ];" +
		"[ -9.088284790690905E-4    8.644729632667306E-4    1.7220251080057994E-5    1.0 ]]");
		SimpleVector center = proj.computeCameraCenter();
		System.out.println("center = "+ center);
		System.out.println(proj.computeP());
		System.out.println(proj.getRTKinv());
		double scale = 206.591022 / proj.getRTKinv().getElement(2, 2);
		System.out.println(proj.getRTKinv().multipliedBy(scale) +  "\n" + scale + " " + proj.computeSourceToDetectorDistance(new SimpleVector(0.320, 0.320))[0]);

		SimpleMatrix m = proj.computeP().getSubMatrix(0, 0, 3, 3);
		System.out.println("M="+ m.inverse(SimpleMatrix.InversionType.INVERT_QR).multipliedBy(2));

		float [] cann = new float[3*4];
		float [] invAR = new float[3*3];
		float [] srcP = new float[3];
		SimpleMatrix projMat = proj.computeP();
		double [][] mat = new double [3][4];
		projMat.copyTo(mat);
		CUDAForwardProjectorWithMotion cudaForwardProjector = new CUDAForwardProjectorWithMotion();
		cudaForwardProjector.voxelSize = new float [] {0.5f, 0.5f, 0.5f};
		cudaForwardProjector.volumeSize = new float [] {512f, 512f, 512f};
		cudaForwardProjector.computeCanonicalProjectionMatrix(cann, invAR, srcP, new Jama.Matrix(mat));
		for (int i =0; i < 9; i++){
			System.out.println(invAR[i]);
		}
		for (int i =0; i < 3; i++){
			System.out.println(srcP[i]);
		}
	}

	/**
	 * destroys the CUDA context and frees the allocated memory
	 */
	private void unload(){
		JCudaDriver.cuMemFree(gProjection);
		// destory context
		JCudaDriver.cuCtxDestroy(cuCtx);
	}


	/**
	 * loads the actual CUDA kernel and performs the projection
	 * @param projectionNumber the projection number.
	 * @return
	 */
	private ImageProcessor project(int projectionNumber){
		init();
		prepareProjection(projectionNumber);

		Pointer dOut = Pointer.to(gProjection);
		Pointer pStride = Pointer.to(new int[]{width});
		Pointer pstepsize = Pointer.to(new float[]{(float) 1});

		int offset = 0;

		offset = CUDAUtil.align(offset, Sizeof.POINTER);
		JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
		offset += Sizeof.POINTER;

		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, pStride, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pstepsize, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		JCudaDriver.cuParamSetSize(function, offset);

		dim3 gridSize = new dim3(
				CUDAUtil.iDivUp(width, bpBlockSize[0]), 
				CUDAUtil.iDivUp(height, bpBlockSize[0]), 
				1);


		//System.out.println("Grid: " + gridSize);

		JCudaDriver.cuFuncSetBlockShape(function, bpBlockSize[0], bpBlockSize[1], 1);
		JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
		JCudaDriver.cuCtxSynchronize();

		JCudaDriver.cuMemcpyDtoH(Pointer.to(projection), gProjection, width * height * Sizeof.FLOAT);
		FloatProcessor fl = new FloatProcessor(width, height, projection, null);
		

		// TODO: Normalization is never considered in the backprojectors, 
		// 		 thus, iteratively applying forward and backward projections
		//		 would yield to a scaling issue!
		// conversion from [g*mm/cm^3] = [g*0.1cm/cm^3] to [g/cm^2]
		// fl.multiply(1.0 / 10);
		
		if (geometry instanceof ProjectionTableFileTrajectory){
			fl.flipVertical();
		}
		return fl;
	}

	/**
	 * Starts projection and returns Projection Data, as ImagePlus
	 * @return the projection stack
	 */
	public ImagePlus project(){
		ImageStack stack = new ImageStack(width, height);
		for (int i = 0; i < geometry.getNumProjectionMatrices(); i++){
			stack.addSlice("Projection " + i, project(i).duplicate());
		}
		if (largeVolumeMode){
			// play it again, Sam!
			while (currentStep < nSteps) {
				currentStep++;

				if (currentStep == nSteps) break;
				if (debug) System.out.println("Processing step " + currentStep + " of " + nSteps);
				copyVolumeToCard();
				for (int i = 0; i < geometry.getNumProjectionMatrices(); i++){
					ImageUtil.addProcessors(stack.getProcessor(i+1), project(i));
					//stack.addSlice("slice " + i, project(i).duplicate());
				}
			}
		}
		ImagePlus image = new ImagePlus();
		image.setStack("Forward Projection of " + tex3D.getTitle(), stack);
		unload();
		//image.show();
		return image;
	}

	private void copyVolumeToCard(){

		// write the volume from ImagePlus to a float array.
		for (int k = 0; k < subVolumeZ; k++){
			int index = ((nSteps - currentStep -1)*subVolumeZ) + k + 1;
			if (index <= tex3D.getStackSize()){
				ImageProcessor currentSlice = tex3D.getStack().getProcessor(index);
				for (int i = 0; i< volumeSize[0]; i++){
					for(int j =0; j< volumeSize[1]; j++){
						boolean flip =  false;

						int index2 = (int) (
								(
										(
												(((int)volumeSize[1]) * (k))
												+ ((int)j)
										) * ((int)volumeSize[0])
								)
								+ ((int)i));
						if (flip) {
							index2 = (int) (
									// Note that we must flip the coordinates of the volume slices in order to be compatible with the CUDA Backprojector...
									(
											(
													(((int)volumeSize[1]) * (subVolumeZ - k -1))
													+ ((int)volumeSize[1]-j-1)
											) * ((int)volumeSize[0])
									)
									+ ((int)volumeSize[0] - i -1));
						}
						if (index2 >= (((int)volumeSize[0]) * ((int)volumeSize[1])* ((int)subVolumeZ))){
							System.out.println(k + " " + i + " " + j);
							break;
						}
						h_volume[index2] = currentSlice.getPixelValue(i, j);
					}
				}
			} else {
				System.out.println("Not in Volume: " + index + " " + nSteps + " " + currentStep);
				for (int i = 0; i< volumeSize[0]; i++){
					for(int j =0; j< volumeSize[1]; j++){
						h_volume[(int)((((((int)volumeSize[1]) * k) + j) * ((int)volumeSize[0]))+i)] = 0;
					}
				}
			}
		}
		int test = currentStep;
		volumeEdgeMaxPoint[2] = (float) ((test * subVolumeZ) + subVolumeZ -0.5 - CONRAD.SMALL_VALUE);
		volumeEdgeMinPoint[2] = (float) ((test * subVolumeZ) -0.5 - CONRAD.SMALL_VALUE);
		if (debug) System.out.println("New volume z min: " + volumeEdgeMinPoint[2] + " new volume z max: " + volumeEdgeMaxPoint[2]);
		if (gVolumeEdgeMaxPoint == null){
			gVolumeEdgeMaxPoint = CUDAUtil.copyFloatArrayToDevice(volumeEdgeMaxPoint, module, "gVolumeEdgeMaxPoint");
			gVolumeEdgeMinPoint = CUDAUtil.copyFloatArrayToDevice(volumeEdgeMinPoint, module, "gVolumeEdgeMinPoint");
		} else {
			CUDAUtil.updateFloatArrayOnDevice(gVolumeEdgeMaxPoint, volumeEdgeMaxPoint, module);
			CUDAUtil.updateFloatArrayOnDevice(gVolumeEdgeMinPoint, volumeEdgeMinPoint, module);
		}  

		if (gVolume == null) {
			gVolume = new CUarray();
			// Create the 3D array that will contain the volume data
			// and will be accessed via the 3D texture
			CUDA_ARRAY3D_DESCRIPTOR allocateArray = new CUDA_ARRAY3D_DESCRIPTOR();
			allocateArray.Width = (int) volumeSize[0];
			allocateArray.Height = (int) volumeSize[1];
			allocateArray.Depth = subVolumeZ;
			allocateArray.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
			allocateArray.NumChannels = 1;
			JCudaDriver.cuArray3DCreate(gVolume, allocateArray);
		}

		// Copy the volume data data to the 3D array
		CUDA_MEMCPY3D copy = new CUDA_MEMCPY3D();
		copy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
		copy.srcHost = Pointer.to(h_volume);
		copy.srcPitch = (int) volumeSize[0] * Sizeof.FLOAT;
		copy.srcHeight = (int) volumeSize[1];

		copy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
		copy.dstArray = gVolume;
		copy.dstPitch = (int) volumeSize[0] * Sizeof.FLOAT;
		copy.dstHeight = (int) volumeSize[1];
		copy.WidthInBytes = (int) volumeSize[0] * Sizeof.FLOAT;
		copy.Height = (int) volumeSize[1];
		copy.Depth = subVolumeZ;
		JCudaDriver.cuMemcpy3D(copy);
	}

	/**
	 * Start GUI configuration. Reads from global Configuration.
	 */
	public void configure() throws Exception {
		// TODO Auto-generated method stub
		voxelSize = new float [3];
		volumeSize = new float [3];
		Configuration config = Configuration.getGlobalConfiguration();
		voxelSize[0] = (float) config.getGeometry().getVoxelSpacingX();
		voxelSize[1] = (float) config.getGeometry().getVoxelSpacingY();
		voxelSize[2] = (float) config.getGeometry().getVoxelSpacingZ();
		volumeSize[0] = config.getGeometry().getReconDimensionX();
		volumeSize[1] = config.getGeometry().getReconDimensionY();
		volumeSize[2] = config.getGeometry().getReconDimensionZ();
		volumeEdgeMinPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMinPoint[i] = (float) (-0.5 + CONRAD.SMALL_VALUE);
		}
		volumeEdgeMaxPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMaxPoint[i] = (float) (volumeSize[i] -0.5 - CONRAD.SMALL_VALUE);
		}
		width = config.getGeometry().getDetectorWidth();
		height = config.getGeometry().getDetectorHeight();
		geometry = config.getGeometry();

		if (debug) System.out.println("Projection Matrices: " + geometry.getNumProjectionMatrices());
		projection = new float[width * height];
		configured = true;
	}

	/**
	 * returns whether the projector was already configured or not.
	 */
	public boolean isConfigured() {
		return configured;
	}

	/**
	 * Returns a reference to literature describing this algorithm in Bibtex format
	 */
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Galigekere03-CBR,\n" +
		"  author = {{Galigekere}, R. R. and {Wiesent}, K. and {Holdsworth}, D. W.},\n" +
		"  title = \"{{Cone-Beam Reprojection Using Projection-Matrices}}\",\n" +
		"  journal = {{IEEE Transactions on Medical Imaging}},\n" +
		"  year = 2003,\n" +
		"  volume = 22,\n"+
		"  number = 10,\n" +
		"  pages = {1202-1214}\n" +
		"}";
		return bibtex;
	}

	/**
	 * Returns a reference to literature describing this algorithm in Medline
	 */
	public String getMedlineCitation() {
		return "Galigekere RR, Wiesent K, and Holdsworth DW. Cone-Beam Reprojection Using Projection-Matrices. IEEE Transactions on Medical Imaging 22(10):1202-14 2003.";
	}



}

/*
 * Copyright (C) 2010-2014 - Andreas Maier, Martin Berger, Jang Hwang Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
