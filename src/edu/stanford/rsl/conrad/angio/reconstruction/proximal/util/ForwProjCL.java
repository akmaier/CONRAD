/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.util;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class ForwProjCL{

	static int bpBlockSize[] = {16, 16};

	private static boolean debug = false;
	/**
	 * The OpenCL context
	 */
	protected CLContext context;

	/**
	 * The OpenCL program
	 */
	private CLProgram program;

	/**
	 * The OpenCL device
	 */
	private CLDevice device;

	/**
	 * The OpenCL kernel function binding
	 */
	private CLKernel kernelFunction;

	/**
	 * The OpenCL command queue
	 */
	protected CLCommandQueue commandQueue;	

	/**
	 * The 3D volume texture reference
	 */
	private CLImage3d<FloatBuffer> gTex3D = null;

	/**
	 * The projection stack
	 */
	//private CLBuffer<FloatBuffer> gVolume = null;

	private CLBuffer<FloatBuffer> gVolumeEdgeMaxPoint = null;
	private CLBuffer<FloatBuffer> gVolumeEdgeMinPoint = null;
	private CLBuffer<FloatBuffer> gVoxelElementSize = null;
	protected CLBuffer<FloatBuffer> gInvARmatrix = null;
	protected CLBuffer<FloatBuffer> gSrcPoint = null;
	private CLBuffer<FloatBuffer> gProjection = null;
	
	private SimpleVector originShift = null;
	
	private Grid3D tex;
	
	protected Projection[] projectionMatrices = null;
	
	private boolean initialized = false;
	private boolean largeVolumeMode = false;
	private int nSteps;
	private int currentStep = 0;
	private float [] voxelSize = null;
	private float [] volumeSize = null;
	private double [] origin = null;
	private boolean configured = false;
	private float [] volumeEdgeMinPoint = null;
	private float [] volumeEdgeMaxPoint = null;
	// buffer for 3D volume:
	private float [] h_volume;
	private int subVolumeZ;

	private float [] projection;
	private int width;
	private int height;
	protected int nrProj;

	/**
	 * Sets the volume to project
	 * @param inputTex
	 */
	private void setTex3D(Grid3D inputTex) {
		this.tex = inputTex;
		volumeSize = new float[]{inputTex.getSize()[0],inputTex.getSize()[1],inputTex.getSize()[2]};
		if(inputTex.getSpacing() != null)
			voxelSize = new float[]{(float) inputTex.getSpacing()[0],(float) inputTex.getSpacing()[1],(float) inputTex.getSpacing()[2]};
		else{
			System.out.println("Cannot obtain voxel spacing from volume! Using configuration spacing instead!");
			Trajectory g = Configuration.getGlobalConfiguration().getGeometry();
			voxelSize = new float[]{(float) g.getVoxelSpacingX(), (float) g.getVoxelSpacingY(), (float) g.getVoxelSpacingZ()};
		}
		if(inputTex.getOrigin() != null)
			origin = new double[]{inputTex.getOrigin()[0], inputTex.getOrigin()[1], inputTex.getOrigin()[2]};
		else{
			System.out.println("Cannot obtain origin from volume! Using configuration origin instead!");
			Trajectory g = Configuration.getGlobalConfiguration().getGeometry();
			origin = new double[]{g.getOriginX(), g.getOriginY(), g.getOriginZ()};
		}
		volumeEdgeMaxPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMaxPoint[i] = (float) (volumeSize[i] -0.5 - CONRAD.SMALL_VALUE);
		}
	}
	

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
	public void computeCanonicalProjectionMatrix(CLBuffer<FloatBuffer> invARmatrix, CLBuffer<FloatBuffer> srcPoint,  Projection proj){
		computeCanonicalProjectionMatrix(null, invARmatrix, srcPoint, proj);
	}

	public void computeCanonicalProjectionMatrix(CLBuffer<FloatBuffer> detectorDirections, CLBuffer<FloatBuffer> invARmatrix, CLBuffer<FloatBuffer> srcPoint,  Projection proj){
		// Inverse scaling by dx, dy, dz
		SimpleMatrix invVoxelScale = new SimpleMatrix(3,3);
		invVoxelScale.setElementValue(0,0, 1.0/voxelSize[0]);
		invVoxelScale.setElementValue(1,1, 1.0/voxelSize[1]);
		invVoxelScale.setElementValue(2,2, 1.0/voxelSize[2]);


		// New invARmatrix in the Canonical coord sys
		SimpleMatrix invARmatrixMat = proj.getRTKinv();
		//SimpleMatrix invARmatrixMatTest = proj.computeP().getSubMatrix(0, 0, 3, 3).inverse(InversionType.INVERT_SVD);
		SimpleMatrix invAR = SimpleOperators.multiplyMatrixProd(invVoxelScale,invARmatrixMat);     // invVoxelScale * (invARmatrix_ * T0)
		for (int r=0; r<3; ++r) {
			for (int c=0; c<3; ++c) {
				// 3x3 matrix 1st row (indices 0-2), 2nd row (indices 3-5),
				//   and 3rd row (indices 6-8)
				invARmatrix.getBuffer().put((float) invAR.getElement(r,c));
			}
		}
		
		// shifted origin is express in updated T4 last column as origin shift in WC is added to it
		if(originShift==null)
			originShift = getOriginTransform();
		
		// New srcPoint in the Canonical coord sys
		SimpleVector srcPtW = proj.computeCameraCenter().negated();//computeSrcPt(projectionMatrix, invARmatrixMat);
		srcPoint.getBuffer().put((float) -(-0.5 * (volumeSize[0] -1.0) + originShift.getElement(0)*invVoxelScale.getElement(0,0) + invVoxelScale.getElement(0,0) * srcPtW.getElement(0))); 
		srcPoint.getBuffer().put((float) -(-0.5 * (volumeSize[1] -1.0) + originShift.getElement(1)*invVoxelScale.getElement(1,1) + invVoxelScale.getElement(1,1) * srcPtW.getElement(1))); 
		srcPoint.getBuffer().put((float) -(-0.5 * (volumeSize[2] -1.0) + originShift.getElement(2)*invVoxelScale.getElement(2,2) + invVoxelScale.getElement(2,2) * srcPtW.getElement(2))); 

		if(detectorDirections!=null){

			SimpleVector x = new SimpleVector(1,0,0);
			SimpleVector z = new SimpleVector(0,1,0);
			
			SimpleMatrix R = proj.getR().transposed();
			x=SimpleOperators.multiply(R,x);
			z=SimpleOperators.multiply(R,z);
			x.normalizeL2();
			z.normalizeL2();
			SimpleVector udir = x;
			SimpleVector vdir = z;
			/*

			SimpleVector x = projectionMatrix.getSubRow(0, 0, 3);
			SimpleVector y = projectionMatrix.getSubRow(1, 0, 3);
			SimpleVector imgPlane = projectionMatrix.getSubRow(2, 0, 3);


			SimpleVector udir = crossProduct(y,imgPlane).negated();
			SimpleVector vdir = crossProduct(x,imgPlane);
			udir.normalizeL2();
			vdir.normalizeL2();
			 */
			detectorDirections.getBuffer().put((float) udir.getElement(0));
			detectorDirections.getBuffer().put((float) udir.getElement(1));
			detectorDirections.getBuffer().put((float) udir.getElement(2));
			detectorDirections.getBuffer().put(0.f);
			detectorDirections.getBuffer().put((float) vdir.getElement(0));
			detectorDirections.getBuffer().put((float) vdir.getElement(1));
			detectorDirections.getBuffer().put((float) vdir.getElement(2));
			detectorDirections.getBuffer().put(0.f);
		}
	}

	/**
	 * computes the location of the Source Point given a 3x4 projection matrix and and inverted 3x3 AR Projection matrix.
	 * Used in computeCanonicalProjectionMatrix
	 * 
	 * @param projectionMatrix the original projection matrix
	 * @param invertedProjMatrix the inverted AR projection matrix
	 * @return the source point
	 */
	public static Jama.Matrix computeSrcPt(Jama.Matrix projectionMatrix, Jama.Matrix invertedProjMatrix) {
		Jama.Matrix at = projectionMatrix.getMatrix(0, 2, 3, 3);
		//at = at.times(-1.0);
		return invertedProjMatrix.times(at);
	}
	
	protected SimpleVector getOriginTransform(){
		SimpleVector currOrigin = new SimpleVector(this.origin);
		// compute centered origin as assumed by forward projector
		SimpleVector centeredOffset = new SimpleVector(this.volumeSize);
		SimpleVector voxelSpacing = new SimpleVector(this.voxelSize);
		centeredOffset.subtract(1);
		centeredOffset.multiplyElementWiseBy(voxelSpacing);
		centeredOffset.divideBy(-2);
		// compute the actual shift
		return SimpleOperators.subtract(currOrigin, centeredOffset);
	}

	/**
	 * Initiates communication with the graphics card.
	 */
	private void init(){
		if (!initialized) {
			largeVolumeMode = false;
			// Initialize JOCL.
			context = OpenCLUtil.createContext();

			try {
				// get the fastest device
				device = context.getMaxFlopsDevice();
				// create the command queue
				commandQueue = device.createCommandQueue();

				// initialize the program
				if (program==null || !program.getContext().equals(this.context)){
					program = context.createProgram(ForwProjCL.class.getResourceAsStream("projectCL.cl")).build();
				}

				// (1) check space on device - At the moment we simply use 90% of the overall available memory
				// (2) createFloatBuffer uses a byteBuffer internally --> h_volume.length cannot be > 2^31/4 = 2^31/2^2 = 2^29
				// 	   Thus, 2^29 would already cause a overflow (negative sign) of the integer in the byte buffer! Maximum length is (2^29-1) float or (2^31-4) bytes!
				// Either we are limited by the maximum addressable memory, i.e. (2^31-4) bytes or by the device limit "device.getGlobalMemSize()*0.9"
				long availableMemory =  Math.min((long)(device.getGlobalMemSize()*0.9),2147483647);
				long requiredMemory = (long) Math.ceil((((double) volumeSize[0]) * volumeSize[1] * ((double) volumeSize[2]) * Float.SIZE/8) 
						+ (((double) height) * width * Float.SIZE/8));
				if (debug) {
					System.out.println("Total available Memory on graphics card:" + availableMemory/1024/1024);
					System.out.println("Required Memory on graphics card:" + requiredMemory/1024/1024);
				}
				if (requiredMemory > availableMemory){
					// divup operation here
					nSteps = (int) OpenCLUtil.iDivUp(requiredMemory, availableMemory);
					if (debug) System.out.println("Switching to large volume mode with nSteps = " + nSteps);
					largeVolumeMode = true;
				}
				if (debug) {
					//TODO replace
					//CUdevprop prop = new CUdevprop();
					//JCudaDriver.cuDeviceGetProperties(prop, dev);
					//System.out.println(prop.toFormattedString());
				}

				// create the computing kernel
				kernelFunction = program.createCLKernel("projectKernel");


				long memorysize = (long) (volumeSize[0] * volumeSize[1] * volumeSize[2] * Float.SIZE / 8);
				if (largeVolumeMode){
					subVolumeZ = OpenCLUtil.iDivUp((int) volumeSize[2], nSteps);
					if(debug) System.out.println("SubVolumeZ: " + subVolumeZ);
					h_volume = new float[(int) (volumeSize[0] * volumeSize[1] * subVolumeZ)];
					memorysize = (int) (volumeSize[0] * volumeSize[1] * subVolumeZ * Float.SIZE / 8);
					if(debug)System.out.println("Memory: " + memorysize);
				} else {
					h_volume = new float[(int) (volumeSize[0] * volumeSize[1] * volumeSize[2])];
					subVolumeZ = (int) volumeSize[2];
					nSteps = 1;
				}

				copyVolumeToCard();

				gVoxelElementSize = context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
				gVoxelElementSize.getBuffer().put(voxelSize);
				gVoxelElementSize.getBuffer().rewind();

				gProjection = context.createFloatBuffer(width * height, Mem.WRITE_ONLY);

				commandQueue
				.putWriteImage(gTex3D, true)
				//.putWriteBuffer(gProjection, true)
				.putWriteBuffer(gVoxelElementSize,true)
				.putWriteBuffer(gVolumeEdgeMinPoint, true)
				.putWriteBuffer(gVolumeEdgeMaxPoint, true)
				.finish();
				
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}

			initialized = true;
		}

	}


	/**
	 * Get the current OpenCL program instance
	 */
	public CLProgram getOpenCLForwardProjectorInstance(){
		return program;
	}

	
	/**
	 * Load the inverted projection matrices for all projections and reset the projection data.
	 * @param projectionNumber
	 */
	protected void prepareAllProjections(){
		if (gInvARmatrix == null)
			gInvARmatrix = context.createFloatBuffer(3*3*nrProj, Mem.READ_ONLY);
		if (gSrcPoint == null)
			gSrcPoint = context.createFloatBuffer(3*nrProj, Mem.READ_ONLY);
		
		gInvARmatrix.getBuffer().rewind();
		gSrcPoint.getBuffer().rewind();
		for (int i=0; i < nrProj; ++i){
			computeCanonicalProjectionMatrix(gInvARmatrix, gSrcPoint, projectionMatrices[i]);
		}
		gInvARmatrix.getBuffer().rewind();
		gSrcPoint.getBuffer().rewind();
		
		commandQueue
		.putWriteBuffer(gSrcPoint, true)
		.putWriteBuffer(gInvARmatrix, true)
		.finish();
	}

	/**
	 * release all CL related objects and free memory
	 */
	public void unload(){
		if (commandQueue != null && !commandQueue.isReleased())
			commandQueue.release();
		//release all buffers
		if (gTex3D != null && !gTex3D.isReleased())
			gTex3D.release();
		if (gVolumeEdgeMaxPoint != null && !gVolumeEdgeMaxPoint.isReleased())
			gVolumeEdgeMaxPoint.release();
		if (gVolumeEdgeMinPoint != null && !gVolumeEdgeMinPoint.isReleased())
			gVolumeEdgeMinPoint.release();
		if (gVoxelElementSize != null && !gVoxelElementSize.isReleased())
			gVoxelElementSize.release();
		if (gInvARmatrix != null && !gInvARmatrix.isReleased())
			gInvARmatrix.release();
		if (gSrcPoint != null && !gSrcPoint.isReleased())
			gSrcPoint.release();
		if (gProjection != null && !gProjection.isReleased())
			gProjection.release();
		if (kernelFunction != null && !kernelFunction.isReleased())
			kernelFunction.release();
		if (program != null && !program.isReleased())
			program.release();
		if (context != null && !context.isReleased())
			context.release();
	}


	/**
	 * loads the actual OpenCL kernel and performs the projection
	 * @param projectionNumber the projection number.
	 * @return the image as image processor
	 */
	public ImageProcessor project(int projectionNumber){
		init();
		
		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(gProjection)
		.putArg(width)
		.putArg(height)
		.putArg(1.f)
		.putArg(gTex3D)
		.putArg(gVoxelElementSize)
		.putArg(gVolumeEdgeMinPoint)
		.putArg(gVolumeEdgeMaxPoint)
		.putArg(gSrcPoint)
		.putArg(gInvARmatrix)
		.putArg(projectionNumber);

		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {width, height}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}


		// add kernel function to the queue
		commandQueue
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		.finish()
		.putReadBuffer(gProjection, true)
		.finish();

		// copy result from device to host
		gProjection.getBuffer().rewind();
		gProjection.getBuffer().get(projection);
		gProjection.getBuffer().rewind();

		FloatProcessor fl = new FloatProcessor(width, height, projection, null);
		
		// TODO: Normalization is never considered in the backprojectors, 
		// 		 thus, iteratively applying forward and backward projections
		//		 would yield to a scaling issue!
		// conversion from [g*mm/cm^3] = [g*0.1cm/cm^3] to [g/cm^2]
		// fl.multiply(1.0 / 10);
		
		return fl;
	}

	/**
	 * Starts projection and returns Projection Data, as ImagePlus
	 * @return the projection stack
	 */
	public Grid3D project(){

		ImagePlus image = null;

		try {
			ImageStack stack = new ImageStack(width, height);
			for (int i = 0; i < nrProj; i++){
				stack.addSlice("Projection " + i, project(i).duplicate());
			}
			if (largeVolumeMode){
				// play it again, Sam!
				while (currentStep < nSteps) {
					currentStep++;

					if (currentStep == nSteps) break;
					if (debug) System.out.println("Processing step " + currentStep + " of " + nSteps);
					copyVolumeToCard();
					for (int i = 0; i < nrProj; i++){
						ImageUtil.addProcessors(stack.getProcessor(i+1), project(i));
						//stack.addSlice("slice " + i, project(i).duplicate());
					}
				}
			}
			image = new ImagePlus();
			image.setStack("Forward Projection", stack);
			image.getCalibration().pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX();
			image.getCalibration().pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			unload();
		}
		//unload();
		//image.show();
		return ImageUtil.wrapImagePlus(image);
	}

	private void copyVolumeToCard(){
		// write the volume from ImagePlus to a float array.
		for (int k = 0; k < subVolumeZ; k++){
			int index = ((nSteps - currentStep -1)*subVolumeZ) + k;
			if (index < tex.getSize()[2]){
				Grid2D currentSlice = tex.getSubGrid(index);
				for (int i = 0; i< volumeSize[0]; i++){
					for(int j = 0; j< volumeSize[1]; j++){
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
									// Note that we must flip the coordinates of the volume slices in order to be compatible with the OpenCL Backprojector...
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
			gVolumeEdgeMaxPoint = context.createFloatBuffer(volumeEdgeMaxPoint.length, Mem.READ_ONLY);
		}
		if (gVolumeEdgeMinPoint == null){
			gVolumeEdgeMinPoint = context.createFloatBuffer(volumeEdgeMinPoint.length, Mem.READ_ONLY);
		}

		//} else {
		gVolumeEdgeMaxPoint.getBuffer().put(volumeEdgeMaxPoint);
		gVolumeEdgeMinPoint.getBuffer().put(volumeEdgeMinPoint);
		//}  
		gVolumeEdgeMaxPoint.getBuffer().rewind();
		gVolumeEdgeMinPoint.getBuffer().rewind();

		if (gTex3D == null) {
			// Create the 3D array that will contain the volume data
			// and will be accessed via the 3D texture
			CLBuffer<FloatBuffer> hvolumeBuffer = context.createFloatBuffer(h_volume.length, Mem.READ_ONLY);
			hvolumeBuffer.getBuffer().put(h_volume);
			hvolumeBuffer.getBuffer().rewind();

			// set the texture
			CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
			gTex3D = context.createImage3d(hvolumeBuffer.getBuffer(), (int)volumeSize[0], (int)volumeSize[1], subVolumeZ, format, Mem.READ_ONLY);
			hvolumeBuffer.release();
		}
		
		prepareAllProjections();
		
	}

	public void configure(Grid3D img, int[] projSize, double[] projSpacing, Projection[] pMat){
		
		this.setTex3D(img);
		
		projectionMatrices = pMat;
		width = projSize[0];
		height = projSize[1];
		nrProj = pMat.length;
		
		volumeEdgeMinPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMinPoint[i] = (float) (-0.5 + CONRAD.SMALL_VALUE);
		}

		if (debug) System.out.println("Projection Matrices: " + nrProj);
		projection = new float[width * height];
		configured = true;
	}

	/**
	 * returns whether the projector was already configured or not.
	 */
	public boolean isConfigured() {
		return configured;
	}



}