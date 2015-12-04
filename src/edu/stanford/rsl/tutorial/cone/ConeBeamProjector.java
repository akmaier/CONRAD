package edu.stanford.rsl.tutorial.cone;


import java.io.IOException;
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

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.opencl.TestOpenCL;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

public class ConeBeamProjector {

	final boolean debug = false;
	final boolean verbose = false;

	static int bpBlockSize[] = {16, 16};

	//opencl variables
	protected CLContext context;
	protected CLDevice device;
	private CLProgram program;
	private CLImageFormat format;
	private CLBuffer<FloatBuffer> gVolumeEdgeMaxPoint = null;
	private CLBuffer<FloatBuffer> gVolumeEdgeMinPoint = null;
	private CLBuffer<FloatBuffer> gVoxelElementSize = null;
	protected CLBuffer<FloatBuffer> gInvARmatrix = null;
	protected CLBuffer<FloatBuffer> gSrcPoint = null;
	private CLBuffer<FloatBuffer> sinogram = null;
	private CLImage3d<FloatBuffer> imageGrid = null;
	protected CLCommandQueue queue = null;
	private CLKernel kernelFunction;
	// Length of arrays to process
	int localWorkSize;
	int globalWorkSizeU;
	int globalWorkSizeV; 
	
	//imaging variables
	private int width;
	private int height;
	protected Trajectory geometry;
	private int currentStep = 0;
	private float [] voxelSize = null;
	private float [] volumeSize = null;
	private float [] volumeEdgeMinPoint = null;
	private float [] volumeEdgeMaxPoint = null;
	// buffer for 3D volume:
	private int subVolumeZ;

	public ConeBeamProjector() {
		configure();
		initCL();
	}
	
	private void initCL(){
		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		program = null;
		// initialize the program
		if (program==null || !program.getContext().equals(context)){
			try {
				program = context.createProgram(TestOpenCL.class.getResourceAsStream("projectCL.cl")).build();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// create image from input grid
		format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		
		// Length of arrays to process
		localWorkSize = Math.min(device.getMaxWorkGroupSize(), 16); // Local work size dimensions
		globalWorkSizeU = OpenCLUtil.roundUp(localWorkSize, width); // rounded up to the nearest multiple of localWorkSize
		globalWorkSizeV = OpenCLUtil.roundUp(localWorkSize, height); // rounded up to the nearest multiple of localWorkSize
		
		queue = device.createCommandQueue();
		kernelFunction = program.createCLKernel("projectKernel");
		setEdgeMaxima();
		prepareAllProjections();

		gVoxelElementSize = context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
		gVoxelElementSize.getBuffer().put(voxelSize);
		gVoxelElementSize.getBuffer().rewind();
		
		queue
		.putWriteBuffer(gVoxelElementSize,true)
		.putWriteBuffer(gVolumeEdgeMinPoint, true)
		.putWriteBuffer(gVolumeEdgeMaxPoint, true)
		.finish();
		
	}

	public Grid2D projectPixelDriven(Grid3D grid, int projIdx) {
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		int maxV = geometry.getDetectorHeight();
		int maxU = geometry.getDetectorWidth();
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		if(projIdx+1 > maxProjs || 0 > projIdx){
			System.err.println("ConeBeamProjector: Invalid projection index");
			return null;
		}
		Grid2D sino = new Grid2D(maxU,maxV); //
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		for (int x = 0; x < imgSizeX - 1; x++) {
			double xTrans = x * spacingX - originX;
			for (int y = 0; y < imgSizeY - 1; y++) {
				double yTrans = y * spacingY - originY;
				for (int z = 0; z < imgSizeZ - 1; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z * spacingZ - originZ, 1);
					// for(int p = 0; p < maxProjs; p++) {
					int p = projIdx; //
					SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
					double coordU = point2d.getElement(0) / point2d.getElement(2);
					double coordV = point2d.getElement(1) / point2d.getElement(2);
					if (coordU >= maxU - 1 || coordV >= maxV - 1 || coordU <= 0 || coordV <= 0)
						continue;
					float val = grid.getAtIndex(x, y, z);
					InterpolationOperators.addInterpolateLinear(sino, coordU, coordV, val); //
					// }
				}
			}
		}

		return sino;
	}


	public Grid3D projectPixelDriven(Grid3D grid) {
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		int maxV = geometry.getDetectorHeight();
		int maxU = geometry.getDetectorWidth();
		int imgSizeX = geometry.getReconDimensionX();
		int imgSizeY = geometry.getReconDimensionY();
		int imgSizeZ = geometry.getReconDimensionZ();
		Projection[] projMats = geometry.getProjectionMatrices();
		int maxProjs = geometry.getProjectionStackSize();
		Grid3D sino = new Grid3D(maxU,maxV,maxProjs);
		double spacingX = geometry.getVoxelSpacingX();
		double spacingY = geometry.getVoxelSpacingY();
		double spacingZ = geometry.getVoxelSpacingZ();
		double originX = -geometry.getOriginX();
		double originY = -geometry.getOriginY();
		double originZ = -geometry.getOriginZ();
		for(int x = 0; x < imgSizeX-1; x++) {
			double xTrans = x*spacingX-originX;
			for(int y = 0; y < imgSizeY-1 ; y++) {
				double yTrans = y*spacingY-originY;
				for(int z = 0; z < imgSizeZ-1; z++) {
					SimpleVector point3d = new SimpleVector(xTrans, yTrans, z*spacingZ-originZ, 1);
					for(int p = 0; p < maxProjs; p++) {
						SimpleVector point2d = SimpleOperators.multiply(projMats[p].computeP(), point3d);
						double coordU = point2d.getElement(0) / point2d.getElement(2);
						double coordV = point2d.getElement(1) / point2d.getElement(2);
						if (coordU >= maxU-1 || coordV>= maxV -1 || coordU <=0 || coordV <= 0)
							continue;
						float val = grid.getAtIndex(x, y, z);
						InterpolationOperators.addInterpolateLinear(sino, coordU, coordV, p, val);
					}
				}
			}
		}

		return sino;
	}


	private void setEdgeMaxima(){

		int test = currentStep;
		volumeEdgeMaxPoint[2] = (float) ((test * subVolumeZ) + subVolumeZ -0.5 - CONRAD.SMALL_VALUE);
		volumeEdgeMinPoint[2] = (float) ((test * subVolumeZ) -0.5 - CONRAD.SMALL_VALUE);
		
		gVolumeEdgeMaxPoint = context.createFloatBuffer(volumeEdgeMaxPoint.length, Mem.READ_ONLY);
		gVolumeEdgeMinPoint = context.createFloatBuffer(volumeEdgeMinPoint.length, Mem.READ_ONLY);

		gVolumeEdgeMaxPoint.getBuffer().put(volumeEdgeMaxPoint);
		gVolumeEdgeMinPoint.getBuffer().put(volumeEdgeMinPoint);

		gVolumeEdgeMaxPoint.getBuffer().rewind();
		gVolumeEdgeMinPoint.getBuffer().rewind();
	}

	protected void prepareAllProjections(){

		float [] cann = new float[3*4];
		float [] invAR = new float[3*3];
		float [] srcP = new float[3];

		//if (gInvARmatrix == null)
		gInvARmatrix = context.createFloatBuffer(invAR.length*geometry.getNumProjectionMatrices(), Mem.READ_ONLY);
		//if (gSrcPoint == null)
		gSrcPoint = context.createFloatBuffer(srcP.length*geometry.getNumProjectionMatrices(), Mem.READ_ONLY);

		for (int i=0; i < geometry.getNumProjectionMatrices(); ++i){
			SimpleMatrix projMat = geometry.getProjectionMatrix(i).computeP();
			double [][] mat = new double [3][4];
			projMat.copyTo(mat);
			computeCanonicalProjectionMatrix(cann, invAR, srcP, new Jama.Matrix(mat));

			gInvARmatrix.getBuffer().put(invAR);
			gSrcPoint.getBuffer().put(srcP);
		}

		gInvARmatrix.getBuffer().rewind();
		gSrcPoint.getBuffer().rewind();

		queue
		.putWriteBuffer(gSrcPoint, true)
		.putWriteBuffer(gInvARmatrix, true)
		.finish();
	}

	public void computeCanonicalProjectionMatrix(float [] canonicalProjMatrix, float [] invARmatrix, float [] srcPoint, Jama.Matrix projectionMatrix){

		double [] du = {geometry.getPixelDimensionX(), 0};
		double [] dv = {0, geometry.getPixelDimensionY()};


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

	public static Jama.Matrix computeSrcPt(Jama.Matrix projectionMatrix, Jama.Matrix invertedProjMatrix) {
		Jama.Matrix at = projectionMatrix.getMatrix(0, 2, 3, 3);
		//at = at.times(-1.0);
		return invertedProjMatrix.times(at);
	}

	public void configure(){
		Configuration.loadConfiguration();
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		voxelSize = new float [3];
		volumeSize = new float [3];

		voxelSize[0] = (float) geometry.getVoxelSpacingX();
		voxelSize[1] = (float) geometry.getVoxelSpacingY();
		voxelSize[2] = (float) geometry.getVoxelSpacingZ();

		volumeSize[0] = geometry.getReconDimensionX();
		volumeSize[1] = geometry.getReconDimensionY();
		volumeSize[2] = geometry.getReconDimensionZ();

		volumeEdgeMinPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMinPoint[i] = (float) (-0.5 + CONRAD.SMALL_VALUE);
		}
		volumeEdgeMaxPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMaxPoint[i] = (float) (volumeSize[i] -0.5 - CONRAD.SMALL_VALUE);
		}
		width = geometry.getDetectorWidth();
		height = geometry.getDetectorHeight();

		subVolumeZ = (int) volumeSize[2];
		
		if (debug) System.out.println("Projection Matrices: " + geometry.getNumProjectionMatrices());
	}

	public void unload(){

		if(sinogram != null && !sinogram.isReleased())
			sinogram.release();
		if(imageGrid != null && !imageGrid.isReleased())
			imageGrid.release();

	}

	public int getMaxProjections() {
		return geometry.getProjectionStackSize();
	}

	public void projectRayDrivenCL(OpenCLGrid2D[] sinoCL, OpenCLGrid3D gridCL){

		imageGrid = context.createImage3d(gridCL.getDelegate().getCLBuffer().getBuffer(), (int)volumeSize[0], (int)volumeSize[1], (int)volumeSize[2],format, Mem.READ_ONLY);

		queue
		.putCopyBufferToImage(gridCL.getDelegate().getCLBuffer(), imageGrid)
		.finish();

		for(int p = 0; p < geometry.getProjectionStackSize(); p++) {
	
			kernelFunction
			.putArg(sinoCL[p].getDelegate().getCLBuffer())
			.putArg(width)
			.putArg(height)
			.putArg(1.f)
			.putArg(imageGrid)
			.putArg(gVoxelElementSize)
			.putArg(gVolumeEdgeMinPoint)
			.putArg(gVolumeEdgeMaxPoint)
			.putArg(gSrcPoint)
			.putArg(gInvARmatrix)
			.putArg(p);

			queue
			.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSizeU, globalWorkSizeV,localWorkSize, localWorkSize)
			.finish();

			kernelFunction.rewind();
			sinoCL[p].getDelegate().notifyDeviceChange();
			sinoCL[p].getDelegate().prepareForHostOperation();
		}
		
		kernelFunction.release();
		queue.release();
	}
	//2d method
	/**
	 * loads the actual OpenCL kernel and performs the projection
	 * @param projectionNumber the projection number.
	 * @return the image as image processor
	 */

	public void projectRayDrivenCL(OpenCLGrid2D sinoCL, OpenCLGrid3D gridCL, int projIdx){
		
		imageGrid = context.createImage3d(gridCL.getDelegate().getCLBuffer().getBuffer(), (int)gridCL.getSize()[0], (int)gridCL.getSize()[1], (int)gridCL.getSize()[2],format, Mem.READ_ONLY);

		queue
		.putCopyBufferToImage(gridCL.getDelegate().getCLBuffer(), imageGrid)
		.finish();

		kernelFunction
		.putArg(sinoCL.getDelegate().getCLBuffer())
		.putArg(width)
		.putArg(height)
		.putArg(1.f)
		.putArg(imageGrid)
		.putArg(gVoxelElementSize)
		.putArg(gVolumeEdgeMinPoint)
		.putArg(gVolumeEdgeMaxPoint)
		.putArg(gSrcPoint)
		.putArg(gInvARmatrix)
		.putArg(projIdx);
	
		queue
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSizeU, globalWorkSizeV,localWorkSize, localWorkSize).finish();

		kernelFunction.rewind();
		sinoCL.getDelegate().notifyDeviceChange();
	}

	public Grid2D projectRayDrivenCL(Grid3D grid, int projIdx) throws Exception {
		configure();

		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		OpenCLGrid2D sinoCL = new OpenCLGrid2D(new Grid2D(width,height));
		sinoCL.getDelegate().prepareForDeviceOperation();

		projectRayDrivenCL(sinoCL, gridCL, projIdx);

		gridCL.release();

		Grid2D sino = new Grid2D(sinoCL);
		unload();
		return sino;
	}
	
	public Grid3D projectRayDrivenCL(Grid3D grid) {

		configure();

		OpenCLGrid3D gridCL = new OpenCLGrid3D(grid);
		OpenCLGrid2D [] sinoCL = new OpenCLGrid2D[geometry.getProjectionStackSize()];
		for (int i=0; i < geometry.getProjectionStackSize(); i++) {
			sinoCL[i] = new OpenCLGrid2D(new Grid2D(width,height));
			sinoCL[i].getDelegate().prepareForDeviceOperation();
		}
		
		projectRayDrivenCL(sinoCL, gridCL);
		gridCL.release();

		Grid3D sino = new Grid3D(width,height,geometry.getProjectionStackSize());
		for(int i = 0; i<sinoCL.length; i++){
			sino.setSubGrid(i, new Grid2D(sinoCL[i]));
			sinoCL[i].release();
		}

		unload();
		return sino;

	}

	// calculates the sinogram out of a volume grid at the projectionindex projIdx
	public void fastProjectRayDrivenCL(OpenCLGrid2D sinoCL, OpenCLGrid3D grid, int projIdx) {
		sinoCL.getDelegate().prepareForDeviceOperation();
		grid.getDelegate().prepareForDeviceOperation();
		
		projectRayDrivenCL(sinoCL, grid, projIdx);

		unload();
	}
	
	// calculates the sinogram out of a volume grid at the projectionindex projIdx
	public void fastProjectRayDrivenCL(OpenCLGrid3D sinoCL, OpenCLGrid3D grid) throws Exception {

		OpenCLGrid2D sinoCLBuffer = new OpenCLGrid2D(new Grid2D(width,height));
		sinoCLBuffer.setOrigin(0,0);
		sinoCLBuffer.setSpacing(geometry.getPixelDimensionX(),geometry.getPixelDimensionY());
		sinoCL.getDelegate().prepareForDeviceOperation();

		for(int pIdx = 0; pIdx < geometry.getProjectionStackSize(); pIdx++){
			sinoCLBuffer.getDelegate().prepareForDeviceOperation();
			fastProjectRayDrivenCL(sinoCLBuffer,grid,pIdx);
			sinoCLBuffer.getDelegate().notifyDeviceChange();
			queue.putCopyBuffer(sinoCLBuffer.getDelegate().getCLBuffer(), sinoCL.getDelegate().getCLBuffer(),0,pIdx*(int)sinoCLBuffer.getDelegate().getCLBuffer().getCLSize(),sinoCLBuffer.getDelegate().getCLBuffer().getCLSize(),null).finish();
		}

		sinoCLBuffer.release();
		sinoCL.getDelegate().notifyDeviceChange();

		unload();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
