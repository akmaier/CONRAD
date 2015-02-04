/*
 * Copyright (C) 2014 Michael Manhart
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.conrad.opencl;

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

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

public class OpenCLForwardProjectorDynamicVolume  implements Citeable {

	/**
	 * forward projector configuration
	 */
	private boolean isInitialized = false;
	private boolean isConfigured = false;
	private float [] voxelSize;
	private int   [] volumeSize;
	private float [] volumeEdgeMinPoint;
	private float [] volumeEdgeMaxPoint;
	private Trajectory g_fwd;
	private Trajectory g_bwd;	
	
	/**
	 * OpenCL
	 */
	static int bpBlockSize[] = {16, 16};
	private CLContext cl_context;
	private CLProgram cl_program;
	private CLDevice cl_device;
	private CLKernel cl_kernelFunction;
	private CLCommandQueue cl_commandQueue;
	private CLImage3d<FloatBuffer> cl_VolumeTexture3D = null;
	private CLBuffer<FloatBuffer>  cl_VolumeEdgeMaxPoint = null;
	private CLBuffer<FloatBuffer>  cl_VolumeEdgeMinPoint = null;
	private CLBuffer<FloatBuffer>  cl_VoxelElementSize = null;
	private CLBuffer<FloatBuffer>  cl_InvARmatrixFwd = null;
	private CLBuffer<FloatBuffer>  cl_SrcPointFwd = null;
	private CLBuffer<FloatBuffer>  cl_InvARmatrixBwd = null;
	private CLBuffer<FloatBuffer>  cl_SrcPointBwd = null;	
	private CLBuffer<FloatBuffer>  cl_Projection = null;
	
	
	/**
	 * configuration of the projection and volume geometries 
	 */
	public void configure(Trajectory g_fwd, Trajectory g_bwd, int[] volumeSize, float[] voxelSize) {
		this.voxelSize = new float [3];
		this.volumeSize = new int [4];

		this.voxelSize[0] = voxelSize[0];
		this.voxelSize[1] = voxelSize[1];
		this.voxelSize[2] = voxelSize[2];
		this.volumeSize[0] = volumeSize[0];
		this.volumeSize[1] = volumeSize[1];
		this.volumeSize[2] = volumeSize[2];
		this.volumeSize[3] = volumeSize[0]*volumeSize[1]*volumeSize[2];
		volumeEdgeMinPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMinPoint[i] = (float) (-0.5 + CONRAD.SMALL_VALUE);
		}
		volumeEdgeMaxPoint = new float[3];
		for (int i=0; i < 3; i ++){
			volumeEdgeMaxPoint[i] = (float) (volumeSize[i] -0.5 - CONRAD.SMALL_VALUE);
		}
		this.g_fwd = g_fwd;
		this.g_bwd = g_bwd;

		isConfigured = true;
	}	
		
	/**
	 * Initiates communication with the graphics card.
	 */
	private void initialize() {
		if(!isConfigured) {
			System.err.println("OpenCLForwardProjectorDynamicVolume.initialize(): Need to configure before initalization!");
			return;
		}
		
		if (!isInitialized) {
			// Initialize JOCL.
			cl_context = OpenCLUtil.createContext();
			try {
				// get the fastest device
				cl_device = cl_context.getMaxFlopsDevice();
				// create the command queue
				cl_commandQueue = cl_device.createCommandQueue();
				// initialize the program
				if (cl_program == null || !cl_program.getContext().equals(this.cl_context)){
					cl_program = cl_context.createProgram(TestOpenCL.class.getResourceAsStream("projectCL.cl")).build();
				}
				// create the computing kernel
				cl_kernelFunction = cl_program.createCLKernel("projectKernel");

				// volume properties
				cl_VoxelElementSize = cl_context.createFloatBuffer(voxelSize.length, Mem.READ_ONLY);
				cl_VoxelElementSize.getBuffer().put(voxelSize);
				cl_VoxelElementSize.getBuffer().rewind();
				cl_VolumeEdgeMinPoint = cl_context.createFloatBuffer(volumeEdgeMinPoint.length, Mem.READ_ONLY);
				cl_VolumeEdgeMinPoint.getBuffer().put(volumeEdgeMinPoint);
				cl_VolumeEdgeMinPoint.getBuffer().rewind();
				cl_VolumeEdgeMaxPoint = cl_context.createFloatBuffer(volumeEdgeMaxPoint.length, Mem.READ_ONLY);
				cl_VolumeEdgeMaxPoint.getBuffer().put(volumeEdgeMaxPoint);
				cl_VolumeEdgeMaxPoint.getBuffer().rewind();				

				
				
				// projection memory
				cl_Projection = cl_context.createFloatBuffer(g_fwd.getDetectorWidth()*g_fwd.getDetectorHeight(), Mem.WRITE_ONLY);
									
				cl_commandQueue
				.putWriteBuffer(cl_VoxelElementSize,true)
				.putWriteBuffer(cl_VolumeEdgeMinPoint, true)
				.putWriteBuffer(cl_VolumeEdgeMaxPoint, true)
				.finish();
				
			} catch (IOException e) {
				// TODO: handle exception
				e.printStackTrace();
			}
			cl_InvARmatrixFwd = cl_context.createFloatBuffer(9*g_fwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			cl_SrcPointFwd = cl_context.createFloatBuffer(3*g_fwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			prepareProjectionMatrices(g_fwd, cl_InvARmatrixFwd, cl_SrcPointFwd);
			cl_InvARmatrixBwd = cl_context.createFloatBuffer(9*g_bwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			cl_SrcPointBwd = cl_context.createFloatBuffer(3*g_bwd.getNumProjectionMatrices(), Mem.READ_ONLY);	
			prepareProjectionMatrices(g_bwd, cl_InvARmatrixBwd, cl_SrcPointBwd);
			
			isInitialized = true;
		}

	}	

	
	
	/**
	 * Compute inverted projection matrices and source positions for all projections and write in GPU memory
	 * @param projectionNumber
	 */
	private void prepareProjectionMatrices(Trajectory g, CLBuffer<FloatBuffer> cl_InvARmatrix, CLBuffer<FloatBuffer> cl_SrcPoint){
		float [] cann = new float[3*4];
		float [] invAR = new float[3*3];
		float [] srcP = new float[3];
		
		
		for (int i=0; i < g.getNumProjectionMatrices(); ++i){
			SimpleMatrix projMat = g.getProjectionMatrix(i).computeP();
			double [][] mat = new double [3][4];
			projMat.copyTo(mat);
			computeCanonicalProjectionMatrix(cann, invAR, srcP, new Jama.Matrix(mat));
			cl_InvARmatrix.getBuffer().put(invAR);
			cl_SrcPoint.getBuffer().put(srcP);
		}
		
		cl_InvARmatrix.getBuffer().rewind();
		cl_SrcPoint.getBuffer().rewind();
		
		cl_commandQueue
		.putWriteBuffer(cl_SrcPoint, true)
		.putWriteBuffer(cl_InvARmatrix, true)
		.finish();
	}	
	
	public void setVolume(float[] volumeBuffer) {
		if(!isConfigured) {
			System.err.println("OpenCLForwardProjectorDynamicVolume.setVolume(): Need to configure before forward projection!");
			return;
		}	
		if(volumeBuffer.length != volumeSize[3])
		{
			System.err.println("OpenCLForwardProjectorDynamicVolume.setVolume(): Invalid volume buffer size!");
			return;
		}
		initialize();
		if(cl_VolumeTexture3D != null) {
			cl_VolumeTexture3D.release();
			cl_VolumeTexture3D = null;
		}
		// create 3d texture for projected volumes
		CLBuffer<FloatBuffer> hvolumeBuffer = cl_context.createFloatBuffer(volumeBuffer.length, Mem.READ_ONLY);
		hvolumeBuffer.getBuffer().put(volumeBuffer);
		hvolumeBuffer.getBuffer().rewind();
		CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
		cl_VolumeTexture3D = cl_context.createImage3d(hvolumeBuffer.getBuffer(), volumeSize[0], volumeSize[1], volumeSize[2], format, Mem.READ_ONLY);
		cl_commandQueue
		.putWriteImage(cl_VolumeTexture3D, true)
		.finish();
		hvolumeBuffer.release();
	}

	public void applyForwardProjection(int projectionID, boolean fwd, float[] projection_out, boolean motion) {
		if(!isConfigured) {
			System.err.println("OpenCLForwardProjectorDynamicVolume.applyForwardProjection(): Need to configure before forward projection!");
			return;
		}
		if(cl_VolumeTexture3D == null) {
			System.err.println("OpenCLForwardProjectorDynamicVolume.applyForwardProjection(): Need to set volume before forward projection!");
			return;			
		}
		
		Trajectory g = fwd?g_fwd:g_bwd;
		if (fwd && motion)
		{
			cl_InvARmatrixFwd = cl_context.createFloatBuffer(9*g_fwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			cl_SrcPointFwd = cl_context.createFloatBuffer(3*g_fwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			prepareProjectionMatrices(g_fwd, cl_InvARmatrixFwd, cl_SrcPointFwd);

		} else  if (!fwd && motion)
		{
			cl_InvARmatrixBwd = cl_context.createFloatBuffer(9*g_bwd.getNumProjectionMatrices(), Mem.READ_ONLY);
			cl_SrcPointBwd = cl_context.createFloatBuffer(3*g_bwd.getNumProjectionMatrices(), Mem.READ_ONLY);	
			prepareProjectionMatrices(g_bwd, cl_InvARmatrixBwd, cl_SrcPointBwd);	
		}
		
		int projSize = g.getDetectorWidth()*g.getDetectorHeight();
		if(projection_out.length != projSize)  {
			System.err.println("OpenCLForwardProjectorDynamicVolume.applyForwardProjection(): Wrong projection buffer size!");
			return;
		}
		
		// write kernel parameters
		cl_kernelFunction.rewind();
		cl_kernelFunction
		.putArg(cl_Projection)
		.putArg(g.getDetectorWidth())
		.putArg(g.getDetectorHeight())
		.putArg(0.5f)
		.putArg(cl_VolumeTexture3D)
		.putArg(cl_VoxelElementSize)
		.putArg(cl_VolumeEdgeMinPoint)
		.putArg(cl_VolumeEdgeMaxPoint)
		.putArg(fwd?cl_SrcPointFwd:cl_SrcPointBwd)
		.putArg(fwd?cl_InvARmatrixFwd:cl_InvARmatrixBwd)
		.putArg(projectionID);

		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(cl_device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(cl_device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {g.getDetectorWidth(), g.getDetectorHeight()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}
		
		// add kernel function to the queue
		cl_commandQueue
		.put2DRangeKernel(cl_kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		.finish()
		.putReadBuffer(cl_Projection, true)
		.finish();

		// copy result from device to host
		cl_Projection.getBuffer().rewind();
		cl_Projection.getBuffer().get(projection_out);
		cl_Projection.getBuffer().rewind();		
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

	public void computeCanonicalProjectionMatrix(float[] canonicalProjMatrix, float[] invARmatrix, float[] srcPoint, Jama.Matrix projectionMatrix){

		double [] du = {Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionX(), 0};
		double [] dv = {0, Configuration.getGlobalConfiguration().getGeometry().getPixelDimensionY()};

		du[0] = 1;
		dv[1] = 1;

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
	 * release all CL related objects and free memory
	 */
	public void release(){
		if (cl_commandQueue != null)
			cl_commandQueue.release();
		if (cl_VolumeTexture3D != null) {
			cl_VolumeTexture3D.release();
			cl_VolumeTexture3D = null;
		}
		if (cl_VolumeEdgeMaxPoint != null)
			cl_VolumeEdgeMaxPoint.release();
		if (cl_VolumeEdgeMinPoint != null)
			cl_VolumeEdgeMinPoint.release();
		if (cl_VoxelElementSize != null)
			cl_VoxelElementSize.release();
		if (cl_InvARmatrixFwd != null)
			cl_InvARmatrixFwd.release();
		if (cl_InvARmatrixBwd != null)
			cl_InvARmatrixBwd.release();
		if (cl_SrcPointFwd != null)
			cl_SrcPointFwd.release();
		if (cl_SrcPointBwd != null)
			cl_SrcPointBwd.release();	
		if (cl_Projection != null)
			cl_Projection.release();
		if (cl_kernelFunction != null)
			cl_kernelFunction.release();
		if (cl_program != null)
			cl_program.release();
		if (cl_context != null)
			cl_context.release();
		isConfigured = false;
		isInitialized = false;
	}	
	
	@Override
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

	@Override
	public String getMedlineCitation() {
		return "Galigekere RR, Wiesent K, and Holdsworth DW. Cone-Beam Reprojection Using Projection-Matrices. IEEE Transactions on Medical Imaging 22(10):1202-14 2003.";
	}

}
