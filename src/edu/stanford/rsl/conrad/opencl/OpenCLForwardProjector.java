package edu.stanford.rsl.conrad.opencl;

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
import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.trajectories.ProjectionTableFileTrajectory;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;



/**
 * Forward projection expects input of a volumetric phantom scaled to mass density. Projection result {@latex.inline $p(\\vec{x})$} is then the accumulated mass along the ray {@latex.inline $\\vec{x}$} which consists of the line segments {@latex.inline $x_i$} in {@latex.inline $[\\textnormal{cm}]$} with the mass densities {@latex.inline $\\mu_i$} in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^3}]$}.
 * The actual projection is then computed as:<br>
 * {@latex.inline $$p(\\vec{x}) = \\sum_{i} x_i \\cdot \\mu_i$$}<BR>
 * The projection values are then returned in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^2}]$}
 * @author akmaier, berger (refactored from CUDA)
 *
 */
public class OpenCLForwardProjector implements GUIConfigurable, Citeable {

	static int bpBlockSize[] = {16, 16};

	private static boolean debug = true;
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
	private boolean obtainGeometryFromVolume = false;
	private boolean flipProjections = false;

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

	private ImagePlus tex3D = null;
	
	private float [] projection;
	private int width;
	private int height;
	protected int nrProj;

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
		if (obtainGeometryFromVolume){
			Grid3D inputTex = ImageUtil.wrapImagePlus(tex3d);
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
		tex3D = tex3d;
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
					program = context.createProgram(TestOpenCL.class.getResourceAsStream("projectCL.cl")).build();
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

		CLBuffer<FloatBuffer> invAR = OpenCLUtil.getStaticContext().createFloatBuffer(3*3, Mem.READ_ONLY);
		CLBuffer<FloatBuffer> srcP = OpenCLUtil.getStaticContext().createFloatBuffer(3, Mem.READ_ONLY);
		OpenCLForwardProjector clForwardProjector = new OpenCLForwardProjector();
		clForwardProjector.voxelSize = new float [] {0.5f, 0.5f, 0.5f};
		clForwardProjector.volumeSize = new float [] {512f, 512f, 512f};
		try {
			clForwardProjector.configure();
		} catch (Exception e) {
			e.printStackTrace();
		}
		invAR.getBuffer().rewind();
		srcP.getBuffer().rewind();
		clForwardProjector.computeCanonicalProjectionMatrix(invAR, srcP, proj);
		invAR.getBuffer().rewind();
		srcP.getBuffer().rewind();
		while(invAR.getBuffer().hasRemaining())
			System.out.println(invAR.getBuffer().get());
		while(srcP.getBuffer().hasRemaining())
			System.out.println(srcP.getBuffer().get());
	}

	/**
	 * release all CL related objects and free memory
	 */
	private void unload(){
		if (commandQueue != null)
			commandQueue.release();
		//release all buffers
		if (gTex3D != null)
			gTex3D.release();
		if (gVolumeEdgeMaxPoint != null)
			gVolumeEdgeMaxPoint.release();
		if (gVolumeEdgeMinPoint != null)
			gVolumeEdgeMinPoint.release();
		if (gVoxelElementSize != null)
			gVoxelElementSize.release();
		if (gInvARmatrix != null)
			gInvARmatrix.release();
		if (gSrcPoint != null)
			gSrcPoint.release();
		if (gProjection != null)
			gProjection.release();
		if (kernelFunction != null)
			kernelFunction.release();
		if (program != null)
			program.release();
		if (context != null)
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
		
		if (flipProjections){
			fl.flipVertical();
		}
		return fl;
	}

	/**
	 * Starts projection and returns Projection Data, as ImagePlus
	 * @return the projection stack
	 */
	public ImagePlus project(){

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
			image.setStack("Forward Projection of " + tex3D.getTitle(), stack);
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


	/**
	 * Start GUI configuration. Reads from global Configuration.
	 */
	@Override
	public void configure() throws Exception {
		obtainGeometryFromVolume = UserUtil.queryBoolean("Try to obtain volume parameters from ImagePlus/Grid3D (Otherwise from configuration)?");
		Configuration config = Configuration.getGlobalConfiguration();
		
		if (!obtainGeometryFromVolume){
			voxelSize = new float [3];
			volumeSize = new float [3];
			origin = new double [3];
			voxelSize[0] = (float) config.getGeometry().getVoxelSpacingX();
			voxelSize[1] = (float) config.getGeometry().getVoxelSpacingY();
			voxelSize[2] = (float) config.getGeometry().getVoxelSpacingZ();
			volumeSize[0] = config.getGeometry().getReconDimensionX();
			volumeSize[1] = config.getGeometry().getReconDimensionY();
			volumeSize[2] = config.getGeometry().getReconDimensionZ();
			origin[0] = (float) config.getGeometry().getOriginX();
			origin[1] = (float) config.getGeometry().getOriginY();
			origin[2] = (float) config.getGeometry().getOriginZ();
			volumeEdgeMaxPoint = new float[3];
			for (int i=0; i < 3; i ++){
				volumeEdgeMaxPoint[i] = (float) (volumeSize[i] -0.5 - CONRAD.SMALL_VALUE);
			}
		}
		else{
			voxelSize =  null;
			volumeSize = null;
			origin = null;
		}
		projectionMatrices = config.getGeometry().getProjectionMatrices();
		width = config.getGeometry().getDetectorWidth();
		height = config.getGeometry().getDetectorHeight();
		nrProj = config.getGeometry().getNumProjectionMatrices();
		flipProjections = (config.getGeometry() instanceof ProjectionTableFileTrajectory);
		
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
	@Override
	public boolean isConfigured() {
		return configured;
	}

	/**
	 * Returns a reference to literature describing this algorithm in Bibtex format
	 */
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

	/**
	 * Returns a reference to literature describing this algorithm in Medline
	 */
	@Override
	public String getMedlineCitation() {
		return "Galigekere RR, Wiesent K, and Holdsworth DW. Cone-Beam Reprojection Using Projection-Matrices. IEEE Transactions on Medical Imaging 22(10):1202-14 2003.";
	}



}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
