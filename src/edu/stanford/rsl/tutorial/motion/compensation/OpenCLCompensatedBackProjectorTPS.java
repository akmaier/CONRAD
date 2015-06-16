package edu.stanford.rsl.tutorial.motion.compensation;

import ij.IJ;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.CLImageFormat.ChannelOrder;
import com.jogamp.opencl.CLImageFormat.ChannelType;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;

/**
 * DO NOT USE IN LARGEVOLUMEMODE
 * @author Marco
 *
 */
public class OpenCLCompensatedBackProjectorTPS extends VOIBasedReconstructionFilter implements Runnable, Citeable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8615490043940236889L;

	private ImageGridBuffer inputQueue = new ImageGridBuffer();

	static int bpBlockSize[] = {32, 16};

	private static boolean debug = false;
	/**
	 * The OpenCL context
	 */
	private CLContext context;

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
	private CLCommandQueue commandQueue;	

	/**
	 * The 2D projection texture reference
	 */
	private CLImage2d<FloatBuffer> projectionTex = null;

	/**
	 * The volume data that is to be reconstructed
	 */
	protected float h_volume[];


	/**
	 * The global variable of the module which stores the
	 * view matrix.
	 */
	private CLBuffer<FloatBuffer> projectionMatrix = null;
	private CLBuffer<IntBuffer> volStride = null;
	private CLBuffer<FloatBuffer> volumePointer = null;
	private CLBuffer<FloatBuffer> projectionArray = null;

	protected ImageGridBuffer projections = new ImageGridBuffer();
	protected ArrayList<Integer> projectionsAvailable;
	protected ArrayList<Integer> projectionsDone;
	private boolean largeVolumeMode = false;
	private int nSteps = 1;
	private int subVolumeZ = 0;

	private boolean initialized = false;

	private float[][] coeff;
	private float[][] pts;
	private float[][] A;
	private float[][] b;

	public OpenCLCompensatedBackProjectorTPS(float[][] coeff, float[][] pts, float[][] A, float[][] b) {
		super();
		this.coeff = coeff;
		this.A = A;
		this.b = b;
		this.pts = pts;

	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		projectionMatrix = null;
		volStride = null;
		volumePointer = null;
		projectionArray = null;
		projections = null;
		projectionsAvailable =null;
		projectionsDone = null;
		h_volume = null;
		initialized = false;

		// JOCL members
		program = null;
		device = null;
		kernelFunction = null;
		commandQueue = null;
		projectionTex = null;
		context = null;
	}

	@Override
	public void configure() throws Exception{
		configured = true;
	}

	protected void init(){
		if (!initialized) {
			largeVolumeMode = false;

			int reconDimensionX = getGeometry().getReconDimensionX();
			int reconDimensionY = getGeometry().getReconDimensionY();
			int reconDimensionZ = getGeometry().getReconDimensionZ();
			projectionsAvailable = new ArrayList<Integer>();
			projectionsDone = new ArrayList<Integer>();

			// Initialize JOCL.
			context = OpenCLUtil.getStaticContext();

			try {
				// get the fastest device
				//device = context.getMaxFlopsDevice();
				// create the command queue
				commandQueue = OpenCLUtil.getStaticCommandQueue();
				device = commandQueue.getDevice();

				// initialize the program
				if (program==null || !program.getContext().equals(this.context)){
					program = context.createProgram(OpenCLCompensatedBackProjector.class.getResourceAsStream("compensatedBackprojectCLTPS.cl")).build();
				}


			} catch (Exception e) {
				if (commandQueue!=null)
					commandQueue.release();
				if (kernelFunction != null)
					kernelFunction.release();
				if (program != null)
					program.release();
			}

			// check space on device:
			long memory = device.getMaxMemAllocSize();
			long availableMemory = (memory);
			long requiredMemory = (long)(((
					((double) reconDimensionX) * reconDimensionY * ((double) reconDimensionZ) * 4) 
					+ (((double)Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()) * Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth() * 4)));
			if (debug) {
				System.out.println("Total available Memory on OpenCL card:" + availableMemory);
				System.out.println("Required Memory on OpenCL card:" + requiredMemory);
			}
			if (requiredMemory > availableMemory){
				nSteps = (int)OpenCLUtil.iDivUp (requiredMemory, availableMemory);
				if (debug) System.out.println("Switching to large volume mode with nSteps = " + nSteps);
				largeVolumeMode = true;
			}
			if (debug) {
				//TODO replace
				/*
				CUdevprop prop = new CUdevprop();
				JCudaDriver.cuDeviceGetProperties(prop, dev);
				System.out.println(prop.toFormattedString());
				 */
			}

			// create the computing kernel
			if(debug)
				kernelFunction = program.createCLKernel("backprojectKernel_returnMotion");
			else
				kernelFunction = program.createCLKernel("backprojectKernel");

			// create the reconstruction volume;
			int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * 4;
			if (largeVolumeMode){
				subVolumeZ = OpenCLUtil.iDivUp(reconDimensionZ, nSteps);
				if(debug) System.out.println("SubVolumeZ: " + subVolumeZ);
				h_volume = new float[reconDimensionX * reconDimensionY * subVolumeZ];
				memorysize = reconDimensionX * reconDimensionY * subVolumeZ * 4;
				if(debug)System.out.println("Memory: " + memorysize);
			} else {
				h_volume = new float[reconDimensionX * reconDimensionY * reconDimensionZ];	
			}

			// compute adapted volume size 
			//    volume size in x = multiple of bpBlockSize[0]
			//    volume size in y = multiple of bpBlockSize[1]

			int adaptedVolSize[] = new int[3];
			if ((reconDimensionX % bpBlockSize[0] ) == 0){
				adaptedVolSize[0] = reconDimensionX;
			} else {
				adaptedVolSize[0] = ((reconDimensionX / bpBlockSize[0]) + 1) * bpBlockSize[0];
			}
			if ((reconDimensionY % bpBlockSize[1] ) == 0){
				adaptedVolSize[1] = reconDimensionY;
			} else {
				adaptedVolSize[1] = ((reconDimensionY / bpBlockSize[1]) + 1) * bpBlockSize[1];
			}
			adaptedVolSize[2] = reconDimensionZ;
			int volStrideHost [] = new int[2];
			// compute volstride and copy it to constant memory
			volStrideHost[0] = adaptedVolSize[0];
			volStrideHost[1] = adaptedVolSize[0] * adaptedVolSize[1];

			// copy volume to device
			volumePointer = context.createFloatBuffer(h_volume.length, Mem.WRITE_ONLY);
			volumePointer.getBuffer().put(h_volume);
			volumePointer.getBuffer().rewind();

			// copy volume stride to device
			volStride = context.createIntBuffer(volStrideHost.length, Mem.READ_ONLY);
			volStride.getBuffer().put(volStrideHost);
			volStride.getBuffer().rewind();

			commandQueue.
			putWriteBuffer(volumePointer, true).
			putWriteBuffer(volStride, true).
			finish();

			initialized = true;
		}

	}

	private synchronized void unload(){
		if (initialized) {

			if ((projectionVolume != null) && (!largeVolumeMode)) {

				commandQueue.putReadBuffer(volumePointer, true).finish();
				volumePointer.getBuffer().rewind();
				volumePointer.getBuffer().get(h_volume);
				volumePointer.getBuffer().rewind();


				int width = projectionVolume.getSize()[0];
				int height = projectionVolume.getSize()[1];
				if (this.useVOImap) {
					for (int k = 0; k < projectionVolume.getSize()[2]; k++){
						for (int j = 0; j < height; j++){
							for (int i = 0; i < width; i++){			
								float value = h_volume[(((height * k) + j) * width) + i];
								if (voiMap[i][j][k]) {
									projectionVolume.setAtIndex(i, j, k, value);
								} else {
									projectionVolume.setAtIndex(i, j, k, 0);
								}
							}
						}
					}
				} else {
					for (int k = 0; k < projectionVolume.getSize()[2]; k++){
						for (int j = 0; j < height; j++){
							for (int i = 0; i < width; i++){			
								float value = h_volume[(((height * k) + j) * width) + i];
								projectionVolume.setAtIndex(i, j, k, value);
							}
						}
					}
				}
			} else {
				System.out.println("Check ProjectionVolume. It seems null.");
			}


			h_volume = null;


			// free memory on device
			commandQueue.release();

			if (projectionTex != null)
				projectionTex.release();
			if (projectionMatrix != null)
				projectionMatrix.release();
			if (volStride != null)
				volStride.release();
			if (projectionArray != null)
				projectionArray.release();
			if (volumePointer != null)
				volumePointer.release();

			kernelFunction.release();
			program.release();


			commandQueue = null;
			projectionArray = null;
			projectionMatrix = null;
			projectionTex = null;
			volStride = null;
			volumePointer = null;
			kernelFunction = null;
			program = null;
			context = null;


			initialized = false;
		}
	}

	private synchronized void initProjectionMatrix(int projectionNumber){
		// load projection Matrix for current Projection.
		SimpleMatrix pMat = getGeometry().getProjectionMatrix(projectionNumber).computeP();
		float [] pMatFloat = new float[pMat.getCols() * pMat.getRows()];
		for (int j = 0; j< pMat.getRows(); j++) {
			for (int i = 0; i< pMat.getCols(); i++) {

				pMatFloat[(j * pMat.getCols()) + i] = (float) pMat.getElement(j, i);
			}
		}

		// Obtain the global pointer to the view matrix from
		// the module
		if (projectionMatrix == null)
			projectionMatrix = context.createFloatBuffer(pMatFloat.length, Mem.READ_ONLY);

		projectionMatrix.getBuffer().put(pMatFloat);
		projectionMatrix.getBuffer().rewind();
		commandQueue.putWriteBuffer(projectionMatrix, true).finish();
	}

	private synchronized void initProjectionData(Grid2D projection){
		initialize(projection);
		if (projection != null){ 
			float [] proj= new float[projection.getWidth() * projection.getHeight()];

			for(int i = 0; i< projection.getWidth(); i++){
				for (int j =0; j < projection.getHeight(); j++){
					proj[(j*projection.getWidth()) + i] = projection.getPixelValue(i, j);
				}
			}

			if (projectionArray == null) {
				// Create the array that will contain the
				// projection data. 
				projectionArray = context.createFloatBuffer(projection.getWidth()*projection.getHeight(), Mem.READ_ONLY);
			}

			// Copy the projection data to the array 
			projectionArray.getBuffer().put(proj);
			projectionArray.getBuffer().rewind();

			// set the texture
			CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
			projectionTex = context.createImage2d(projectionArray.getBuffer(), projection.getWidth(), projection.getHeight(), format, Mem.READ_ONLY);
			//projectionArray.release();

		} else {
			System.out.println("Projection was null!!");
		}
	}


	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Rohkohl08-CCR,\n" +
				"  author = {{Scherl}, H. and {Keck}, B. and {Kowarschik}, M. and {Hornegger}, J.},\n" +
				"  title = {{Fast GPU-Based CT Reconstruction using the Common Unified Device Architecture (CUDA)}},\n" +
				"  booktitle = {{Nuclear Science Symposium, Medical Imaging Conference 2007}},\n" +
				"  publisher = {IEEE},\n" +
				"  volume={6},\n" +
				"  address = {Honolulu, HI, United States},\n" +
				"  year = {2007}\n" +
				"  pages= {4464--4466},\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Scherl H, Keck B, Kowarschik M, Hornegger J. Fast GPU-Based CT Reconstruction using the Common Unified Device Architecture (CUDA). In Nuclear Science Symposium, Medical Imaging Conference Record, IEEE, Honolulu, HI, United States, 2008 6:4464-6.";
	}

	public void waitForResult() {
		OpenCLRun();
	}

	@Override
	public void backproject(Grid2D projection, int projectionNumber)
	{
		appendProjection(projection, projectionNumber);
	}

	private void appendProjection(Grid2D projection, int projectionNumber){	
		projections.add(projection, projectionNumber);
		projectionsAvailable.add(new Integer(projectionNumber));
	}

	private synchronized void projectSingleProjection(int projectionNumber, int dimz){
		// load projection matrix
		initProjectionMatrix(projectionNumber);
		// load projection
		Grid2D projection = projections.get(projectionNumber); 
		initProjectionData(projection);
		
		// Correct for constant part of distance weighting + For angular sampling
		double D =  getGeometry().getSourceToDetectorDistance();
		float projectionMultiplier = (float)(10 * D*D * 2* Math.PI * getGeometry().getPixelDimensionX() / getGeometry().getNumProjectionMatrices());

		if (!largeVolumeMode) {
			//projections.remove(projectionNumber);
		}
		// backproject for each slice
		// OpenCL Grids are only two dimensional!
		int reconDimensionZ = dimz;
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();

		OpenCLGrid3D xdeform = null;
		OpenCLGrid3D ydeform = null;
		OpenCLGrid3D zdeform = null;
		if (debug){
			xdeform = new OpenCLGrid3D(new Grid3D(getGeometry().getReconDimensionX(),getGeometry().getReconDimensionY(),getGeometry().getReconDimensionZ()));
			ydeform = new OpenCLGrid3D(xdeform);
			zdeform = new OpenCLGrid3D(xdeform);
			xdeform.getDelegate().prepareForDeviceOperation();
			ydeform.getDelegate().prepareForDeviceOperation();
			zdeform.getDelegate().prepareForDeviceOperation();
		}

		CLBuffer<FloatBuffer> coeffPtr = context.createFloatBuffer(coeff[projectionNumber].length, Mem.READ_ONLY);
		coeffPtr.getBuffer().put(coeff[projectionNumber]);
		coeffPtr.getBuffer().rewind();
		commandQueue.putWriteBuffer(coeffPtr, true);

		CLBuffer<FloatBuffer> ptsGlobalPtr = context.createFloatBuffer(pts[projectionNumber].length, Mem.READ_ONLY);
		ptsGlobalPtr.getBuffer().put(pts[projectionNumber]);
		ptsGlobalPtr.getBuffer().rewind();
		commandQueue.putWriteBuffer(ptsGlobalPtr, true);
		CLBuffer<FloatBuffer> ptsLocalPtr = context.createFloatBuffer(pts[projectionNumber].length, Mem.READ_WRITE);

		CLBuffer<FloatBuffer> APtr = context.createFloatBuffer(A[projectionNumber].length, Mem.READ_ONLY);
		APtr.getBuffer().put(A[projectionNumber]);
		APtr.getBuffer().rewind();
		commandQueue.putWriteBuffer(APtr, true);
		CLBuffer<FloatBuffer> bPtr = context.createFloatBuffer(b[projectionNumber].length, Mem.READ_ONLY);
		bPtr.getBuffer().put(b[projectionNumber]);
		bPtr.getBuffer().rewind();
		commandQueue.putWriteBuffer(bPtr, true);
		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(volumePointer)
		.putArg(coeffPtr)
		.putNullArg((int)coeffPtr.getCLSize())
		.putArg(ptsGlobalPtr)
		.putNullArg((int)ptsGlobalPtr.getCLSize())
		.putArg(APtr)
		.putArg(bPtr)
		.putArg((int)pts[projectionNumber].length/3)
		.putArg(getGeometry().getReconDimensionX())
		.putArg(getGeometry().getReconDimensionY())
		.putArg(reconDimensionZ)
		.putArg((int) lineOffset)
		.putArg((float) voxelSpacingX)
		.putArg((float) voxelSpacingY)
		.putArg((float) voxelSpacingZ)
		.putArg((float) offsetX)
		.putArg((float) offsetY)
		.putArg((float) offsetZ)
		.putArg(projectionTex)
		.putArg(volStride)
		.putArg(projectionMatrix)
		.putArg(projectionMultiplier);

		if (debug){
			kernelFunction.putArg(xdeform.getDelegate().getCLBuffer())
			.putArg(ydeform.getDelegate().getCLBuffer())
			.putArg(zdeform.getDelegate().getCLBuffer());
		}

		int[] realLocalSize = {Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]), Math.min(device.getMaxWorkGroupSize(),bpBlockSize[1])};
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {getGeometry().getReconDimensionX(), getGeometry().getReconDimensionY()}; 

		// Call the OpenCL kernel, writing the results into the volume which is pointed at
		commandQueue
		.putWriteImage(projectionTex, true)
		.finish()
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		.finish();

		if (debug){
			xdeform.getDelegate().notifyDeviceChange();
			ydeform.getDelegate().notifyDeviceChange();
			zdeform.getDelegate().notifyDeviceChange();

			xdeform.show("Xdeform");
			ydeform.show("Ydeform");
			zdeform.show("Zdeform");

			xdeform.release();
			ydeform.release();
			zdeform.release();
		}

		coeffPtr.release();
		ptsGlobalPtr.release();
		ptsLocalPtr.release();
		APtr.release();
		bPtr.release();
		if (showStatus)
			IJ.showProgress(projectionNumber, Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices());
	}

	public void OpenCLRun() {
		try {
			while (projectionsAvailable.size() > 0) {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				if (showStatus) {
					float status = (float) (1.0 / projections.size());
					if (largeVolumeMode) {
						IJ.showStatus("Streaming Projections to OpenCL Buffer");
					} else {
						IJ.showStatus("Backprojecting with OpenCL");
					}
					IJ.showProgress(status);
				}
				if (!largeVolumeMode) {			
					workOnProjectionData();
				} else {
					checkProjectionData();
				}
			}
			//			System.out.println("large Volume " + largeVolumeMode);
			if (largeVolumeMode){
				// we have collected all projections.
				// now we can reconstruct subvolumes and stich them together.
				int reconDimensionZ = getGeometry().getReconDimensionZ();
				double voxelSpacingX = getGeometry().getVoxelSpacingX();
				double voxelSpacingY = getGeometry().getVoxelSpacingY();
				double voxelSpacingZ = getGeometry().getVoxelSpacingZ();
				useVOImap = false;
				initialize(projections.get(0));
				double originalOffsetZ = offsetZ;
				double originalReconDimZ = reconDimensionZ;
				reconDimensionZ = subVolumeZ;
				int maxProjectionNumber = projections.size();
				float all = nSteps * maxProjectionNumber*2;
				for (int n =0; n < nSteps; n++){ // For each subvolume
					// set all to 0;
					Arrays.fill(h_volume, 0);

					volumePointer.getBuffer().rewind();
					volumePointer.getBuffer().put(h_volume);
					volumePointer.getBuffer().rewind();
					commandQueue.putWriteBuffer(volumePointer, true).finish();

					offsetZ = originalOffsetZ - (reconDimensionZ*voxelSpacingZ*n);
					for (int p = 0; p < maxProjectionNumber; p ++){ // For all projections
						float currentStep = (n*maxProjectionNumber*2) + p;
						if (showStatus) {
							IJ.showStatus("Backprojecting with OpenCL");
							IJ.showProgress(currentStep/all);
						}
						//System.out.println("Current: " + p);
						try {
							projectSingleProjection(p, reconDimensionZ);
						} catch (Exception e){
							System.out.println("Backprojection of projection " + p + " was not successful.");
							e.printStackTrace();
						}
					}
					// Gather volume
					commandQueue.putReadBuffer(volumePointer, true).finish();
					volumePointer.getBuffer().rewind();
					volumePointer.getBuffer().get(h_volume);
					volumePointer.getBuffer().rewind();

					// move data to ImagePlus;
					if (projectionVolume != null) {
						for (int k = 0; k < reconDimensionZ; k++){
							int index = (n*subVolumeZ) + k;
							if (showStatus) {
								float currentStep = (n*maxProjectionNumber*2) + maxProjectionNumber + k;
								IJ.showStatus("Fetching Volume from OpenCL");
								IJ.showProgress(currentStep/all);
							}
							if (index < originalReconDimZ) {
								for (int j = 0; j < projectionVolume.getSize()[1]; j++){
									for (int i = 0; i < projectionVolume.getSize()[0]; i++){
										float value = h_volume[(((projectionVolume.getSize()[1] * k) + j) * projectionVolume.getSize()[0]) + i];
										double[][] voxel = new double [4][1];
										voxel[0][0] = (voxelSpacingX * i) - offsetX;
										voxel[1][0] = (voxelSpacingY * j) - offsetY;
										voxel[2][0] = (voxelSpacingZ * index) - originalOffsetZ;

										// exception for the case "interestedInVolume == null" and largeVolume is enabled 
										if (interestedInVolume == null) {
											projectionVolume.setAtIndex(i, j, index, value);
										} else {
											if (interestedInVolume.contains(voxel[0][0], voxel[1][0], voxel[2][0])) {
												projectionVolume.setAtIndex(i, j, index, value);
											} else {
												projectionVolume.setAtIndex(i, j, index, 0);
											}
										}
									}
								}
							}
						}
					}
				}
			}


		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		if (showStatus) IJ.showProgress(1.0);
		unload();
		if (debug) System.out.println("Unloaded");
	}

	private synchronized void workOnProjectionData(){
		if (projectionsAvailable.size() > 0){
			Integer current = projectionsAvailable.get(0);
			projectionsAvailable.remove(0);
			int p = current.intValue();
			projectSingleProjection(p,  
					getGeometry().getReconDimensionZ());
			projectionsDone.add(current);
		}	
	}

	private synchronized void checkProjectionData(){
		if (projectionsAvailable.size() > 0){
			Integer current = projectionsAvailable.get(0);
			projectionsAvailable.remove(current);
			projectionsDone.add(current);
		}	
	}


	public Grid3D reconstructCL(){
		init();
		int n = inputQueue.size();
		if (showStatus)
			IJ.showStatus(this.getToolName());
		for (int i = 0; i < n; i++){
			backproject(inputQueue.get(i), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();

		//projectionVolume.show();
		return projectionVolume;


	}


	public void loadInputQueue(ImageGridBuffer inp) throws IOException {
		inputQueue = inp;
		projections = inp;
	}

	public float[][] getCoeff() {
		return coeff;
	}

	public void setCoeff(float[][] coeff) {
		this.coeff = coeff;
	}

	public float[][] getPts() {
		return pts;
	}

	public void setPts(float[][] pts) {
		this.pts = pts;
	}

	public float[][] getA() {
		return A;
	}

	public void setA(float[][] a) {
		A = a;
	}

	public float[][] getB() {
		return b;
	}

	public void setB(float[][] b) {
		this.b = b;
	}


}
/*
 * Copyright (C) 2010-2014 Marco B�gel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */