package edu.stanford.rsl.conrad.opencl;

import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.io.IOException;
import java.nio.FloatBuffer;
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
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class OpenCLBackProjector extends VOIBasedReconstructionFilter implements Runnable, Citeable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8615490043940236889L;

	protected static int bpBlockSize[] = {32, 16};

	private static boolean debug = true;
	/**
	 * The OpenCL context
	 */
	protected CLContext context;

	/**
	 * The OpenCL program
	 */
	protected CLProgram program;

	/**
	 * The OpenCL device
	 */
	protected CLDevice device;

	/**
	 * The OpenCL kernel function binding
	 */
	protected CLKernel kernelFunction;

	/**
	 * The OpenCL command queue
	 */
	protected CLCommandQueue commandQueue;	

	/**
	 * The 2D projection texture reference
	 */
	protected CLImage2d<FloatBuffer> projectionTex = null;

	/**
	 * The volume data that is to be reconstructed
	 */
	protected float h_volume[];


	/**
	 * The global variable of the module which stores the
	 * view matrix.
	 */
	protected CLBuffer<FloatBuffer> projectionMatrix = null;
	protected CLBuffer<FloatBuffer> volumePointer = null;
	private CLBuffer<FloatBuffer> projectionArray = null;

	protected ImageGridBuffer projections;
	protected ArrayList<Integer> projectionsAvailable;
	protected ArrayList<Integer> projectionsDone;
	protected boolean largeVolumeMode = false;
	private int nSteps = 1;
	private long subVolumeZ = 0;

	private boolean initialized = false;

	public OpenCLBackProjector () {
		super();
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		projectionMatrix = null;
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

	protected void createProgram() throws IOException{
		// initialize the program
		if (program==null || !program.getContext().equals(this.context)){
			program = context.createProgram(TestOpenCL.class.getResourceAsStream("backprojectCL.cl")).build();
		}
	}

	protected void init(){
		if (!initialized) {
			largeVolumeMode = false;

			long reconDimensionX = getGeometry().getReconDimensionX();
			long reconDimensionY = getGeometry().getReconDimensionY();
			long reconDimensionZ = getGeometry().getReconDimensionZ();
			projections = new ImageGridBuffer();
			projectionsAvailable = new ArrayList<Integer>();
			projectionsDone = new ArrayList<Integer>();

			// Initialize JOCL.
			context = OpenCLUtil.createContext();

			try {
				// get the fastest device
				device = context.getMaxFlopsDevice();
				// create the command queue
				commandQueue = device.createCommandQueue();

				createProgram();

			} catch (Exception e) {
				if (commandQueue != null)
					commandQueue.release();
				if (kernelFunction != null)
					kernelFunction.release();
				if (program != null)
					program.release();
				// destory context
				if (context != null)
					context.release();
				// TODO: handle exception
				e.printStackTrace();
			}

			// (1) check space on device - At the moment we simply use 90% of the overall available memory
			// (2) createFloatBuffer uses a byteBuffer internally --> h_volume.length cannot be > 2^31/4 = 2^31/2^2 = 2^29
			// 	   Thus, 2^29 would already cause a overflow (negative sign) of the integer in the byte buffer! Maximum length is (2^29-1) float or (2^31-4) bytes!
			// Either we are limited by the maximum addressable memory, i.e. (2^31-4) bytes or by the device limit "device.getGlobalMemSize()*0.9"
			long availableMemory =  Math.min((long)(device.getGlobalMemSize() * 0.9),2147483647);
			long requiredMemory = (long)(((
					((double) reconDimensionX) * reconDimensionY * ((double) reconDimensionZ) * 4) 
					+ (((double)Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()) * Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth() * 4)));
			if (debug) {
				CONRAD.log("Total available Memory on OpenCL card:" + availableMemory);
				CONRAD.log("Required Memory on OpenCL card:" + requiredMemory);
			}
			if (requiredMemory > availableMemory){
				nSteps = (int)OpenCLUtil.iDivUp (requiredMemory, availableMemory);
				if (debug) CONRAD.log("Switching to large volume mode with nSteps = " + nSteps);
				largeVolumeMode = true;
			}

			// create the computing kernel
			kernelFunction = program.createCLKernel("backprojectKernel");

			// create the reconstruction volume;
			// createFloatBuffer uses a byteBuffer internally --> h_volume.length cannot be > 2^31/4 = 2^31/2^2 = 2^29
			// Thus, 2^29 would already cause a overflow (negative sign) of the integer in the byte buffer! Maximum length is (2^29-1)
			long memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * 4;
			if (largeVolumeMode){
				subVolumeZ = OpenCLUtil.iDivUp((int)reconDimensionZ, nSteps);
				if(debug) CONRAD.log("SubVolumeZ: " + subVolumeZ);
				h_volume = new float[(int)(reconDimensionX * reconDimensionY * subVolumeZ)];
				memorysize = reconDimensionX * reconDimensionY * subVolumeZ * 4;
				if(debug)CONRAD.log("Memory: " + memorysize);
			} else {
				h_volume = new float[(int)(reconDimensionX * reconDimensionY * reconDimensionZ)];	
			}

			// copy volume to device
			volumePointer = context.createFloatBuffer(h_volume.length, Mem.WRITE_ONLY);
			volumePointer.getBuffer().put(h_volume);
			volumePointer.getBuffer().rewind();

			commandQueue.
			putWriteBuffer(volumePointer, true).
			finish();

			initialized = true;
		}

	}

	protected synchronized void unload(){
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
				CONRAD.log("Check ProjectionVolume. It seems null.");
			}


			h_volume = null;


			// free memory on device
			commandQueue.release();

			if (projectionTex != null)
				projectionTex.release();
			if (projectionMatrix != null)
				projectionMatrix.release();
			if (projectionArray != null)
				projectionArray.release();
			if (volumePointer != null)
				volumePointer.release();

			kernelFunction.release();
			program.release();
			// destory context
			context.release();


			commandQueue = null;
			projectionArray = null;
			projectionMatrix = null;
			projectionTex = null;
			volumePointer = null;
			kernelFunction = null;
			program = null;
			context = null;


			initialized = false;
		}
	}

	protected synchronized void initProjectionMatrix(int projectionNumber){
		// load projection Matrix for current Projection.
		if (getGeometry().getProjectionMatrix(projectionNumber)== null) {
			CONRAD.log("No geometry found for projection " +projectionNumber + ". Skipping.");
			return;
		}
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
		commandQueue.putWriteBuffer(projectionMatrix, true);
		//System.out.println("Uploading matrix " + projectionNumber);
	}

	protected synchronized void initProjectionData(Grid2D projection){
		initialize(projection);
		if (projection != null){ 
			if (projectionArray == null) {
				// Create the array that will contain the projection data. 
				projectionArray = context.createFloatBuffer(projection.getWidth()*projection.getHeight(), Mem.READ_ONLY);
			}
			
			// Copy the projection data to the array 
			projectionArray.getBuffer().put(projection.getBuffer());
			projectionArray.getBuffer().rewind();

			if(projectionTex != null && !projectionTex.isReleased()){			
				projectionTex.release();
			}
			
			// set the texture
			CLImageFormat format = new CLImageFormat(ChannelOrder.INTENSITY, ChannelType.FLOAT);
			projectionTex = context.createImage2d(projectionArray.getBuffer(), projection.getWidth(), projection.getHeight(), format, Mem.READ_ONLY);
			//projectionArray.release();

		} else {
			CONRAD.log("Projection was null!!");
		}
	}

	@Override
	public String getName() {
		return "OpenCL Backprojector";
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

	protected synchronized void projectSingleProjection(int projectionNumber, int dimz){
		// load projection matrix
		initProjectionMatrix(projectionNumber);

		// Correct for constant part of distance weighting + For angular sampling
		double D =  getGeometry().getSourceToDetectorDistance();
		float projectionMultiplier = (float)(10 * D*D * 2* Math.PI * getGeometry().getPixelDimensionX() / getGeometry().getNumProjectionMatrices());

		initProjectionData(projections.get(projectionNumber));
		//System.out.println("Uploading projection " + projectionNumber);
		if (!largeVolumeMode) {
			projections.remove(projectionNumber);
		}
		// backproject for each slice
		// OpenCL Grids are only two dimensional!
		int reconDimensionZ = dimz;
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();

		// write kernel parameters
		kernelFunction.rewind();
		kernelFunction
		.putArg(volumePointer)
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
		.putArg(projectionMatrix)
		.putArg(projectionMultiplier);

		
		int[] realLocalSize = new int[2];
		realLocalSize[0] = Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]);
		realLocalSize[1] = Math.max(1, Math.min(device.getMaxWorkGroupSize()/realLocalSize[0], bpBlockSize[1]));
		
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {getGeometry().getReconDimensionX(), getGeometry().getReconDimensionY()}; 
		if ((globalWorkSize[0] % realLocalSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / realLocalSize[0]) + 1) * realLocalSize[0];
		}
		if ((globalWorkSize[1] % realLocalSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / realLocalSize[1]) + 1) * realLocalSize[1];
		}

		// Call the OpenCL kernel, writing the results into the volume which is pointed at
		commandQueue
		.putWriteImage(projectionTex, true)
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		//.finish()
		//.putReadBuffer(dOut, true)
		.finish();
	}

	public void loadInputQueue(Grid3D input) throws IOException {
		ImageGridBuffer buf = new ImageGridBuffer();
		buf.set(input);
		inputQueue = buf;
		projections = buf;
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
			CONRAD.log("large Volume " + largeVolumeMode);
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
				reconDimensionZ = (int)subVolumeZ;
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
							CONRAD.log("Backprojection of projection " + p + " was not successful.");
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
							int index = (n*(int)subVolumeZ) + k;
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
		if (debug) CONRAD.log("Unloaded");
	}

	private synchronized void workOnProjectionData(){
		if (projectionsAvailable.size() > 0){
			Integer current = projectionsAvailable.get(0);
			projectionsAvailable.remove(0);
			projectSingleProjection(current.intValue(),  
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

	public Grid3D reconstructCompleteQueue(){
		init();
		int n = inputQueue.size();
		for (int i = 0; i < n; i++){
			backproject(inputQueue.get(i), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();

		//projectionVolume.show();
		return projectionVolume;


	}

	public void reconstructOffline(ImagePlus imagePlus) throws Exception {
		ImagePlusDataSink sink = new ImagePlusDataSink();
		configure();
		init();
		for (int i = 0; i < imagePlus.getStackSize(); i++){
			backproject(ImageUtil.wrapImageProcessor(imagePlus.getStack().getProcessor(i+1)), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();
		//int [] size = projectionVolume.getSize();
		//System.out.println(size [0] + " " + size [1] + " " + size[2]);
		for (int k = 0; k < projectionVolume.getSize()[2]; k++){
			FloatProcessor fl = new FloatProcessor(projectionVolume.getSize()[0], projectionVolume.getSize()[1]);
			for (int j = 0; j< projectionVolume.getSize()[1]; j++){
				for (int i = 0; i< projectionVolume.getSize()[0]; i++){
					fl.putPixelValue(i, j, projectionVolume.getAtIndex(i, j, k));
				}
			}
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		sink.close();
		ImagePlus revan = ImageUtil.wrapGrid3D(sink.getResult(), "Reconstruction of " + imagePlus.getTitle());
		revan.setTitle(imagePlus.getTitle() + " reconstructed");
		revan.show();
	}

	@Override
	protected void reconstruct() throws Exception {
		init();
		for (int i = 0; i < nImages; i++){
			backproject(inputQueue.get(i), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();
		int [] size = projectionVolume.getSize();

		for (int k = 0; k < size[2]; k++){
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		sink.close();
	}

	@Override
	public String getToolName(){
		return "OpenCL Backprojector";
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */