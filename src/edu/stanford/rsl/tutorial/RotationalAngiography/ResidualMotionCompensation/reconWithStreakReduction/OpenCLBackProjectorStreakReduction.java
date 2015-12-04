package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.reconWithStreakReduction;


import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

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
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.opencl.TestOpenCL;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class OpenCLBackProjectorStreakReduction extends VOIBasedReconstructionFilter implements Runnable, Citeable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8615490043940236889L;

	private boolean forceSmallVolume = false;

	static int bpBlockSize[] = {32, 16};

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
	 * The 2D projection texture reference
	 */
	private CLImage2d<FloatBuffer> projectionTex = null;

	/**
	 * The volume data that is to be reconstructed
	 */
	protected float h_volume[];
	
	private int gat_ign = 3;


	/**
	 * The global variable of the module which stores the
	 * view matrix.
	 */
	protected CLBuffer<FloatBuffer> projectionMatrix = null;
	private CLBuffer<IntBuffer> volStride = null;
	private CLBuffer<FloatBuffer> volumePointer = null;
	private CLBuffer<FloatBuffer> projectionArray = null;
	private CLBuffer<FloatBuffer> destBuffer = null;

	protected ImageGridBuffer projections;
	protected ArrayList<Integer> projectionsAvailable;
	protected ArrayList<Integer> projectionsDone;
	private boolean largeVolumeMode = false;
	private int nSteps = 1;
	private int subVolumeZ = 0;

	private boolean initialized = false;

	public OpenCLBackProjectorStreakReduction () {
		super();
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
		forceSmallVolume = UserUtil.queryBoolean("Force Small Volume Mode?");
		configured = true;
	}
	
	public void configure(boolean forceSmallVolumeMode) throws Exception{
		this.forceSmallVolume = forceSmallVolumeMode;
		configured = true;
	}

	protected void init(){
		if (!initialized) {
			largeVolumeMode = false;

			int reconDimensionX = getGeometry().getReconDimensionX();
			int reconDimensionY = getGeometry().getReconDimensionY();
			int reconDimensionZ = getGeometry().getReconDimensionZ();
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

				// initialize the program
				if (program==null || !program.getContext().equals(this.context)){
					program = context.createProgram(this.getClass().getResourceAsStream("backprojectCLStreakReduction.cl")).build();
					
					//program = context.createProgram(TestOpenCL.class.getResourceAsStream("C:\\LME\\Desktop\\Bachelorarbeit Katrin Mentl\\KONRAD\\CONRAD\\src\\edu\\stanford\\rsl\\science\\mentl\\cardiacVasculatureRecon\\reconWithScatterCorrection\\backprojectCLScatterCorr.cl")).build();
				}
				
				


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

			// check space on device:
			long memory = device.getMaxMemAllocSize();
			long availableMemory = (memory);
			long requiredMemory = (long)(((
					((double) reconDimensionX) * reconDimensionY * ((double) reconDimensionZ) * 4) 
					+ (((double)Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()) * Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth() * 4)));
			if (debug) {
				CONRAD.log("Total available Memory on OpenCL card:" + availableMemory);
				CONRAD.log("Required Memory on OpenCL card:" + requiredMemory);
			}
			if (!forceSmallVolume) {
				if (requiredMemory > availableMemory){
					nSteps = (int)OpenCLUtil.iDivUp (requiredMemory, availableMemory);
					if (debug) CONRAD.log("Switching to large volume mode with nSteps = " + nSteps);
					largeVolumeMode = true;
				}
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
			kernelFunction = program.createCLKernel("backprojectKernel");

			// create the reconstruction volume;
			int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * 4;
			
			//h_volume = new float[reconDimensionX * reconDimensionY * reconDimensionZ];	
	
			//int gat_ign = 4;
			h_volume = new float[(2*gat_ign +1) *reconDimensionX*reconDimensionY*reconDimensionZ];
			for(int i = 0; i < h_volume.length; i = i + 2*gat_ign+1){
				h_volume[i] = 0.0f;
				for(int j = 0; j < gat_ign; j++){
					h_volume[i + j] = -100001.0f;
				}
				for(int j = gat_ign; j < 2*gat_ign; j++){
					h_volume[i + j] = 100001.0f;
				}
			}
			
			destBuffer = context.createFloatBuffer(reconDimensionX*reconDimensionY*reconDimensionZ, Mem.WRITE_ONLY);

			// compute adapted volume size 
			//    volume size in x = multiple of bpBlockSize[0]
			//    volume size in y = multiple of bpBlockSize[1]

			System.out.println("RecondimX: " + reconDimensionX);
			int adaptedVolSize[] = new int[3];
			if ((reconDimensionX % bpBlockSize[0] ) == 0){
				adaptedVolSize[0] = reconDimensionX;
			} else {
				adaptedVolSize[0] = ((reconDimensionX / bpBlockSize[0]) + 1) * bpBlockSize[0];
			}
			System.out.println("AdaptedVolSize0: " + adaptedVolSize[0]);
			
			System.out.println("RecondimX: " + reconDimensionX);
			if ((reconDimensionY % bpBlockSize[1] ) == 0){
				adaptedVolSize[1] = reconDimensionY;
			} else {
				adaptedVolSize[1] = ((reconDimensionY / bpBlockSize[1]) + 1) * bpBlockSize[1];
			}
			System.out.println("AdaptedVolSize1: " + adaptedVolSize[0]);
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
				
				int reconDimensionX = getGeometry().getReconDimensionX();
				int reconDimensionY = getGeometry().getReconDimensionY();
				int reconDimensionZ = getGeometry().getReconDimensionZ();

				
				commandQueue.putReadBuffer(volumePointer, true).finish();
				volumePointer.getBuffer().rewind();
				volumePointer.getBuffer().get(h_volume);
				volumePointer.getBuffer().rewind();
				destBuffer.getBuffer().rewind();
				
				int length = h_volume.length;
				//int gat_ign = 4;
				float[] values = new float[reconDimensionX*reconDimensionY*reconDimensionZ];
				//h_volume = new float[length/7];
				h_volume = new float[reconDimensionX*reconDimensionY*reconDimensionZ];
				
				System.out.println("Length/(2*gat_ign+1): " + length/(2*gat_ign+1));
				System.out.println("recDimensions: " + reconDimensionX*reconDimensionY*reconDimensionZ);
				
				
				//subtract the volumina of the 6 ignore volumina
				//create the correct grid afterwards
				/*int v = 0;
				for(int i = 0; i < length; i = i + 2*gat_ign+1){
					float reconValue = volumePointer.getBuffer().get();
					for(int ign = 0; ign < 2*gat_ign; ign++){
						float val = volumePointer.getBuffer().get();
						if(val < 100000.0f && val > -100000.0f){
							//subtract the values that shall be ignored
							reconValue = reconValue - val;
						}
					}
					if(v < values.length){
						h_volume[v] = reconValue;
						v++;
					}
				}*/
				
				/*v = 0;
				for (int x=0;x < projectionVolume.getSize()[0];++x) {	
					for (int y=0;y < projectionVolume.getSize()[1];++y) {			//TODO MOEGLICHE FEHLERQUELLE
						for(int z = 0; z< projectionVolume.getSize()[2]; z++){
							projectionVolume.setAtIndex(x,y,z,values[v]);
							v ++;
						}
					}
				}*/



				/*int width = projectionVolume.getSize()[0];
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
				}*/
				

				destBuffer.getBuffer().rewind();
				for (int x=0; x < reconDimensionX ;++x) {	
					for (int y=0; y < reconDimensionY;++y) {
						for(int z = 0; z< reconDimensionZ; z++){
							//grid.setAtIndex(x, y, z, imgBuffer.getBuffer().get());
							projectionVolume.setAtIndex(x, y, z, destBuffer.getBuffer().get());
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
			if (volStride != null)
				volStride.release();
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
			volStride = null;
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
		commandQueue.putWriteBuffer(projectionMatrix, true).finish();
		//System.out.println("Uploading matrix " + projectionNumber);
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

	private synchronized void projectSingleProjection(int projectionNumber, int dimz){
		// load projection matrix
		initProjectionMatrix(projectionNumber);
		// load projection
		Grid2D projection = (Grid2D)projections.get(projectionNumber).clone(); 

		// Correct for constant part of distance weighting + For angular sampling
		double D =  getGeometry().getSourceToDetectorDistance();
		NumericPointwiseOperators.multiplyBy(projection, (float)(10 * D*D * 2* Math.PI * getGeometry().getPixelDimensionX() / getGeometry().getNumProjectionMatrices()));

		initProjectionData(projection);
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
		.putArg(destBuffer)
		.putArg((int) lineOffset)
		.putArg(reconDimensionZ)
		.putArg((float) voxelSpacingX)
		.putArg((float) voxelSpacingY)
		.putArg((float) voxelSpacingZ)
		.putArg((float) offsetX)
		.putArg((float) offsetY)
		.putArg((float) offsetZ)
		.putArg(projectionTex)
		.putArg(volStride)
		.putArg(projectionMatrix);

		int[] realLocalSize = {Math.min(device.getMaxWorkGroupSize(),bpBlockSize[0]), Math.min(device.getMaxWorkGroupSize(),bpBlockSize[1])};
		// rounded up to the nearest multiple of localWorkSize
		int[] globalWorkSize = {getGeometry().getReconDimensionX(), getGeometry().getReconDimensionY()}; 
		if ((globalWorkSize[0] % bpBlockSize[0] ) != 0){
			globalWorkSize[0] = ((globalWorkSize[0] / bpBlockSize[0]) + 1) * bpBlockSize[0];
		}
		if ((globalWorkSize[1] % bpBlockSize[1] ) != 0){
			globalWorkSize[1] = ((globalWorkSize[1] / bpBlockSize[1]) + 1) * bpBlockSize[1];
		}

		// Call the OpenCL kernel, writing the results into the volume which is pointed at
		commandQueue
		.putWriteImage(projectionTex, true)
		.finish()
		.put2DRangeKernel(kernelFunction, 0, 0, globalWorkSize[0], globalWorkSize[1], realLocalSize[0], realLocalSize[1])
		//.finish()
		//.putReadBuffer(dOut, true)
		.finish();
		projectionTex.release();
		projectionTex = null;
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
					//process all projections
					workOnProjectionData();
				} else {
					checkProjectionData();
				}
			}
			CONRAD.log("large Volume " + largeVolumeMode);
			//if (largeVolumeMode){

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

	public void reconstructOffline(ImagePlus imagePlus) throws Exception {
		ImagePlusDataSink sink = new ImagePlusDataSink();
		configure();
		init();
		for (int i = 0; i < imagePlus.getStackSize(); i++){
			backproject(ImageUtil.wrapImageProcessor(imagePlus.getStack().getProcessor(i+1)), i);
		}
		waitForResult();
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();
		int [] size = projectionVolume.getSize();
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

	/**
	 * @return the forceSmallVolume
	 */
	public boolean isForceSmallVolume() {
		return forceSmallVolume;
	}

	/**
	 * @param forceSmallVolume the forceSmallVolume to set
	 */
	public void setForceSmallVolume(boolean forceSmallVolume) {
		this.forceSmallVolume = forceSmallVolume;
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
