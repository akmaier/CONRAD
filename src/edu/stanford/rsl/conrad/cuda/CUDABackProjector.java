package edu.stanford.rsl.conrad.cuda;

import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.util.ArrayList;
import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
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
import jcuda.runtime.JCuda;
import jcuda.runtime.dim3;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.reconstruction.VOIBasedReconstructionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class CUDABackProjector extends VOIBasedReconstructionFilter implements Runnable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8615490043940236889L;
	/**
	 * The CUDA module containing the kernel
	 */
	private CUmodule module = null;
	//private static Object lock = new Object();
	private static boolean debug = true;

	// Pre-determined kernel block size
	static int bpBlockSize[] = {32, 16};

	/**
	 * The handle for the CUDA function of the kernel that is to be called
	 */
	private CUfunction function = null;

	/**
	 * The volume data that is to be reconstructed
	 */
	protected float h_volume[];

	/**
	 * The 2D projection texture reference
	 */
	private CUtexref projectionTex = null;

	/**
	 * The grid size of the kernel execution
	 */
	private dim3 gridSize = null;

	/**
	 * the context
	 */
	private CUcontext cuCtx = null;


	/**
	 * The global variable of the module which stores the
	 * view matrix.
	 */
	private CUdeviceptr projectionMatrix = null;
	private CUdeviceptr volStride = null;
	private CUdeviceptr volumePointer = null;
	private CUarray projectionArray = null;


	protected ImageGridBuffer projections;
	protected ArrayList<Integer> projectionsAvailable;
	protected ArrayList<Integer> projectionsDone;
	private boolean largeVolumeMode = false;
	private int nSteps = 1;
	private int subVolumeZ = 0;

	private boolean initialized = false;

	public CUDABackProjector () {
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
		cuCtx = null;
		gridSize = null;
		projectionTex = null;
		h_volume = null;
		initialized = false;
		function = null;
		module = null;
	}

	@Override
	public void configure() throws Exception{
		boolean success = true;
		configured = success;
	}

	public void reset(){
		projectionArray = null;
		JCuda.cudaThreadExit();
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
			// Initialize the JCudaDriver. Note that this has to be done from 
			// the same thread that will later use the JCudaDriver API.
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			CUdevice dev = CUDAUtil.getBestDevice();
			cuCtx = new CUcontext();
			JCudaDriver.cuCtxCreate(cuCtx, 0, dev);
			// check space on device:
			int [] memory = new int [1];
			int [] total = new int [1]; 
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			JCudaDriver.cuMemGetInfo(memory, total);
			int availableMemory = (int) (CUDAUtil.correctMemoryValue(memory[0]) / ((long)1024 * 1024));
			int requiredMemory = (int)(((
					((double) reconDimensionX) * reconDimensionY * ((double) reconDimensionZ) * Sizeof.FLOAT) 
					+ (((double)Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight()) * Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth() * Sizeof.FLOAT)) 
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
			JCudaDriver.cuModuleLoad(module, "backprojectWithCuda.ptx");

			// Obtain a function pointer to the kernel function. This function
			// will later be called. 
			// 
			function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module,
			"_Z17backprojectKernelPfiiffffff");
			// create the reconstruction volume;
			int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * Sizeof.FLOAT;
			if (largeVolumeMode){
				subVolumeZ = CUDAUtil.iDivUp(reconDimensionZ, nSteps);
				if(debug) System.out.println("SubVolumeZ: " + subVolumeZ);
				h_volume = new float[reconDimensionX * reconDimensionY * subVolumeZ];
				memorysize = reconDimensionX * reconDimensionY * subVolumeZ * Sizeof.FLOAT;
				if(debug)System.out.println("Memory: " + memorysize);
			} else {
				h_volume = new float[reconDimensionX * reconDimensionY * reconDimensionZ];	
			}
			// copy volume to device
			volumePointer = new CUdeviceptr();
			JCudaDriver.cuMemAlloc(volumePointer, memorysize);
			JCudaDriver.cuMemcpyHtoD(volumePointer, Pointer.to(h_volume), memorysize);

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


			volStride = new CUdeviceptr();
			JCudaDriver.cuModuleGetGlobal(volStride, new int[1], module, "gVolStride");
			JCudaDriver.cuMemcpyHtoD(volStride, Pointer.to(volStrideHost), Sizeof.INT * 2);

			// Calculate new grid size
			gridSize = new dim3(
					CUDAUtil.iDivUp(adaptedVolSize[0], bpBlockSize[0]), 
					CUDAUtil.iDivUp(adaptedVolSize[1], bpBlockSize[1]), 
					adaptedVolSize[2]);


			// Obtain the global pointer to the view matrix from
			// the module
			projectionMatrix = new CUdeviceptr();
			JCudaDriver.cuModuleGetGlobal(projectionMatrix, new int[1], module, "gProjMatrix");

			initialized = true;
		}

	}

	private synchronized void unload(){
		if (initialized) {

			if (projectionArray != null) {
				JCudaDriver.cuArrayDestroy(projectionArray);
			}

			int reconDimensionX = getGeometry().getReconDimensionX();
			int reconDimensionY = getGeometry().getReconDimensionY();
			int reconDimensionZ = getGeometry().getReconDimensionZ();


			if ((projectionVolume != null) && (!largeVolumeMode)) {
				// fetch data
				int memorysize = reconDimensionX * reconDimensionY * reconDimensionZ * 4;
				JCudaDriver.cuMemcpyDtoH(Pointer.to(h_volume), volumePointer, memorysize);
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

			JCudaDriver.cuMemFree(volumePointer);
			// destory context
			JCudaDriver.cuCtxDestroy(cuCtx);

			reset();

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
		JCudaDriver.cuMemcpyHtoD(projectionMatrix, Pointer.to(pMatFloat), Sizeof.FLOAT * pMatFloat.length);
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
				// Create the 2D array that will contain the
				// projection data. 
				projectionArray = new CUarray();
				CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
				ad.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
				ad.Width = projection.getWidth();
				ad.Height = projection.getHeight();
				ad.NumChannels = 1;//projection.getNChannels();
				JCudaDriver.cuArrayCreate(projectionArray, ad);
			}

			// Copy the projection data to the array  
			CUDA_MEMCPY2D copy2 = new CUDA_MEMCPY2D();
			copy2.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
			copy2.srcHost = Pointer.to(proj);
			copy2.srcPitch = projection.getWidth() * Sizeof.FLOAT;
			copy2.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
			copy2.dstArray = projectionArray;
			copy2.WidthInBytes = projection.getWidth() * Sizeof.FLOAT;
			copy2.Height = projection.getHeight();
			JCudaDriver.cuMemcpy2D(copy2);

			// Obtain the texture reference from the module, 
			// set its parameters and assign the projection  
			// array as its reference.
			projectionTex = new CUtexref();
			JCudaDriver.cuModuleGetTexRef(projectionTex, module, "gTex2D");
			JCudaDriver.cuTexRefSetFilterMode(projectionTex,
					CUfilter_mode.CU_TR_FILTER_MODE_LINEAR);
			JCudaDriver.cuTexRefSetAddressMode(projectionTex, 0,
					CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP);
			JCudaDriver.cuTexRefSetFlags(projectionTex,
					JCudaDriver.CU_TRSF_READ_AS_INTEGER);
			JCudaDriver.cuTexRefSetFormat(projectionTex,
					CUarray_format.CU_AD_FORMAT_FLOAT, 4);
			JCudaDriver.cuTexRefSetArray(projectionTex, projectionArray,
					JCudaDriver.CU_TRSA_OVERRIDE_FORMAT);

			// Set the texture references as parameters for the function call
			JCudaDriver.cuParamSetTexRef(function, JCudaDriver.CU_PARAM_TR_DEFAULT,
					projectionTex);
		} else {
			System.out.println("Projection was null!!");
		}
	}

	@Override
	public String getName() {
		return "CUDA Backprojector";
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
		cudaRun();
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
		NumericPointwiseOperators.multiplyBy(projection, (float)(D*D * 2* Math.PI / getGeometry().getNumProjectionMatrices()));		
		
		initProjectionData(projection);
		if (!largeVolumeMode) {
			projections.remove(projectionNumber);
		}
		// backproject for each slice
		// CUDA Grids are only two dimensional!
		int [] zed = new int[1];
		int reconDimensionZ = dimz;
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();

		zed[0] = reconDimensionZ;
		Pointer dOut = Pointer.to(volumePointer);
		Pointer pWidth = Pointer.to(new int[]{(int) lineOffset});
		Pointer pZOffset = Pointer.to(zed);
		float [] vsx = new float[]{(float) voxelSpacingX};
		Pointer pvsx = Pointer.to(vsx);
		Pointer pvsy = Pointer.to(new float[]{(float) voxelSpacingY});
		Pointer pvsz = Pointer.to(new float[]{(float) voxelSpacingZ});
		Pointer pox = Pointer.to(new float[]{(float) offsetX});
		Pointer poy = Pointer.to(new float[]{(float) offsetY});
		Pointer poz = Pointer.to(new float[]{(float) offsetZ});

		int offset = 0;
		//System.out.println(dimz + " " + zed[0] + " " + offsetZ + " " + voxelSpacingZ);
		offset = CUDAUtil.align(offset, Sizeof.POINTER);
		JCudaDriver.cuParamSetv(function, offset, dOut, Sizeof.POINTER);
		offset += Sizeof.POINTER;

		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, pWidth, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.INT);
		JCudaDriver.cuParamSetv(function, offset, pZOffset, Sizeof.INT);
		offset += Sizeof.INT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsx, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsy, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pvsz, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;


		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, pox, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, poy, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		offset = CUDAUtil.align(offset, Sizeof.FLOAT);
		JCudaDriver.cuParamSetv(function, offset, poz, Sizeof.FLOAT);
		offset += Sizeof.FLOAT;

		JCudaDriver.cuParamSetSize(function, offset);

		// Call the CUDA kernel, writing the results into the volume which is pointed at
		JCudaDriver.cuFuncSetBlockShape(function, bpBlockSize[0], bpBlockSize[1], 1);
		JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
		JCudaDriver.cuCtxSynchronize();

	}

	public void cudaRun() {
		try {
			while (projectionsAvailable.size() > 0) {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				if (showStatus) {
					float status = (float) (1.0 / projections.size());
					if (largeVolumeMode) {
						IJ.showStatus("Streaming Projections to CUDA Buffer");
					} else {
						IJ.showStatus("Backprojecting with CUDA");
					}
					IJ.showProgress(status);
				}
				if (!largeVolumeMode) {			
					workOnProjectionData();
				} else {
					checkProjectionData();
				}
			}
			System.out.println("large Volume " + largeVolumeMode);
			if (largeVolumeMode){
				// we have collected all projections.
				// now we can reconstruct subvolumes and stich them together.
				int reconDimensionX = getGeometry().getReconDimensionX();
				int reconDimensionY = getGeometry().getReconDimensionY();
				int reconDimensionZ = getGeometry().getReconDimensionZ();
				double voxelSpacingX = getGeometry().getVoxelSpacingX();
				double voxelSpacingY = getGeometry().getVoxelSpacingY();
				double voxelSpacingZ = getGeometry().getVoxelSpacingZ();
				useVOImap = false;
				initialize(projections.get(0));
				double originalOffsetZ = offsetZ;
				double originalReconDimZ = reconDimensionZ;
				reconDimensionZ = subVolumeZ;
				int memorysize = reconDimensionX * reconDimensionY * subVolumeZ * Sizeof.FLOAT;
				int maxProjectionNumber = projections.size();
				float all = nSteps * maxProjectionNumber*2;
								
				for (int n =0; n < nSteps; n++){ // For each subvolume
					// set all to 0;
					Arrays.fill(h_volume, 0);
					JCudaDriver.cuMemcpyHtoD(volumePointer, Pointer.to(h_volume), memorysize);
					offsetZ = originalOffsetZ - (reconDimensionZ*voxelSpacingZ*n);

					for (int p = 0; p < maxProjectionNumber; p ++){ // For all projections
						float currentStep = (n*maxProjectionNumber*2) + p;
						if (showStatus) {
							IJ.showStatus("Backprojecting with CUDA");
							IJ.showProgress(currentStep/all);
						}						
//						System.out.println("Current: " + p);
						try {
							projectSingleProjection(p, reconDimensionZ);
						} catch (Exception e){
							System.out.println("Backprojection of projection " + p + " was not successful.");
							e.printStackTrace();
						}
					}
					// Gather volume
					JCudaDriver.cuMemcpyDtoH(Pointer.to(h_volume), volumePointer, memorysize);

					// move data to ImagePlus;
					if (projectionVolume != null) {
						for (int k = 0; k < reconDimensionZ; k++){
							int index = (n*subVolumeZ) + k;
							if (showStatus) {
								float currentStep = (n*maxProjectionNumber*2) + maxProjectionNumber + k;
								IJ.showStatus("Fetching Volume from CUDA");
								IJ.showProgress(currentStep/all);
							}
							if (index < originalReconDimZ) {
								for (int j = 0; j < projectionVolume.getSize()[1]; j++){
									for (int i = 0; i < projectionVolume.getSize()[0]; i++){										
										double[][] voxel = new double [4][1];

										int idx = (((projectionVolume.getSize()[1] * k) + j) * projectionVolume.getSize()[0]) + i;
										float value = h_volume[idx];
										
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
		System.out.println(size [0] + " " + size [1] + " " + size[2]);
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
		reset();
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
		return "CUDA Backprojector";
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
