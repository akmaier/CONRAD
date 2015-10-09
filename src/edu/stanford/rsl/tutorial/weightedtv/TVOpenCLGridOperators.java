/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.weightedtv;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.SequenceInputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridOperators;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridInterface;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * This TVOpenCLGridOperators class is derived from OpenCLGridOperators. Here all the computations use 2D range kernel
 * instead of 1D range kernel.
 * It contains the TV operators for 2D and 3D cases.
 * @author Yixing Huang
 * *****************************************************************************************
 * OpenCLGridOperators encapsulates all grid operators which are implemented in OpenCL. 
 * It is implemented as singleton, because all grids share the same operations. 
 * All non-void kernels have to be stored in the property nonVoidKernels, to make sure that memory is allocated on the device.
 */
public class TVOpenCLGridOperators extends OpenCLGridOperators{

	private String kernelFile = "TVPointwiseOperators.cl";
	protected String extendedKernelFile = null;

	private static HashMap<CLDevice,CLProgram> deviceProgramMap;
	private static HashMap<String, HashMap<CLProgram, CLKernel>> programKernelMap;
	private static HashMap<CLDevice, CLCommandQueue> deviceCommandQueueMap;
	private static HashMap<String, Integer> kernelNameLocalSizeMap;
	private static Set<String> nonVoidKernels = new HashSet<String>(Arrays.asList(new String[] {"maximum", "minimum", "sum", "stddev", "dotProduct", "normL1", "countNegativeElements"}) ); 
	
	
	// singleton implementation
	protected TVOpenCLGridOperators() { 
		programKernelMap = new HashMap<String, HashMap<CLProgram,CLKernel>>();
		deviceProgramMap = new HashMap<CLDevice,CLProgram>();
		deviceCommandQueueMap = new HashMap<CLDevice,CLCommandQueue>();
		kernelNameLocalSizeMap = new HashMap<String, Integer>();
	}
	
	static TVOpenCLGridOperators op = new TVOpenCLGridOperators();
	
	public static TVOpenCLGridOperators getInstance() {
		return op;
	}
	
	
	/**
	 * This class encapsulate the complete OpenCLSetup which contains all OpenCL properties belonging to and influencing each other. It is implemented as singleton.
	 */
	protected class OpenCLSetup {
		private CLDevice device;
		private CLContext context;
		private CLProgram program;
		private CLCommandQueue commandQueue;
		private int localSize;
		private CLKernel kernel;	

		/**
		 * An OpenCL setup depends on the kernel name (the operation) and the device where the OpenCLGrid has stored its buffer. Although it is not necessary to store the device, we do so, because the complete OpenCL setup is stored in one class.
		 * @param kernelName OpenCL kernel name of the operation
		 * @param device OpenCL device where the OpenCLGrid has stored its buffer
		 */
		public OpenCLSetup(String kernelName, CLDevice device) { 
		
			
			// device
			this.device = device;
			
			// Program
			CLProgram program = deviceProgramMap.get(device);
			if(program == null)
			{
				InputStream programFile;
				if (extendedKernelFile == null) {
					programFile = TVOpenCLGridOperators.class.getResourceAsStream(kernelFile);
				}
				else {
					 programFile = new SequenceInputStream(TVOpenCLGridOperators.class.getResourceAsStream(kernelFile), TVOpenCLGridOperators.class.getResourceAsStream(extendedKernelFile));
				}
								
				try {
					program = device.getContext().createProgram(programFile).build();
				} catch (IOException e) {
					e.printStackTrace();
					program = null;
				}
				deviceProgramMap.put(device, program);
			}
			this.program = program;
			
			
			// Kernel
			HashMap<CLProgram, CLKernel> programMap = programKernelMap.get(kernelName);
			if (programMap == null){
				programMap = new HashMap<CLProgram, CLKernel>();
				programKernelMap.put(kernelName, programMap);
			}
			CLKernel kernel = programMap.get(program);
			if(kernel == null){
				kernel = program.createCLKernel(kernelName);
				programMap.put(program, kernel);
			}
			this.kernel = kernel;
			
			
			// queue
			CLCommandQueue commandQueue = deviceCommandQueueMap.get(device);
			if (commandQueue == null) {
				commandQueue = device.createCommandQueue();
				deviceCommandQueueMap.put(device, commandQueue);
			}
			this.commandQueue = commandQueue;
			
			
			// workgroup (local) size
			Integer workgroupSize = kernelNameLocalSizeMap.get(kernelName);
			if (workgroupSize == null) {
				workgroupSize = (int)kernel.getWorkGroupSize(device);
				kernelNameLocalSizeMap.put(kernelName, workgroupSize);
			}
			this.localSize = workgroupSize; 
			
			
			// context
			this.context = device.getContext();
			
		}
		
		
		public CLDevice getDevice() {
			return device;
		}
		

		public CLContext getContext() {
			return context;
		}
		

		public CLProgram getProgram() {
			return program;
		}
		

		public CLCommandQueue getCommandQueue() {
			return commandQueue;
		}
		

		public int getLocalSize() {
			return localSize;
		}
		

		public CLKernel getKernel() {
			return kernel;
		}
		
		
		public int getGlobalSize(int elementCount) {
			return OpenCLUtil.roundUp(localSize, elementCount);
		}
	}

	
	/**
	 * Run a 2D range kernel with the format 'gridA = gridA operation gridB' 
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,int[]gridSize) { 
	
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		//int localSize = openCLSetup.getLocalSize();	
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);	
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(gridSize[0]).putArg(gridSize[1]).putArg(gridSize[2]);
			queue.put2DRangeKernel(kernel, 0,0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
	
		kernel.rewind();
		return null;
	}
	
	/**
	 *  Run a 2D range kernel with the format 'gridA = gridA operation gridB' for 2D images
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel2D(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,int[]gridSize) { 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		//int localSize = openCLSetup.getLocalSize();	
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);	
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(gridSize[0]).putArg(gridSize[1]);
			queue.put2DRangeKernel(kernel, 0,0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
	
		kernel.rewind();
		return null;
	}
	
	/**
	 * run 2D range kernel with the format 'gridA = gridA operation floata' 
	 * @param kernelName
	 * @param device
	 * @param gridBuffer
	 * @param value
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridBuffer, float value,int[]gridSize){ 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		//int localSize = openCLSetup.getLocalSize();	
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);	
		
			kernel.putArg(gridBuffer).putArg(value).putArg(gridSize[0]).putArg(gridSize[1]).putArg(gridSize[2]);
			queue.put2DRangeKernel(kernel, 0,0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
	
		kernel.rewind();
		return null;
	}
	
	
	/**
	 * run 2D range kernel with the format 'gridC = gridA operation gridB'
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param gridCBuffer
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,CLBuffer<FloatBuffer> gridCBuffer,int[]gridSize) { 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);	
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(gridCBuffer).putArg(gridSize[0]).putArg(gridSize[1]).putArg(gridSize[2]);
			queue.put2DRangeKernel(kernel, 0,0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
		kernel.rewind();
		return null;
	}
	
	/**
	 * run 2D range kernel with the format 'gridC = gridA operation gridB' for 2D images
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param gridCBuffer
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel2D(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,CLBuffer<FloatBuffer> gridCBuffer,int[]gridSize) { 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);	
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(gridCBuffer).putArg(gridSize[0]).putArg(gridSize[1]);
			queue.put2DRangeKernel(kernel, 0,0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
		kernel.rewind();
		return null;
	}
	/**
	 * run 2D range kernel with the format 'gridA = gridB operation floata'
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param eps
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,float eps,int[]gridSize) { 
		int elementCount = gridABuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);
				
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(eps).putArg(gridSize[0]).putArg(gridSize[1]).putArg(gridSize[2]);
			queue.put2DRangeKernel(kernel,0, 0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
		kernel.rewind();
		return null;
	}
	
	/**
	 * run 2D range kernel with the format 'gridA = gridB operation floata' for 2D images
	 * @param kernelName
	 * @param device
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @param eps
	 * @param gridSize
	 * @return
	 */
	private CLBuffer<FloatBuffer> runKernel2D(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,float eps,int[]gridSize) { 
		int elementCount = gridABuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = Math.min(device.getMaxWorkGroupSize(), 16);
		//int globalSize = openCLSetup.getGlobalSize(elementCount);
		int globalWorkSizeX = OpenCLUtil.roundUp(localSize, gridSize[0]); // rounded up to the nearest multiple of localWorkSize
		int globalWorkSizeY = OpenCLUtil.roundUp(localSize, gridSize[1]);
				
		
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(eps).putArg(gridSize[0]).putArg(gridSize[1]);
			queue.put2DRangeKernel(kernel,0, 0, globalWorkSizeX, globalWorkSizeY,localSize,localSize);
		
		
		queue.finish();
		kernel.rewind();
		return null;
	}

	
	/**
	 * compute Image Gradient
	 * @param imgGrid
	 * @param imgGradient
	 */
	public void computeImageGradient( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel("computeImageGradient",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * compute Image gradient for 2D images/case
	 * @param imgGrid
	 * @param imgGradient
	 */
	public void computeImageGradient2D( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel2D("computeImageGradient2D",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();

	}
	
	/**
	 * compute image gradient for 2D anisotropic weighted TV (AwTV)
	 * @param imgGrid
	 * @param imgGradient
	 */
	public void computeImageGradient2DX( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel2D("computeImageGradient2DX",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * update weight matrix 
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeWeightMatrixUpdate(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeWeightMatrixUpdate",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	
	}
	
	/**
	 * update weight matrix for AwTV along Y direction
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeWeightMatrixUpdateY(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeWeightMatrixUpdateY",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	}
	
	/**
	 * Update weight matrix for AwTV along X direction
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeWeightMatrixUpdateX(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeWeightMatrixUpdateX",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	}
	
	/**
	 * 
	 * @param imgGradient
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeAdaptiveWeightMatrixUpdate(NumericGrid imgGradient,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeAdaptiveWeightMatrixUpdate",device,clmemGradient,clmemWmatrix,eps,imgGradient.getSize());
	}
	
	/**
	 * B=100 for instance in Y direction
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeDirectionalWeightedTVWeightMatrixUpdate(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeDirectionalWeightedTVWeightMatrixUpdate",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	}
	
	/**
	 * update weight matrix for 2D images
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeWeightMatrixUpdate2D(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel2D("computeWeightMatrixUpdate2D",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	}
	
	/**
	 * update weight matrix for 2D AwTV along X direction
	 * @param imgGrid
	 * @param Wmatrix
	 * @param eps
	 */
	public void computeWeightMatrixUpdate2DX(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel2D("computeWeightMatrixUpdate2DX",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
	}
	
	/**
	 * update weight matrix, compute the image gradient in XY plane, Z direction is not included
	 * @param imgGrid
	 * @param Wmatrix
	 * @param weps
	 */
	public void computeWeightMatrixUpdate2(NumericGrid imgGrid,NumericGrid Wmatrix,float weps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("computeWeightMatrixUpdate2",device,clmemGradient,clmemWmatrix,weps,imgGrid.getSize());
	}
	
/**
 * compute weighted TV gradient
 * @param imgGrid
 * @param Wmatrix
 * @param wTVgradient
 */
	public void computeWeightedTVGradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel("computeWeightedTVGradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();		
	}
	
	/**
	 * 
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
	public void computeWeightedTVGradientY(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel("computeWeightedTVGradientY",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();		
	}
	
	/**
	 * compute weight TV gradient for AwTV along X
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
	public void computeWeightedTVGradientX(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel("computeWeightedTVGradientX",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	/**
	 * 
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
	public void computeAdaptiveWeightedTVGradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel("computeAdaptiveWeightedTVGradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();	
	}
	
	
	/** 
	 * compute weighted directional TV gradient, here in Y direction the gradient has a large weight B=100 for instance
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
		public void computeDirectionalWeightedTVGradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
			OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
			OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
			OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
			clImgGrid.getDelegate().prepareForDeviceOperation();
			clWmatrix.getDelegate().prepareForDeviceOperation();
			clwTVgradient.getDelegate().prepareForDeviceOperation();
			CLDevice device=clImgGrid.getDelegate().getCLDevice();
			
			CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
			CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
			CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
			runKernel("computeDirectionalWeightedTVGradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
			clwTVgradient.getDelegate().notifyDeviceChange();	
		}
		
		/**
		 * compute weighted TV gradient for 2D images
		 * @param imgGrid
		 * @param Wmatrix
		 * @param wTVgradient
		 */
	public void computeWeightedTVGradient2D(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel2D("computeWeightedTVGradient2D",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
			
	}
	
	/**
	 * compute the AwTV gradient for 2D images along X
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
	public void computeWeightedTVGradient2DX(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel2D("computeWeightedTVGradient2DX",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();		
	}
	
	/**
	 * only in XY plane, Z direction is not included
	 * @param imgGrid
	 * @param Wmatrix
	 * @param wTVgradient
	 */
	public void computeWeightedTVGradient2(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clwTVgradient = (OpenCLGridInterface)wTVgradient;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clwTVgradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemwTVgradient=clwTVgradient.getDelegate().getCLBuffer();
		runKernel("computeWeightedTVGradient2",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();	
	}
	
	/**
	 * get weighted TV value
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getWeightedTV(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getWeightedTV",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * get anisotropic weighted TV along Y
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getWeightedTVY(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getWeightedTVY",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * get TV value for AwTV along X
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getWeightedTVX(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getWeightedTVX",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * 
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getAdaptiveWeightedTV(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getAdaptiveWeightedTV",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * get directional weighted TV along Y direction, the gradient at Y direction has a larger weight B=100 for instance
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getDirectionalWeightedTV(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getDirectionalWeightedTV",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * only in XY plane, Z direction is not included
	 * @param imgGrid
	 * @param Wmatrix
	 * @param tempZSum
	 */
	public void getWeightedTV2(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface cltempZSum = (OpenCLGridInterface)tempZSum;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clWmatrix.getDelegate().prepareForDeviceOperation();
		cltempZSum.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemtempZSum=cltempZSum.getDelegate().getCLBuffer();
		runKernel("getWeightedTV2",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * add the FOV mask
	 * @param imgGrid
	 * @param radius
	 */
	public void maskFOV(NumericGrid imgGrid, float radius){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		runKernel("FOVmask",device,clmemImg,radius,imgGrid.getSize());
	}
	


	
	

	
	//TODO: Check the methods getAllInstances, getAllOpenCLGridOperatorProgramsAsString, and getCompleteRessourceAsString why they are necessary. 
	
	/**
	 * Auxiliary method that lists all instances of GridOperators
	 * Users can derive from OpenCLGridOperators and define their cl-file path
	 * in the field "programFile"
	 * 
	 * Make sure that you add an instance of your personal OpenCLGridOperators in this method
	 * @return All instances of existing OpenCLGridOperator classes
	 */
	public static TVOpenCLGridOperators[] getAllInstances(){
		// TODO: replace with automatic search on java class path
		// Problem is that this might be really slow. // Comment by Michael Dorner: It IS very slow.
		return new TVOpenCLGridOperators[]{
				new TVOpenCLGridOperators() 	// Comment by Michael Dorner: GridOperators are singletons. Therefore we should see any constructor in there. Additionally, why creating an array with only one entry?

		};
	}

	
	/**
	 * Obtains all OpenCLGridOperators instances and concatenates all related 
	 * cl-source files to one long string
	 * @return Concatenated cl-source code
	 */
	public String getAllOpenCLGridOperatorProgramsAsString(){
		String out = "";
		TVOpenCLGridOperators[] instances = getAllInstances();
		for (int i = 0; i < instances.length; i++) {
			try {
				out += instances[i].getCompleteRessourceAsString();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return out;
	}

	/**
	 * Reads a cl-program file and returns it as String
	 * @return A cl-program file as String
	 * @throws IOException
	 */
	protected String getCompleteRessourceAsString() throws IOException{
		InputStream inStream = this.getClass().getResourceAsStream(kernelFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
		String content = "";
		String line = br.readLine();
		while (line != null){
			content += line + "\n";
			line = br.readLine();
		};
		return content;
	}
}

