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

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.iterators.NumericPointwiseIteratorND;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGridInterface;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * OpenCLGridOperators encapsulates all grid operators which are implemented in OpenCL. 
 * It is implemented as singleton, because all grids share the same operations. 
 * All non-void kernels have to be stored in the property nonVoidKernels, to make sure that memory is allocated on the device.
 */
public class TVOpenCLGridOperators{

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
	 * Run a kernel with the format 'grid = operation(grid)' such as abs, min, or pow
	 * @param kernelName kernel name
	 * @param device CLDevice
	 * @param gridBuffer CLBuffer
	 * @return Null if it is a void kernel or a CLBuffer of size (localSize), if it is a non-void kernel
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridBuffer) { 
		int elementCount = gridBuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = openCLSetup.getLocalSize();		
		int globalSize = openCLSetup.getGlobalSize(elementCount);
		
		CLBuffer<FloatBuffer> resultBuffer = null;
				
		if (nonVoidKernels.contains(kernelName)) {
			resultBuffer = context.createFloatBuffer((globalSize/localSize), Mem.READ_WRITE);
			kernel.putArg(gridBuffer).putArg(resultBuffer).putArg(elementCount).putNullArg(localSize*4); // 4 bytes per float
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.putReadBuffer(resultBuffer, true);			
		}
		else {
			kernel.putArg(gridBuffer).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
		}
		
		queue.finish();
		kernel.rewind();
		return resultBuffer;
	}

	
	/**
	 * 
	 * @param Run a kernel with the format 'grid = grid operation value', such as stddev or addBy
	 * @param kernelName kernel name
	 * @param device CLDevice
	 * @param gridBuffer CLBuffer
	 * @return Null if it is a void kernel or a CLBuffer of size (localSize), if it is a non-void kernel
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridBuffer, float value) { 
		int elementCount = gridBuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = openCLSetup.getLocalSize();		
		int globalSize = openCLSetup.getGlobalSize(elementCount);
		
		CLBuffer<FloatBuffer> resultBuffer = null;
				
		if (nonVoidKernels.contains(kernelName)) {
			resultBuffer = context.createFloatBuffer(globalSize/localSize, Mem.READ_ONLY);
			kernel.putArg(gridBuffer).putArg(resultBuffer).putArg(value).putArg(elementCount).putNullArg(4*localSize);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.putReadBuffer(resultBuffer, true);
		}
		else {
			kernel.putArg(gridBuffer).putArg(value).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
		}
		
		queue.finish();
		kernel.rewind();
		return resultBuffer;
	}
	
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer,int[] offset, int[] volSize, boolean offsetleft) { 
		int elementCount = gridABuffer.getCLCapacity(); 
		int left = offsetleft ? 1 : 0;
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		
		//int localSize = (int)Math.min(Math.sqrt(openCLSetup.getLocalSize()),16);	
		int localSize = openCLSetup.getLocalSize();	
		int globalSize = openCLSetup.getGlobalSize(elementCount);

		kernel	.putArg(gridABuffer).putArg(gridBBuffer)
				.putArg(offset[0]).putArg(offset[1]).putArg(offset[2])
				.putArg(volSize[0]).putArg(volSize[1]).putArg(volSize[2])
				.putArg(left);
		
		queue.put1DRangeKernel(kernel,0,globalSize,localSize)
			 .finish();

		kernel.rewind();
		return null;
	}
	

	/**
	 * Run a kernel with the format 'gridA = gridA operation gridB', such as addBy or dotProduct
	 * @param kernelName kernel name
	 * @param device CLDevice
	 * @param gridABuffer
	 * @param gridBBuffer
	 * @return Null if it is a void kernel or a CLBuffer of size (localSize), if it is a non-void kernel
	 */
	private CLBuffer<FloatBuffer> runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> gridABuffer, CLBuffer<FloatBuffer> gridBBuffer) { 
		int elementCount = gridABuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = openCLSetup.getLocalSize();		
		int globalSize = openCLSetup.getGlobalSize(elementCount);
		CLBuffer<FloatBuffer> resultBuffer = null;
				
		if (nonVoidKernels.contains(kernelName)) {
			resultBuffer = context.createFloatBuffer((globalSize/localSize), Mem.READ_ONLY);
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(resultBuffer).putArg(elementCount).putNullArg(4*localSize);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.putReadBuffer(resultBuffer, true);
		}
		else {
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
		}
		
		queue.finish();
		kernel.rewind();
		return resultBuffer;
	}
	//Yixing Huang
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
	
	//Yixing Huang
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
	//Yixing Huang
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

	//Yixing Huang
	public void compute_img_gradient( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel("compute_img_gradient",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemGradient.release();	
	}
	
	public void compute_img_gradient2D( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel2D("compute_img_gradient2D",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemGradient.release();	
	}
	
	public void compute_img_gradient2D_X( NumericGrid imgGrid,  NumericGrid imgGradient){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
	
		clImgGrid.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGradient.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		runKernel2D("compute_img_gradient2D_X",device,clmemImg,clmemGradient,imgGrid.getSize());
		clImgGradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemGradient.release();	
	}
	
	//Yixing Huang
	public void compute_Wmatrix_Update(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_Wmatrix_Update",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
	public void compute_Wmatrix_Update_Y(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_Wmatrix_Update_Y",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
	public void compute_Wmatrix_Update_X(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_Wmatrix_Update_X",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
	public void compute_adaptive_Wmatrix_Update(NumericGrid imgGradient,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGradient;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_adaptive_Wmatrix_Update",device,clmemGradient,clmemWmatrix,eps,imgGradient.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	public void compute_AwTV_Wmatrix_Update(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_AwTV_Wmatrix_Update",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	public void compute_Wmatrix_Update2D(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel2D("compute_Wmatrix_Update2D",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
	public void compute_Wmatrix_Update2D_X(NumericGrid imgGrid,NumericGrid Wmatrix,float eps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel2D("compute_Wmatrix_Update2D_X",device,clmemGradient,clmemWmatrix,eps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
	public void compute_Wmatrix_Update2(NumericGrid imgGrid,NumericGrid Wmatrix,float weps){
		OpenCLGridInterface clWmatrix = (OpenCLGridInterface)Wmatrix;
		OpenCLGridInterface clImgGradient = (OpenCLGridInterface)imgGrid;
		
		clWmatrix.getDelegate().prepareForDeviceOperation();
		clImgGradient.getDelegate().prepareForDeviceOperation();
		CLDevice device=clWmatrix.getDelegate().getCLDevice();

		CLBuffer<FloatBuffer> clmemGradient=clImgGradient.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemWmatrix=clWmatrix.getDelegate().getCLBuffer();
		runKernel("compute_Wmatrix_Update2",device,clmemGradient,clmemWmatrix,weps,imgGrid.getSize());
		//clWmatrix.getDelegate().notifyDeviceChange();
		//clmemGradient.release();
		//clmemWmatrix.release();
	}
	
//Yixing Huang
	public void compute_wTV_Gradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel("compute_wTV_gradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	
	public void compute_wTV_Gradient_Y(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel("compute_wTV_gradient_Y",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	public void compute_wTV_Gradient_X(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel("compute_wTV_gradient_X",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	public void compute_wTV_adaptive_Gradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel("compute_wTV_adaptive_gradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	
	
	//Yixing Huang
		public void compute_AwTV_Gradient(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
			runKernel("compute_AwTV_gradient",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
			clwTVgradient.getDelegate().notifyDeviceChange();
			//clmemImg.release();
			//clmemWmatrix.release();
			//clmemwTVgradient.release();		
		}
	public void compute_wTV_Gradient2D(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel2D("compute_wTV_gradient2D",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	public void compute_wTV_Gradient2D_X(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel2D("compute_wTV_gradient2D_X",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	public void compute_wTV_Gradient2(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid wTVgradient){
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
		runKernel("compute_wTV_gradient2",device,clmemImg,clmemWmatrix,clmemwTVgradient,imgGrid.getSize());
		clwTVgradient.getDelegate().notifyDeviceChange();
		//clmemImg.release();
		//clmemWmatrix.release();
		//clmemwTVgradient.release();		
	}
	
	public void getwTV(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getwTV",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	public void getwTV_Y(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getwTV_Y",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	public void getwTV_X(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getwTV_X",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	public void getwTV_adaptive(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getwTV_adaptive",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	public void getAwTV(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getAwTV",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	
	public void getwTV2(NumericGrid imgGrid,NumericGrid Wmatrix,NumericGrid tempZSum){
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
		runKernel("getwTV2",device,clmemImg,clmemWmatrix,clmemtempZSum,imgGrid.getSize());
		cltempZSum.getDelegate().notifyDeviceChange();
	}
	//Yixing Huang
	public void maskFOV(NumericGrid imgGrid, float radius){
		OpenCLGridInterface clImgGrid = (OpenCLGridInterface)imgGrid;
		clImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clImgGrid.getDelegate().getCLDevice();
		CLBuffer<FloatBuffer> clmemImg=clImgGrid.getDelegate().getCLBuffer();
		runKernel("FOVmask",device,clmemImg,radius,imgGrid.getSize());
	}
	
	public void upSamling(NumericGrid DownImgGrid, NumericGrid UpImgGrid){
		OpenCLGridInterface clDownImgGrid = (OpenCLGridInterface)DownImgGrid;
		OpenCLGridInterface clUpImgGrid = (OpenCLGridInterface)UpImgGrid;
	
		clDownImgGrid.getDelegate().prepareForDeviceOperation();
		clUpImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clDownImgGrid.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemDownImg=clDownImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemUpImg=clUpImgGrid.getDelegate().getCLBuffer();
		runKernel("UpSampling_Y",device,clmemDownImg,clmemUpImg,DownImgGrid.getSize());
		clUpImgGrid.getDelegate().notifyDeviceChange();
	}
	
	public void downSampling(NumericGrid DownImgGrid, NumericGrid UpImgGrid){
		OpenCLGridInterface clDownImgGrid = (OpenCLGridInterface)DownImgGrid;
		OpenCLGridInterface clUpImgGrid = (OpenCLGridInterface)UpImgGrid;
	
		clDownImgGrid.getDelegate().prepareForDeviceOperation();
		clUpImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clDownImgGrid.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemDownImg=clDownImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemUpImg=clUpImgGrid.getDelegate().getCLBuffer();
		runKernel("DownSampling_Y",device,clmemDownImg,clmemUpImg,DownImgGrid.getSize());
		clUpImgGrid.getDelegate().notifyDeviceChange();
	}
	
	public void upSamling_even(NumericGrid DownImgGrid, NumericGrid UpImgGrid){
		OpenCLGridInterface clDownImgGrid = (OpenCLGridInterface)DownImgGrid;
		OpenCLGridInterface clUpImgGrid = (OpenCLGridInterface)UpImgGrid;
	
		clDownImgGrid.getDelegate().prepareForDeviceOperation();
		clUpImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clDownImgGrid.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemDownImg=clDownImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemUpImg=clUpImgGrid.getDelegate().getCLBuffer();
		runKernel("UpSampling_Y_even",device,clmemDownImg,clmemUpImg,DownImgGrid.getSize());
		clUpImgGrid.getDelegate().notifyDeviceChange();
	}
	
	public void upSamling_odd(NumericGrid DownImgGrid, NumericGrid UpImgGrid){
		OpenCLGridInterface clDownImgGrid = (OpenCLGridInterface)DownImgGrid;
		OpenCLGridInterface clUpImgGrid = (OpenCLGridInterface)UpImgGrid;
	
		clDownImgGrid.getDelegate().prepareForDeviceOperation();
		clUpImgGrid.getDelegate().prepareForDeviceOperation();
		CLDevice device=clDownImgGrid.getDelegate().getCLDevice();
	
		CLBuffer<FloatBuffer> clmemDownImg=clDownImgGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemUpImg=clUpImgGrid.getDelegate().getCLBuffer();
		runKernel("UpSampling_Y_odd",device,clmemDownImg,clmemUpImg,DownImgGrid.getSize());
		clUpImgGrid.getDelegate().notifyDeviceChange();
	}
	
	public float normL1(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("normL1", device, clmem);

		float l1 = 0.0f;
		while (result.getBuffer().hasRemaining()) {
			l1 += result.getBuffer().get();
		}
		
		result.release();
		return (float) l1;
	}
	
	
	public void gradient(final NumericGrid gridA,final NumericGrid gridB, int xOffset, int yOffset,int zOffset, boolean offsetleft) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runKernel("gradient", device, clmemA,clmemB, new int[]{xOffset, yOffset, zOffset}, new int[]{gridA.getSize()[0], gridA.getSize()[1], gridA.getSize()[2]}, offsetleft);
		clGridA.getDelegate().notifyDeviceChange();
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

