/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.numeric.opencl;

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
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * OpenCLGridOperators encapsulates all grid operators which are implemented in OpenCL. 
 * It is implemented as singleton, because all grids share the same operations. 
 * All non-void kernels have to be stored in the property nonVoidKernels, to make sure that memory is allocated on the device.
 */
public class OpenCLGridOperators extends NumericGridOperator {

	private String kernelFile = "PointwiseOperators.cl";
	protected String extendedKernelFile = null;

	private static HashMap<CLDevice,CLProgram> deviceProgramMap;
	private static HashMap<String, HashMap<CLProgram, CLKernel>> programKernelMap;
	private static HashMap<CLDevice, CLCommandQueue> deviceCommandQueueMap;
	private static HashMap<String, Integer> kernelNameLocalSizeMap;
	private static Set<String> nonVoidKernels = new HashSet<String>(Arrays.asList(new String[] {"maximum", "minimum", "sum", "stddev", "dotProduct", "normL1", "countNegativeElements"}) ); 
	
	
	// singleton implementation
	protected OpenCLGridOperators() { 
		programKernelMap = new HashMap<String, HashMap<CLProgram,CLKernel>>();
		deviceProgramMap = new HashMap<CLDevice,CLProgram>();
		deviceCommandQueueMap = new HashMap<CLDevice,CLCommandQueue>();
		kernelNameLocalSizeMap = new HashMap<String, Integer>();
	}
	
	static OpenCLGridOperators op = new OpenCLGridOperators();
	
	public static OpenCLGridOperators getInstance() {
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
					programFile = OpenCLGridOperators.class.getResourceAsStream(kernelFile);
				}
				else {
					 programFile = new SequenceInputStream(OpenCLGridOperators.class.getResourceAsStream(kernelFile), OpenCLGridOperators.class.getResourceAsStream(extendedKernelFile));
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
	 * Method to obtain the extended OpenCL resource
	 * Has to be overwritten by the extended OpenCLGridOperators class
	 * 
	 * @return returns null on purpose
	 */
	protected InputStream getExtendedCLResourceAsStream(){
		return null;
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
	
	/**
	 * Run gradient kernel based on subtraction of two grids with an offsetvalue
	 * @param kernelName    name of the kernel
	 * @param device		CLDevice
	 * @param gridResult	Resulting grid
	 * @param gridABuffer	First Grid
	 * @param gridBBuffer	Second Grid
	 * @param offset		offsetvalue for subtraction
	 * @param volSize		Size of volume in int[], eg 128x128x128
	 * @param offsetleft	set the offset left or right
	 */
	private void runKernel (String kernelName, 
			CLDevice device,CLBuffer<FloatBuffer> gridResult, CLBuffer<FloatBuffer> gridABuffer, 
			CLBuffer<FloatBuffer> gridBBuffer, int[] offset, int[] volSize, boolean offsetleft) { 

		int offleft = offsetleft ? 1:0;
		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();

		int localSize = 2;
		
		kernel	.putArg(gridResult).putArg(gridABuffer).putArg(gridBBuffer)
				.putArg(offset[0]) .putArg(offset[1])  .putArg(offset[2])
				.putArg(volSize[0]).putArg(volSize[1]) .putArg(volSize[2])
				.putArg(offleft);
		
		queue.put3DRangeKernel(	kernel,0,0,0,volSize[0],volSize[1],volSize[2],
											   localSize,localSize,localSize).finish(); 
		
		kernel.rewind();
	}
	
	/**
	 * Run divergence of a grid
	 * @param kernelName    name of the kernel: divergencex = gradient in x direction, divergencey = y-direction , divergencez = zdirectiojn
	 * @param device		CLDevice
	 * @param grid			resulting grid
	 * @param gridBuffer	gradient of the grid
	 * @param offset		offsetvalue
	 * @param volSize		size of the volume in int[] such as 128x128x128
	 * @return
	 */
	private void runKernel(String kernelName, CLDevice device, CLBuffer<FloatBuffer> grid, CLBuffer<FloatBuffer> gridBuffer, int offset, int[] volSize) { 

		OpenCLSetup openCLSetup = new OpenCLSetup(kernelName, device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();

		int localSize = openCLSetup.getLocalSize()/64;	
		
		kernel	.putArg(grid).putArg(gridBuffer)
				.putArg(offset)
				.putArg(volSize[0]).putArg(volSize[1]).putArg(volSize[2]);
		
		queue.put2DRangeKernel(kernel,0,0,volSize[0],volSize[1],localSize,localSize).finish();
		
		kernel.rewind();
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

	
	@Override
	public float stddev(final NumericGrid grid, double mean) {	
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		int elementCount = grid.getNumberOfElements();
		
		CLBuffer<FloatBuffer> gridBuffer = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> resultBuffer = runKernel("stddev", device, gridBuffer, (float)mean);

		double sum = 0.0;
		while (resultBuffer.getBuffer().hasRemaining()){
			sum += resultBuffer.getBuffer().get();
		}
		
		resultBuffer.release();
		return (float)Math.sqrt(sum/elementCount) ;	
	}
	
	
	@Override
	public float dotProduct(final NumericGrid gridA, final NumericGrid gridB) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface

		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;
		CLDevice device = clGridA.getDelegate().getCLDevice(); 
		// TODO check if both live on the same device.

		clGridA.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> gridABuffer = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> gridBBuffer = clGridB.getDelegate().getCLBuffer();
		
		CLBuffer<FloatBuffer> resultBuffer = runKernel("dotProduct", device, gridABuffer, gridBBuffer);
		
		float sum = 0.0f;
		while (resultBuffer.getBuffer().hasRemaining()) {
			sum += resultBuffer.getBuffer().get();
		}
		
		resultBuffer.release();
		return sum;
	}
	
	@Override
	public float sum(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> gridBuffer = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("sum", device, gridBuffer);

		float sum = 0.0f;
		while (result.getBuffer().hasRemaining()) {
			sum += result.getBuffer().get();
		}
		
		result.release();
		return sum;
	}
	
	@Override
	public float max(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("maximum", device, clmem);

		float max = -Float.MAX_VALUE;
		while (result.getBuffer().hasRemaining()) {
			max = Math.max(max, result.getBuffer().get());
		}
		
		result.release();
		return max;
	}
	
	
	@Override
	public float min(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("minimum", device, clmem);

		float min = Float.MAX_VALUE;
		while (result.getBuffer().hasRemaining()) {
			min = Math.min(min, result.getBuffer().get());
		}
		
		result.release();
		return min;
	}
	
	@Override
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
	
	@Override
	public int countNegativeElements(final NumericGrid grid) {

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("countNegativeElements", device, clmem);

		int count = 0;
		while (result.getBuffer().hasRemaining()) {
			count += (int)result.getBuffer().get();
		}
		
		result.release();
		return count;
	}
	
	@Override
	public void abs(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("absolute", device, clmem);
		
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public void exp(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("expontial", device, clmem);
		
		clGrid.getDelegate().notifyDeviceChange();
	}
		
	
	@Override
	public void log(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("logarithm", device, clmem);
		
		clGrid.getDelegate().notifyDeviceChange();
	}
	

	@Override
	public void addBy(final NumericGrid grid, float val) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("addByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public void addBy(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runKernel("addBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void addBySave(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runKernel("addBySave", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void addBySave(final NumericGrid gridA, float value) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;

		clGridA.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();

		runKernel("addBySaveVal", device, clmemA, value);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void subtractBy(final NumericGrid grid, float val) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("subtractByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public void subtractBy(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runKernel("subtractBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void subtractBySave(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runKernel("subtractBySave", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void subtractBySave(final NumericGrid gridA, float value) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;

		clGridA.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();

		runKernel("subtractBySaveVal", device, clmemA, value);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void multiplyBy(final NumericGrid grid, float val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("multiplyByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void multiplyBy(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runKernel("multiplyBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void multiplyBySave(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runKernel("multiplyBySave", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void multiplyBySave(final NumericGrid grid, float val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("multiplyBySaveVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void divideBy(final NumericGrid grid, float val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("divideByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void divideBy(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runKernel("divideBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	

	@Override
	public void divideBySave(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runKernel("divideBySave", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void divideBySave(final NumericGrid gridA, float value) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;

		clGridA.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		
		runKernel("divideBySaveVal", device, clmemA, value);
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void copy(final NumericGrid gridA, final NumericGrid gridB) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runKernel("copyGrid", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}
		
	@Override
	public void fill(final NumericGrid grid, float val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("fill", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void fillInvalidValues(final NumericGrid grid, float val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("fillInvalidValues", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void fillInvalidValues(final NumericGrid grid) {		
		fillInvalidValues(grid, 0);
	}
	
	@Override
	public void removeNegative(final NumericGrid grid) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("minimalValue", device, clmem, 0);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void pow(final NumericGrid grid, double val) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runKernel("power", device, clmem, (float)val);
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * convert grid2d[] to grid3d 
	*/
	
	public void convert2DArrayTo3D(NumericGrid gridRes,final NumericGrid[] grid) {
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA[] = new OpenCLGridInterface[grid.length];
		CLBuffer<FloatBuffer> clmemA = OpenCLUtil.getStaticContext().createFloatBuffer(gridRes.getSize()[0]*gridRes.getSize()[1]*gridRes.getSize()[2], Mem.READ_WRITE);
		
		CLCommandQueue queueHelp = OpenCLUtil.getStaticContext().getMaxFlopsDevice().createCommandQueue();
		System.out.println();
		for(int i = 0; i < grid.length;i++){
			clGridA[i] = (OpenCLGridInterface)grid[i];
			clGridA[i].getDelegate().prepareForDeviceOperation();
			queueHelp.putCopyBuffer(clGridA[i].getDelegate().getCLBuffer(), clmemA,0,i*(int)clGridA[i].getDelegate().getCLBuffer().getCLSize(),clGridA[i].getDelegate().getCLBuffer().getCLSize(),null).finish();
			clGridA[i].getDelegate().notifyDeviceChange();
		}
			
		OpenCLGridInterface clGridRes = (OpenCLGridInterface)gridRes;
		clGridRes.getDelegate().prepareForDeviceOperation();
		queueHelp.putCopyBuffer(clmemA, clGridRes.getDelegate().getCLBuffer()).finish();
		
		clGridRes.getDelegate().notifyDeviceChange();
	}
	
	@Override
	/**
	 * subtract of two grids with given offset
	 * @param gridRes = result grid
	 * @param gridA, gridB = input grids
	 * @param x,y,z Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void subtractOffset(NumericGrid gridResult, final NumericGrid gridA,final NumericGrid gridB, int xOffset, int yOffset,int zOffset, boolean offsetleft) {
		if( (xOffset != 0 && (yOffset != 0 || zOffset != 0)) || (yOffset != 0 && zOffset != 0)) {
			System.err.println("It is not possible to calculate a multidimensional gradient problem");
		}
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;
		OpenCLGridInterface clGridResult = (OpenCLGridInterface)gridResult;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		clGridResult.getDelegate().prepareForDeviceOperation();
		
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemRes = clGridResult.getDelegate().getCLBuffer();
		clGridResult.getDelegate().prepareForDeviceOperation();
	
		runKernel("gradient", device,clmemRes, clmemA,clmemB, new int[]{xOffset, yOffset, zOffset}, new int[]{gridA.getSize()[0], gridA.getSize()[1], gridA.getSize()[2]},offsetleft);

		clGridResult.getDelegate().notifyDeviceChange();
	}
	
	@Override
	/**
	 * gradient in x-,y- or z-direction
	 * @param gridRes = result grid
	 * @param grid = input grid
	 * @param value = offsetvalue 
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void gradX(NumericGrid gridRes, final NumericGrid grid, int value, boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,value,0,0,offsetleft);
	}
	@Override
	public void gradY(NumericGrid gridRes, final NumericGrid grid, int value,boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,0,value,0,offsetleft);
	}
	@Override
	public void gradZ(NumericGrid gridRes, final NumericGrid grid, int value, boolean offsetleft) {
		subtractOffset(gridRes,grid,grid,0,0,value,offsetleft);
	}
	
	/**
	 * calculates the divergence in x-,y-, or z-direction
	 * @param gridRes = result grid
	 * @param grid = input grid
	 * @param x,y,z Offset = offsetvalue in x,y,and z direction
	 * @param offsetleft = true if left offset, false if right offset
	 */
	public void divergence(NumericGrid gridRes,final NumericGrid grid, int xOffset, int yOffset,int zOffset, boolean offsetleft) {

		if(xOffset == 0 && yOffset == 0 && zOffset == 0)
			System.err.println("No offset value chosen");
		else if( (xOffset != 0 && (yOffset != 0 || zOffset != 0)) || (yOffset != 0 && zOffset != 0))
			System.err.println("Too many divergence offsets chosen");
		else{

			OpenCLGridInterface clGrid = (OpenCLGridInterface)gridRes;
			OpenCLGridInterface gridBuf = (OpenCLGridInterface)grid;
			CLBuffer<FloatBuffer> clmemGrid = clGrid.getDelegate().getCLBuffer();
			clGrid.getDelegate().prepareForDeviceOperation();

			CLDevice device = clGrid.getDelegate().getCLDevice(); 

			//x:0 y:1 z:2
			if(xOffset != 0){
				gridRes.getGridOperator().gradX(gridRes,grid,xOffset,offsetleft);
				CLBuffer<FloatBuffer> clmemGridBuf = gridBuf.getDelegate().getCLBuffer();
				runKernel("divergencex", device,clmemGrid,clmemGridBuf, xOffset, new int[]{gridRes.getSize()[0], gridRes.getSize()[1],gridRes.getSize()[2]});
			}
			else if(yOffset != 0){
				gridRes.getGridOperator().gradY(gridRes,grid,yOffset,offsetleft);
				CLBuffer<FloatBuffer> clmemGridBuf = gridBuf.getDelegate().getCLBuffer();
				runKernel("divergencey", device,clmemGrid,clmemGridBuf, yOffset, new int[]{gridRes.getSize()[0], gridRes.getSize()[1],gridRes.getSize()[2]});
			}
			else if(zOffset != 0){
				gridRes.getGridOperator().gradZ(gridRes,grid,zOffset,offsetleft);
				CLBuffer<FloatBuffer> clmemGridBuf = gridBuf.getDelegate().getCLBuffer();
				runKernel("divergencez", device, clmemGrid,clmemGridBuf, zOffset, new int[]{gridRes.getSize()[0], gridRes.getSize()[1],gridRes.getSize()[2]});
			}
		}
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
	public static OpenCLGridOperators[] getAllInstances(){
		// TODO: replace with automatic search on java class path
		// Problem is that this might be really slow. // Comment by Michael Dorner: It IS very slow.
		return new OpenCLGridOperators[]{
				new OpenCLGridOperators() 	// Comment by Michael Dorner: GridOperators are singletons. Therefore we should see any constructor in there. Additionally, why creating an array with only one entry?

		};
	}

	
	/**
	 * Obtains all OpenCLGridOperators instances and concatenates all related 
	 * cl-source files to one long string
	 * @return Concatenated cl-source code
	 */
	public String getAllOpenCLGridOperatorProgramsAsString(){
		String out = "";
		OpenCLGridOperators[] instances = getAllInstances();
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
