/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.numeric.opencl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
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

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

/**
 * OpenCLGridOperators encapsulates all grid operators which are implemented in OpenCL. 
 * It is implemented as singleton, because all grids share the same operations. 
 * All non-void kernels have to be stored in the property nonVoidKernels, to make sure that memory is allocated on the device.
 */
public class OpenCLGridOperators extends NumericGridOperator {

	private String kernelFile = "PointwiseOperators.cl";

	private static HashMap<CLDevice,CLProgram> deviceProgramMap;
	private static HashMap<String, HashMap<CLProgram, CLKernel>> programKernelMap;
	private static HashMap<CLDevice, CLCommandQueue> deviceCommandQueueMap;
	private static HashMap<String, Integer> kernelNameLocalSizeMap;
	private static Set<String> nonVoidKernels = new HashSet<String>(Arrays.asList(new String[] {"maximum", "minimum", "sum", "stddev", "dotProduct"}) ); 
	
	
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
	private class OpenCLSetup {
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
				InputStream programFile = OpenCLGridOperators.class.getResourceAsStream(kernelFile);
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
			queue.finish();
			queue.putReadBuffer(resultBuffer, true);
			
			
		}
		else {
			kernel.putArg(gridBuffer).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.finish();
		}
		
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
			queue.finish();
			queue.putReadBuffer(resultBuffer, true);
		}
		else {
			kernel.putArg(gridBuffer).putArg(value).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.finish();
		}
		
		kernel.rewind();
		return resultBuffer;
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
			queue.finish();
			queue.putReadBuffer(resultBuffer, true);
		}
		else {
			kernel.putArg(gridABuffer).putArg(gridBBuffer).putArg(elementCount);
			queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
			queue.finish();
		}
		
		kernel.rewind();
		return resultBuffer;
	}


	@Override
	public double stddev(final NumericGrid grid, double mean) {	
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
		return Math.sqrt(sum/elementCount) ;	
	}
	
	
	@Override
	public double dotProduct(final NumericGrid gridA, final NumericGrid gridB) {		
		// not possible to have a grid that is not implementing OpenCLGridInterface

		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;
		CLDevice device = clGridA.getDelegate().getCLDevice(); 
		// TODO check if both live on the same device.

		clGridA.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> gridABuffer = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> gridBBuffer = clGridB.getDelegate().getCLBuffer();
		
		CLBuffer<FloatBuffer> resultBuffer = runKernel("dotProduct", device, gridABuffer, gridBBuffer);
		
		double sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()) {
			sum += resultBuffer.getBuffer().get();
		}
		
		resultBuffer.release();
		return sum;
	}
	
	
	
	@Override
	public double sum(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> gridBuffer = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runKernel("sum", device, gridBuffer);

		double sum = 0.0;
		while (result.getBuffer().hasRemaining()) {
			sum += result.getBuffer().get();
		}
		
		result.release();
		return sum;
	}
	
	
	public double sumGlobalMemory(final NumericGrid grid) {
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> gridBuffer = clGrid.getDelegate().getCLBuffer();

		int elementCount = gridBuffer.getCLCapacity(); 
		
		OpenCLSetup openCLSetup = new OpenCLSetup("sumGlobalMemory", device);

		CLKernel kernel = openCLSetup.getKernel();
		CLCommandQueue queue = openCLSetup.getCommandQueue();
		CLContext context = openCLSetup.getContext();
		
		int localSize = openCLSetup.getLocalSize();		
		int globalSize = openCLSetup.getGlobalSize(elementCount);
		
		CLBuffer<FloatBuffer> resultBuffer = context.createFloatBuffer(globalSize, Mem.READ_ONLY);
		kernel.putArg(gridBuffer).putArg(resultBuffer).putArg(elementCount);
		queue.put1DRangeKernel(kernel, 0, globalSize, localSize);
		queue.finish();
		queue.putReadBuffer(resultBuffer, true);

		double sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()) {
			sum += resultBuffer.getBuffer().get();
		}
		
		kernel.rewind();
		resultBuffer.release();
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
	
	//TODO: weightedDotProduct do not use the new OpenCL implementation and therefore they are commented out. Update this functions
/*
	@Override
	public double weightedDotProduct(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double addGrid2) {
		if (debug) System.out.println("Bei OpenCL weightedDotProduct");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)grid1;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)grid2;

		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("weightedDotProduct_persist_kernel", program);

		int localWorkSize = 256;
		int globalWorkSize = localWorkSize*this.persistentGroupSize;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount,this.persistentGroupSize),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;

		CLBuffer<FloatBuffer> clmemResult = getPersistentResultBuffer(clGridA.getDelegate().getCLContext());

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();

		kernel.putArg(clmemA).putArg(clmemB).putArg((float)weightGrid2).putArg((float)addGrid2).putArg(clmemResult).putArg(nperGroup).putArg(nperWorkItem).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(clmemResult, true)
		.finish();

		kernel.rewind();


		double sum = 0;
		while (clmemResult.getBuffer().hasRemaining()){
			sum += clmemResult.getBuffer().get();
		}

		return sum;
	}

	@Override
	public double weightedSSD(NumericGrid grid1, NumericGrid grid2, double weightGrid2, double addGrid2) {
		if (debug) System.out.println("Bei OpenCL weightedSSD");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)grid1;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)grid2;

		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("weightedSSD_persist_kernel", program);

		int localWorkSize = 256;
		int globalWorkSize = 32768;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount,128),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(128, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();

		kernel.putArg(clmemA).putArg(clmemB).putArg((float)weightGrid2).putArg((float)addGrid2).putArg(clmemResult).putArg(nperGroup).putArg(nperWorkItem).putArg(elementCount);

		queue.putWriteBuffer(clmemResult, true)
		.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(clmemResult, true)
		.finish();

		kernel.rewind();


		double sum = 0;
		while (clmemResult.getBuffer().hasRemaining()){
			sum += clmemResult.getBuffer().get();
		}

		clmemResult.release();
		return sum;
	}

	@Override
	public double weightedSSD(NumericGrid grid1, NumericGrid grid2, double weightGrid2) {
		if (debug) System.out.println("Bei OpenCL weightedSSD");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)grid1;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)grid2;

		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("weightedSSD", program);

		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg((float)weightGrid2).putArg(clmemResult).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
		queue.putReadBuffer(clmemResult, true);
		queue.finish();

		kernel.rewind();

		double sum = 0;
		while (clmemResult.getBuffer().hasRemaining()){
			sum += clmemResult.getBuffer().get();
		}
		clmemResult.release();
		return sum;
	}
	 */
	
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

