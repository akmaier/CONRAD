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
import java.util.HashMap;

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

public class OpenCLGridOperators extends NumericGridOperator {

	protected String programFile = "PointwiseOperators.cl";

	protected final int persistentGroupSize = 128;
	protected static CLBuffer<FloatBuffer> persistentResultBuffer = null;

	static HashMap<CLDevice,CLProgram> deviceProgramMap;
	static HashMap<String, HashMap<CLProgram, CLKernel>> programKernelMap;
	protected boolean debug = false;

	protected CLBuffer<FloatBuffer> getPersistentResultBuffer(CLContext context){
		if(persistentResultBuffer==null || persistentResultBuffer.isReleased())
			persistentResultBuffer = context.createFloatBuffer(persistentGroupSize, Mem.WRITE_ONLY);
		else
			persistentResultBuffer.getBuffer().rewind();
		return persistentResultBuffer;
	}

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
		// Problem is that this might be really slow. 
		return new OpenCLGridOperators[]{
				new OpenCLGridOperators()
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
		InputStream inStream = this.getClass().getResourceAsStream(this.programFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
		String content = "";
		String line = br.readLine();
		while (line != null){
			content += line + "\n";
			line = br.readLine();
		};
		return content;
	}


	protected CLKernel getKernel(String name, CLProgram program){
		if (programKernelMap == null){
			programKernelMap = new HashMap<String, HashMap<CLProgram,CLKernel>>();
		}
		HashMap<CLProgram, CLKernel> programMap = programKernelMap.get(name);
		if (programMap == null){
			programMap = new HashMap<CLProgram, CLKernel>();
			programKernelMap.put(name, programMap);
		}
		CLKernel kernel = programMap.get(program);
		if(kernel == null){
			kernel = program.createCLKernel(name);
			programMap.put(program, kernel);
		}else{
			kernel.rewind();
		}
		return kernel;
	}

	/**
	 * TODO:
	 * First version of release; need to implement this better to actually parse the maps and release the individual kernels.
	 */
	public static void release(){
		deviceProgramMap = null;
		programKernelMap = null;
	}


	protected CLProgram getProgram(CLDevice device){
		if(deviceProgramMap == null){
			deviceProgramMap = new HashMap<CLDevice,CLProgram>();
		}
		CLProgram prog = deviceProgramMap.get(device);
		if(prog != null){
			return prog;
		}
		else{
			prog = device.getContext().createProgram(getAllOpenCLGridOperatorProgramsAsString()).build();
			deviceProgramMap.put(device, prog);
			return prog;
		}
	}



	public CLBuffer<FloatBuffer> runUnaryKernel(String name, CLDevice device, CLBuffer<FloatBuffer> clmem){
		int elementCount = clmem.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);

		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmem).putArg(clmemResult).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
		queue.putReadBuffer(clmemResult, true);
		queue.finish();

		kernel.rewind();

		return clmemResult;
	}

	public void runUnaryKernelNoReturn(String name, CLDevice device, CLBuffer<FloatBuffer> clmem){
		int elementCount = clmem.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmem).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
		queue.finish();		
		kernel.rewind();
	}

	public void runBinaryGridKernel(String name, CLDevice device, CLBuffer<FloatBuffer> clmemA, CLBuffer<FloatBuffer> clmemB){
		int elementCount = clmemA.getCLCapacity(); 

		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
		.finish();		

		kernel.rewind();
	}

	public void runBinaryGridScalarKernel(String name, CLDevice device, CLBuffer<FloatBuffer> clmem, float value){
		int elementCount = clmem.getCLCapacity();

		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArgs(clmem).putArg(value).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.rewind();
	}


	@Override
	public void addBy(final NumericGrid grid, float val) {
		if (debug) System.out.println("Bei OpenCL add by value");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice();

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("addByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void addBy(final NumericGrid gridA, final NumericGrid gridB){
		if (debug) System.out.println("Bei OpenCL add by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernel("addBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}

	@Override
	public void subtractBy(final NumericGrid grid, float val) {
		if (debug) System.out.println("Bei OpenCL subtract by value");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("subtractByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void subtractBy(final NumericGrid gridA, final NumericGrid gridB){
		if (debug) System.out.println("Bei OpenCL subtract by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernel("subtractBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}

	/*
	@Override
	public double sum(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL sum");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runUnaryKernel("sum", device, clmem);

		double sum = 0;
		while (result.getBuffer().hasRemaining()){
			sum += result.getBuffer().get();
		}

		result.release();
		return sum;
	}
	 */

	@Override
	public double sum(final NumericGrid grid) {
		if (debug) System.out.println("Bei OpenCL sum");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)grid;
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("sum_persist_kernel", program);

		int localWorkSize = 256;
		int globalWorkSize = 32768;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount,128),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;

		CLBuffer<FloatBuffer> clmemResult = getPersistentResultBuffer(clGridA.getDelegate().getCLContext());

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();

		kernel.putArg(clmemA).putArg(clmemResult).putArg(nperGroup).putArg(nperWorkItem).putArg(elementCount);

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
	public void abs(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL abs");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runUnaryKernelNoReturn("absolute", device, clmem);

		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void exp(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL exp");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runUnaryKernelNoReturn("exponent", device, clmem);

		clGrid.getDelegate().notifyDeviceChange();

	}


	@Override
	public void log(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL log");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runUnaryKernelNoReturn("logarithm", device, clmem);

		clGrid.getDelegate().notifyDeviceChange();

	}

	@Override
	public float max(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL max");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runUnaryKernel("maximum", device, clmem);

		float max = -Float.MAX_VALUE;
		while (result.getBuffer().hasRemaining()){
			max = Math.max(max, result.getBuffer().get());
		}

		result.release();
		return max;
	}

	@Override
	public float min(final NumericGrid grid){
		if (debug) System.out.println("Bei OpenCL min");

		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> result = runUnaryKernel("minimum", device, clmem);

		float min = Float.MAX_VALUE;
		while (result.getBuffer().hasRemaining()){
			min = Math.min(min, result.getBuffer().get());
		}

		result.release();
		return min;
	}

	@Override
	public void multiplyBy(final NumericGrid grid, float val) {
		if (debug) System.out.println("Bei OpenCL multiply by value");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("multiplyByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void copy(final NumericGrid gridA, final NumericGrid gridB) {
		if (debug) System.out.println("Bei OpenCL copy");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernel("copy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}



	@Override
	public void multiplyBy(final NumericGrid gridA, final NumericGrid gridB) {
		if (debug) System.out.println("Bei OpenCL multiply by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernel("multiplyBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}

	@Override
	public void divideBy(final NumericGrid grid, float val) {
		if (debug) System.out.println("Bei OpenCL divide by value");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("divideByVal", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void fill(final NumericGrid grid, float val) {
		if (debug) System.out.println("Bei OpenCL fill");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("fill", device, clmem, val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void removeNegative(final NumericGrid grid) {
		if (debug) System.out.println("Bei OpenCL remove negative");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("minimalValue", device, clmem, 0);
		clGrid.getDelegate().notifyDeviceChange();
	}

	@Override
	public void pow(final NumericGrid grid, double val) {
		if (debug) System.out.println("Bei OpenCL pow");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("power", device, clmem, (float) val);
		clGrid.getDelegate().notifyDeviceChange();
	}

	/*
	@Override
	public double stddev(final NumericGrid grid, double mean) {
		if (debug) System.out.println("Bei OpenCL stddev");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGrid = (OpenCLGridInterface)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();

		int elementCount = clmem.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("stddev", program);

		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmem).putArg(clmemResult).putArg((float)mean).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
		queue.putReadBuffer(clmemResult, true);
		queue.finish();

		kernel.rewind();


		double sum = 0;
		while (clmemResult.getBuffer().hasRemaining()){
			sum += clmemResult.getBuffer().get();
		}

		clmemResult.release();
		return Math.sqrt(sum/ elementCount) ;	
	}
	 */

	@Override
	public double stddev(final NumericGrid grid, double mean) {
		if (debug) System.out.println("Bei OpenCL stddev");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)grid;
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("stddev_persist_kernel", program);

		int localWorkSize = 256;
		int globalWorkSize = localWorkSize*this.persistentGroupSize;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount,this.persistentGroupSize),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;


		CLBuffer<FloatBuffer> clmemResult = getPersistentResultBuffer(clGridA.getDelegate().getCLContext());

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();

		kernel.putArg(clmemA).putArg((float)mean).putArg(clmemResult).putArg(nperGroup).putArg(nperWorkItem).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(clmemResult, true)
		.finish();

		kernel.rewind();


		double sum = 0;
		while (clmemResult.getBuffer().hasRemaining()){
			sum += clmemResult.getBuffer().get();
		}

		// normalization (second moment) 
		sum = Math.sqrt(sum/(double)elementCount);

		return sum;
	}

	/*
	@Override
	public double dotProduct(final NumericGrid gridA, final NumericGrid gridB) {
		if (debug) System.out.println("Bei OpenCL dotProduct");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("dotProduct", program);

		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg(clmemResult).putArg(elementCount);

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


	@Override
	public double dotProduct(final NumericGrid gridA, final NumericGrid gridB) {
		if (debug) System.out.println("Bei OpenCL dotProduct");

		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();


		int elementCount = clmemA.getCLCapacity();
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel("dot_persist_kernel", program);

		int localWorkSize = 256;
		int globalWorkSize = 32768;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount,128),localWorkSize)*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;

		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(128, Mem.WRITE_ONLY);

		CLCommandQueue queue = OpenCLUtil.getStaticCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg(clmemResult).putArg(nperGroup).putArg(nperWorkItem).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
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

	/*
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


	@Override
	public void divideBy(final NumericGrid gridA, final NumericGrid gridB) {
		if (debug) System.out.println("Bei OpenCL divide by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGridInterface clGridA = (OpenCLGridInterface)gridA;
		OpenCLGridInterface clGridB = (OpenCLGridInterface)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernel("divideBy", device, clmemA, clmemB);
		clGridA.getDelegate().notifyDeviceChange();
	}

	static OpenCLGridOperators op = new OpenCLGridOperators();

	public static OpenCLGridOperators getInstance() {
		return op;
	}

}
