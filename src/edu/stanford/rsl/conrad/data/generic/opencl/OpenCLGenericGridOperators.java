/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.opencl;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

import edu.stanford.rsl.conrad.data.generic.GenericGrid;
import edu.stanford.rsl.conrad.data.generic.GenericGridOperatorInterface;
import edu.stanford.rsl.conrad.data.generic.datatypes.Gridable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public abstract class OpenCLGenericGridOperators<T extends Gridable<T>> implements GenericGridOperatorInterface<T>{

	
	static HashMap<CLDevice,CLProgram> deviceProgramMap;
	static HashMap<String, HashMap<CLProgram, CLKernel>> programKernelMap;
	protected boolean debug = true;
	
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
	
	protected abstract InputStream getProgramFileAsStream();
	
	protected CLProgram getProgram(CLDevice device){
		if(deviceProgramMap == null){
			deviceProgramMap = new HashMap<CLDevice,CLProgram>();
		}
		CLProgram prog = deviceProgramMap.get(device);
		if(prog != null){
			return prog;
		}
		else{
			InputStream programFile = getProgramFileAsStream();
			try {
				prog = device.getContext().createProgram(programFile).build();
			} catch (IOException e) {
				e.printStackTrace();
				return null;
			}
			deviceProgramMap.put(device, prog);
			return prog;
		}
	}
	
	public CLBuffer<FloatBuffer> runBinaryGridKernel(String name, CLDevice device, CLBuffer<FloatBuffer> clmemA, CLBuffer<FloatBuffer> clmemB, int elementCount) {
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		
		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;
		
		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize*2, Mem.WRITE_ONLY);
		
		CLCommandQueue queue = device.createCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg(clmemResult).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		queue.putReadBuffer(clmemResult, true);
		queue.release();		
		kernel.rewind();

		return clmemResult;
	}
	
	public CLBuffer<FloatBuffer> runUnaryKernel(String name, CLDevice device,
			CLBuffer<FloatBuffer> clmem, int elementCount){
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		
		int localWork = 32;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount/localWork);
		localWork = (elementCount / globalWorkSize)+1;
		
		CLBuffer<FloatBuffer> clmemResult = device.getContext().createFloatBuffer(globalWorkSize*2, Mem.WRITE_ONLY);
		
		CLCommandQueue queue = device.createCommandQueue();
		kernel.putArg(clmem).putArg(clmemResult).putArg(elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		queue.putReadBuffer(clmemResult, true);
		queue.release();
		kernel.rewind();
		
		return clmemResult;
	}
	
	public void runUnaryKernelNoReturn(String name, CLDevice device, CLBuffer<FloatBuffer> clmem, int elementCount){
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = device.createCommandQueue();
		kernel.putArg(clmem).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		queue.release();		
		kernel.rewind();
	}
	
	public void runBinaryGridKernelNoReturn(String name, CLDevice device, CLBuffer<FloatBuffer> clmemA, CLBuffer<FloatBuffer> clmemB, int elementCount){

		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = device.createCommandQueue();
		kernel.putArg(clmemA).putArg(clmemB).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		queue.release();		
		kernel.rewind();
	}
	
	public void runBinaryGridScalarKernel(String name, CLDevice device, CLBuffer<FloatBuffer> clmem, T value, int elementCount){

		float[] argument = value.getAsFloatArray();
		CLBuffer<FloatBuffer> argBuffer = device.getContext().createFloatBuffer(argument.length, Mem.READ_ONLY);
		argBuffer.getBuffer().put(argument);
		argBuffer.getBuffer().rewind();
		
		CLProgram program = getProgram(device);
		CLKernel kernel = getKernel(name, program);
		CLCommandQueue queue = device.createCommandQueue();
		kernel.putArgs(clmem).putArg(argBuffer).putArg(elementCount);

		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);	
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);

		queue.putWriteBuffer(argBuffer, true).put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		queue.release();
		kernel.rewind();
	}


	@Override
	public void addBy(final GenericGrid<T> grid, T val) {
		if (debug) System.out.println("Bei OpenCL add by value");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("addByVal", device, clmem, val, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void addBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB){
		if (debug) System.out.println("Bei OpenCL add by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernelNoReturn("addBy", device, clmemA, clmemB, gridA.getNumberOfElements());
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void subtractBy(final GenericGrid<T> grid, T val) {
		if (debug) System.out.println("Bei OpenCL subtract by value");
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("subtractByVal", device, clmem, val, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void subtractBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB){
		if (debug) System.out.println("Bei OpenCL subtract by");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();

		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();

		runBinaryGridKernelNoReturn("subtractBy", device, clmemA, clmemB, gridA.getNumberOfElements());
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public void multiplyBy(final GenericGrid<T> grid, T val) {
		if (debug) System.out.println("Bei OpenCL multiply by value");
		
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("multiplyByVal", device, clmem, val, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}

	
	@Override
	public void multiplyBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB) {
		if (debug) System.out.println("Bei OpenCL multiply by");
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runBinaryGridKernelNoReturn("multiplyBy", device, clmemA, clmemB, gridA.getNumberOfElements());
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void divideBy(final GenericGrid<T> grid, T val) {
		if (debug) System.out.println("Bei OpenCL divide by value");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("divideByVal", device, clmem, val, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	
	@Override
	public void divideBy(final GenericGrid<T> gridA, final GenericGrid<T> gridB) {
		if (debug) System.out.println("Bei OpenCL divide by");
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runBinaryGridKernelNoReturn("divideBy", device, clmemA, clmemB, gridA.getNumberOfElements());
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void copy(final GenericGrid<T> gridA, final GenericGrid<T> gridB) {
		if (debug) System.out.println("Bei OpenCL copy");
		// not possible to have a grid that is not implementing OpenCLGridInterface
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		
		runBinaryGridKernelNoReturn("copy", device, clmemA, clmemB, gridA.getNumberOfElements());
		clGridA.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public void fill(final GenericGrid<T> grid, T val) {
		if (debug) System.out.println("Bei OpenCL fill");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		runBinaryGridScalarKernel("fill", device, clmem, val, grid.getNumberOfElements());
		clGrid.getDelegate().notifyDeviceChange();
	}
	
	@Override
	public T sum(final GenericGrid<T> grid) {
		if (debug) System.out.println("Bei OpenCL sum");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> res = runUnaryKernel("sum", device, clmem, grid.getNumberOfElements());
		return getSum(res);
	}
	
	public abstract T getSum(CLBuffer<FloatBuffer> clRes);

	@Override
	public T min(final GenericGrid<T> grid) {
		if (debug) System.out.println("Bei OpenCL min");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> res = runUnaryKernel("minimum", device, clmem, grid.getNumberOfElements());
		return getMin(res);
	}
	
	public abstract T getMin(CLBuffer<FloatBuffer> clRes);
	
	

	@Override
	public T max(final GenericGrid<T> grid) {
		if (debug) System.out.println("Bei OpenCL max");
		
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGrid = (OpenCLGenericGridInterface<T>)grid;
		CLDevice device = clGrid.getDelegate().getCLDevice(); 

		clGrid.getDelegate().prepareForDeviceOperation();
		
		CLBuffer<FloatBuffer> clmem = clGrid.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> res = runUnaryKernel("maximum", device, clmem, grid.getNumberOfElements());
		return getMax(res);
	}
	
	public abstract T getMax(CLBuffer<FloatBuffer> clRes);

	
	@Override
	public T dotProduct(final GenericGrid<T> gridA, final GenericGrid<T> gridB) {
		if (debug) System.out.println("Bei OpenCL dotProduct");
		// not possible to have a grid that is not implementing OpenCLGenericGridInterface<T>
		OpenCLGenericGridInterface<T> clGridA = (OpenCLGenericGridInterface<T>)gridA;
		OpenCLGenericGridInterface<T> clGridB = (OpenCLGenericGridInterface<T>)gridB;

		clGridA.getDelegate().prepareForDeviceOperation();
		clGridB.getDelegate().prepareForDeviceOperation();
		
		// TODO check if both live on the same device.
		CLDevice device = clGridA.getDelegate().getCLDevice(); 

		CLBuffer<FloatBuffer> clmemA = clGridA.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> dummy = clGridA.getDelegate().getCLContext().createFloatBuffer(clmemA.getCLCapacity(),Mem.READ_WRITE);
		clGridA.getDelegate().getCLDevice().createCommandQueue().putWriteBuffer(dummy, true).release();
		runBinaryGridKernelNoReturn("copy", device, dummy, clmemA, gridA.getNumberOfElements());
		
		CLBuffer<FloatBuffer> clmemB = clGridB.getDelegate().getCLBuffer();
		runBinaryGridKernelNoReturn("multiplyBy", device, dummy, clmemB, gridA.getNumberOfElements());
		CLBuffer<FloatBuffer> clRes = runUnaryKernel("sum", device, dummy, gridA.getNumberOfElements());
		return getSum(clRes);
	}
}
