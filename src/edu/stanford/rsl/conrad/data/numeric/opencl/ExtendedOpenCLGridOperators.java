/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.conrad.data.numeric.opencl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.numeric.NumericGrid;

/**
 * This is an extension to 'OpenCLGridOperators'. Here you can implement your own kernels and other ideas.
 * All 'runKernel(...)' methods are not meant to be 'protected' or 'public', they belong to the class OpenCLGridOperators exclusively. 
 * If you move your new code to 'OpenCLGridOperators', you are welcome to use this methods.
 * Unfortunately I have not found a good solution for avoiding the singleton and constructor code. 
 * Good ideas for avoiding this boilerplate code are welcome.
 * 
 * @author Michael Dorner
 */
public class ExtendedOpenCLGridOperators extends OpenCLGridOperators {

	protected ExtendedOpenCLGridOperators() {
		extendedKernelFile = "ExtendedPointwiseOperators.cl";
	}

	static ExtendedOpenCLGridOperators op = new ExtendedOpenCLGridOperators();
	
	public static ExtendedOpenCLGridOperators getInstance() {
		return op;
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
		queue.putReadBuffer(resultBuffer, true);
		queue.finish();

		double sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()) {
			sum += resultBuffer.getBuffer().get();
		}
		
		kernel.rewind();
		resultBuffer.release();
		return sum;
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

}
