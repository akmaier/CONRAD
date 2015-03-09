/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */

package edu.stanford.rsl.conrad.data.numeric.opencl;

import java.io.InputStream;
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
	
	/**
	 * Returns the extended OpenCL resource file
	 * @return The corresponding cl kernel file as stream
	 */
	@Override
	protected InputStream getExtendedCLResourceAsStream() {
		return ExtendedOpenCLGridOperators.class.getResourceAsStream(extendedKernelFile);
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
		
		CLBuffer<FloatBuffer> resultBuffer = context.createFloatBuffer(elementCount, Mem.READ_ONLY);
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
	
}
