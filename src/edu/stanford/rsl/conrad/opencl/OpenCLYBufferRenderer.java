package edu.stanford.rsl.conrad.opencl;

import ij.process.FloatProcessor;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Performs rendering using a y buffer. This enables very fast simulation of volumetric rendering.
 * 
 * @author akmaier
 *
 */
public class OpenCLYBufferRenderer extends OpenCLRenderer {

	CLBuffer<IntBuffer> yBuffer;
	CLBuffer<IntBuffer> xBuffer;
	CLBuffer<IntBuffer> yBufferPointer;
	CLBuffer<IntBuffer> xBufferPointer;
	CLBuffer<IntBuffer> pixelCount;
	CLBuffer<IntBuffer> xPixelCount;
	int yBufferSize;
	private boolean debug = false;
	
	public OpenCLYBufferRenderer(CLDevice device) {
		super(device);
		OpenCLUtil.initYXDraw(context);
	}
	
	public void init (int width, int height){
		super.init(width, height);
		// assumption: we have about 500 hits per slice (on average)
		yBufferSize = height * 200;
		yBuffer = generateIntBuffer(yBufferSize, 3, Mem.READ_WRITE);
		xBuffer = generateIntBuffer(yBufferSize*width, 3, Mem.READ_WRITE);
		yBufferPointer = generateIntBuffer(1, 1, Mem.READ_WRITE);
		xBufferPointer = generateIntBuffer(1, 1, Mem.READ_WRITE);
		pixelCount = generateIntBuffer(height, 1, Mem.READ_WRITE);
		xPixelCount = generateIntBuffer(height*width, 1, Mem.READ_WRITE);
	}
	
	public void resetBuffers(){
		
		yBufferPointer.getBuffer().put(0);
		yBufferPointer.getBuffer().rewind();
		xBufferPointer.getBuffer().put(0);
		xBufferPointer.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(yBuffer, false)
		.putWriteBuffer(xBuffer, false)
		.putWriteBuffer(pixelCount, false)
		.putWriteBuffer(xPixelCount, false)
		.putWriteBuffer(yBufferPointer, false)
		.putWriteBuffer(xBufferPointer, false).finish();
	}
	
	public int drawTriangles(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, int id){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 32);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getYXDrawInstance().createCLKernel("drawTrianglesYBufferLocal");
		
		kernel.putArgs(pointBuffer, yBuffer, yBufferPointer, pixelCount)
		.putArg(width)
		.putArg(id)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.
		
		device.createCommandQueue()
		.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		// TODO: probably causes a memory leak on the OpenCL device
		
		device.createCommandQueue().putReadBuffer(yBufferPointer, true).finish();
		int revan = yBufferPointer.getBuffer().get();
		if (debug ) System.out.println("Final append buffer index: " + revan + " local group size: " +localWorkSize);
		yBufferPointer.getBuffer().rewind();
		
		return revan;

	}
	
	public void readAndShowBuffer(int width, int height, CLBuffer<IntBuffer> screenBuffer, String title){
		float [] array = new float [width*height]; 
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				array[(j*width)+i] = screenBuffer.getBuffer().get();
			}
		}
		screenBuffer.getBuffer().rewind();
		FloatProcessor test = new FloatProcessor(width, height, array, null);
		VisualizationUtil.showImageProcessor(test, title).show();
	}
	
	public void drawScreen(CLBuffer<FloatBuffer> screen){
		// draw to screen buffer:
		int elementCount = height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 1);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getYXDrawInstance().createCLKernel("drawYBufferXBuffer");
		
		kernel.putArgs(yBuffer, pixelCount, xBuffer, xBufferPointer, xPixelCount)
		.putArg(width)
		.putArg(elementCount);
		
		
		device.createCommandQueue()
		.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(xPixelCount, true)
		.putReadBuffer(xBuffer, true)
		.finish();
		
		readAndShowBuffer(width, height, xPixelCount, "xPixelCount");		
		
		kernel = OpenCLUtil.getYXDrawInstance().createCLKernel("drawXBufferScreen");
		
		kernel.putArgs(screen, xBuffer, xPixelCount)
		.putArg(width)
		.putArg(elementCount);
		
		device.createCommandQueue()
		.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		// TODO: probably causes a memory leak on the OpenCL device
	}
	
	public void drawSlice(CLBuffer<FloatBuffer> screen){
		// draw to screen buffer:
		int elementCount = height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getYXDrawInstance().createCLKernel("drawYBufferScreen");
		
		kernel.putArgs(screen, yBuffer, pixelCount)
		.putArg(width)
		.putArg(elementCount);
		
		System.out.println(width);
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/