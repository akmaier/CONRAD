package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;

/**
 * Performs rendering using an append buffer. This enables very fast simulation of volumetric rendering.
 * 
 * @author akmaier
 *
 */
public class OpenCLAppendBufferRenderer extends OpenCLRenderer {

	CLBuffer<IntBuffer> appendBuffer;
	CLBuffer<IntBuffer> appendBufferPointer;
	CLBuffer<IntBuffer> pixelCount;
	int appendBufferSize;
	private boolean debug = false;
	
	public OpenCLAppendBufferRenderer(CLDevice device) {
		super(device);
		OpenCLUtil.initTriangleAppendBufferRender(context);
		//OpenCLUtil.initAppendBufferRender(context);
	}
	
	public void release(){
		super.release();
		appendBuffer.release();
		appendBuffer = null;
		appendBufferPointer.release();
		appendBufferPointer = null;
		pixelCount.release();
		pixelCount = null;
	}
	
	public void init (int width, int height){
		super.init(width, height);
		// assumption: we have about 30 hits per pixel (on average)
		appendBufferSize = (int) (width * height * 100);
		appendBuffer = generateIntBuffer(appendBufferSize, 3, Mem.READ_WRITE);
		appendBufferPointer = generateIntBuffer(1, 1, Mem.READ_WRITE);
		pixelCount = generateIntBuffer(width, height, Mem.READ_WRITE);
	}
	
	public void resetBuffers(){
		
		appendBufferPointer.getBuffer().put(0);
		appendBufferPointer.getBuffer().rewind();
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(appendBuffer, false)
		.putWriteBuffer(pixelCount, false)
		.putWriteBuffer(appendBufferPointer, false).finish();
		clc.release();
	}
	
/*	public int drawTriangles(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, int id){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 3);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getAppendBufferRenderInstance().createCLKernel("drawTrianglesAppendBufferLocal");
		
		kernel.putArgs(pointBuffer, appendBuffer, appendBufferPointer, pixelCount)
		.putArg(width)
		.putArg(id)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.putReadBuffer(appendBufferPointer, true).finish();
		
		int revan = appendBufferPointer.getBuffer().get();
		if (debug ) System.out.println("Final append buffer index: " + revan + " local group size: " +globalWorkSize);
		appendBufferPointer.getBuffer().rewind();
		
		clc.release();
		kernel.release();
	
		return revan;

	}*/
	
	public int drawTrianglesGlobal(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, int id, int elementCountU, int elementCountV, int normalsign){
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 32);  		// Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCountU*elementCountV);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getAppendBufferRenderInstance().createCLKernel("drawTrianglesAppendBufferGlobal");
		
		kernel.putArgs(pointBuffer, appendBuffer, appendBufferPointer, pixelCount)
		.putArg(width)
		.putArg(height)
		.putArg(id)
		.putArg(elementCountU)
		.putArg(elementCountV)
		.putArg(normalsign);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.
		
		//CLCommandQueue clc = device.createCommandQueue((1<<31)); // Sequential execution on intel, necessary for printf: (1<<31) == CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL
		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		
		int revan = 1;
		clc.putReadBuffer(appendBufferPointer, true).finish();
		clc.release();
		revan = appendBufferPointer.getBuffer().get();
		if (debug) {
			System.out.println("Final append buffer index: " + revan);
		}
		appendBufferPointer.getBuffer().rewind();
		
		kernel.release();
		return revan;
		
	}
	
	public void drawScreen(CLBuffer<FloatBuffer> screen){
		// draw to screen buffer:
		int elementCount = width * height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getAppendBufferRenderInstance().createCLKernel("drawAppendBufferScreen");
		
		kernel.putArgs(screen, appendBuffer, pixelCount)
		.putArg(width)
		.putArg(elementCount);
		
		CLCommandQueue clc = device.createCommandQueue();

		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.release();
		kernel.release();
	}
	
	/****************************************
	 * Function to get surface information  *
	 * @param surfaceBuffer buffer to store *
	 * the intersection with smallest depth *
	 ****************************************/
	public void getSurfaceInformation(CLBuffer<FloatBuffer> surfaceBuffer, CLBuffer<FloatBuffer> mu, CLBuffer<IntBuffer> priorities){
		int elementCount = width * height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 512);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		// kernel calls getSurfaceInformationFromAppendBuffer(...) in triangleAppendBuffer.cl
		// TODO Perhaps it is also necessary to add getSurfaceInformationFromAppendBuffer(...)
		// in appendBuffer.cl, appendBufferConvex.cl and appendBufferNonConvexMessedUp.cl.
		
		CLKernel kernel = OpenCLUtil.getAppendBufferRenderInstance().createCLKernel("getSurfaceInformationFromAppendBuffer");
		kernel.putArgs(surfaceBuffer, appendBuffer, pixelCount, mu, priorities)
		.putArg(width)
		.putArg(elementCount);
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(surfaceBuffer, true);
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.release();
		kernel.release();
	}
	
	public void drawScreenMonochromatic(CLBuffer<FloatBuffer> screen, CLBuffer<FloatBuffer> mu, CLBuffer<IntBuffer> priorities){
		// draw to screen buffer:
		int elementCount = width * height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 512);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getAppendBufferRenderInstance().createCLKernel("drawAppendBufferScreenMonochromatic");
		kernel.putArgs(screen, appendBuffer, pixelCount, mu, priorities)
		.putArg(width)
		.putArg(elementCount);
		
		//System.out.println(width);
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.release();
		kernel.release();
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/