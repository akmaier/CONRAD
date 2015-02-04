package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Renders OpenCL Splines into a screen buffer.
 * 
 * TODO: move different renderers such as ray casters and zbuffer renderer into different classes, i.e. create new sub classes which are derived from this one. This will reduce duplicate code a lot.
 * 
 * 
 * 
 * @author akmaier
 * @see OpenCLAppendBufferRenderer
 */
public class OpenCLRenderer {

	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> pMatrix;
	protected int width;
	protected int height;
	
	public void release(){
		pMatrix.release();
		pMatrix = null;
	}
	
	public CLBuffer<FloatBuffer> generateFloatBuffer(int width, int height, CLMemory.Mem ... flags){
		CLBuffer<FloatBuffer> screenBuffer = context.createFloatBuffer(width*height, flags);
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				screenBuffer.getBuffer().put(0.f);
			}
		}
		screenBuffer.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(screenBuffer, true).finish();
		return screenBuffer;
	}
	
	public CLBuffer<IntBuffer> generateIntBuffer(int width, int height, CLMemory.Mem ... flags){
		CLBuffer<IntBuffer> screenBuffer = context.createIntBuffer(width*height, flags);
		for (int j = 0; j < height; j++){
			for (int i = 0; i < width; i++){
				screenBuffer.getBuffer().put(0);
			}
		}
		screenBuffer.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(screenBuffer, true).finish();
		return screenBuffer;
	}
	
	public OpenCLRenderer(CLDevice device) {
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initRender(context);	
	}
	
	public void setProjectionMatrix(SimpleMatrix m){
		if (pMatrix == null) {
			pMatrix = context.createFloatBuffer((3*4), Mem.READ_ONLY);
		}
		pMatrix.getBuffer().clear();
		pMatrix.getBuffer().put((float)m.getElement(0,0));
		pMatrix.getBuffer().put((float)m.getElement(0,1));
		pMatrix.getBuffer().put((float)m.getElement(0,2));
		pMatrix.getBuffer().put((float)m.getElement(0,3));
		pMatrix.getBuffer().put((float)m.getElement(1,0));
		pMatrix.getBuffer().put((float)m.getElement(1,1));
		pMatrix.getBuffer().put((float)m.getElement(1,2));
		pMatrix.getBuffer().put((float)m.getElement(1,3));
		pMatrix.getBuffer().put((float)m.getElement(2,0));
		pMatrix.getBuffer().put((float)m.getElement(2,1));
		pMatrix.getBuffer().put((float)m.getElement(2,2));
		pMatrix.getBuffer().put((float)m.getElement(2,3));
		pMatrix.getBuffer().rewind();
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(pMatrix, false).finish();
		clc.release();
	}
	
	public void init (int width, int height){
		this.width = width;
		this.height = height;
	}
	
	public void debugOut(CLBuffer<FloatBuffer> pointBuffer){
		CLCommandQueue clc = device.createCommandQueue();
		clc.putReadBuffer(pointBuffer, true).finish();
		clc.release();
		for (int j=0; j<pointBuffer.getBuffer().capacity();j++){
			System.out.println(pointBuffer.getBuffer().get());
		}
		pointBuffer.getBuffer().rewind();
	}
	
	public void project(CLBuffer<FloatBuffer> pointBuffer){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("project");
		kernel.putArgs(pMatrix, pointBuffer)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.release();
		kernel.release();
	}
	
	public void project(CLBuffer<FloatBuffer> pointBuffer, SimpleVector translation){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("projectTranslate");
		kernel.putArg(pMatrix)
		.putArg((float)translation.getElement(0))
		.putArg((float)translation.getElement(1))
		.putArg((float)translation.getElement(2))
		.putArg(pointBuffer)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		clc.release();
		kernel.release();
	}
	
	public int drawTriangles(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, int id){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("drawTriangles");
		kernel.putArgs(pointBuffer, screenBuffer)
		.putArg(width)
		.putArg(id)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();			
		kernel.release();
		clc.release();
		
		return 0;
	}
	
	public void drawTrianglesZBuffer(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, CLBuffer<IntBuffer> zBuffer, int id){
		int elementCount = pointBuffer.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("drawTrianglesZBuffer");
		kernel.putArgs(pointBuffer, screenBuffer, zBuffer)
		.putArg(width)
		.putArg(id)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.
		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();			
		kernel.release();
		clc.release(); 
	}
	
	public void computeMinMaxValues(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> ranges){
		int elementCount = ranges.getBuffer().capacity() / 4;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("fillMaxMinValues");
		kernel.putArgs(pointBuffer, ranges)
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();			
		kernel.release();
		clc.release();  
	}
	
	public void drawTrianglesRayCast(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> screenBuffer, int controlPoints, int id){
		int elementCount = width*height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		int start = 0;
		int stop = controlPoints;
			
		if (stop > controlPoints) stop = controlPoints;
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("drawTrianglesRayCast");
		kernel.putArgs(pointBuffer, screenBuffer)
		.putArg(width)
		.putArg(controlPoints)
		.putArg(start)
		.putArg(stop)
		.putArg(id)
		.putArg(elementCount);
		start += 1000;
		stop += 1000;

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();			
		kernel.release();
		clc.release(); 		
	}
	
	public void drawTrianglesRayCastRanges(CLBuffer<FloatBuffer> pointBuffer, CLBuffer<FloatBuffer> ranges, CLBuffer<FloatBuffer> screenBuffer, int controlPoints, int id){
		int elementCount = width*height;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getRenderInstance().createCLKernel("drawTrianglesRayCastRanges");
		kernel.putArgs(pointBuffer, ranges, screenBuffer)
		.putArg(width)
		.putArg(controlPoints)
		.putArg(id)
		.putArg(elementCount);
		
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