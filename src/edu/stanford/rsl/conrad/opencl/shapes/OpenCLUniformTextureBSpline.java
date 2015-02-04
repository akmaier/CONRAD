package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLMemory.Mem;



import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLUniformTextureBSpline extends OpenCLUniformBSpline {

	CLImage2d<FloatBuffer> texture;
	
	public OpenCLUniformTextureBSpline(ArrayList<PointND> controlPoints,
			double[] uVector, CLDevice device) {
		super(controlPoints, uVector, device);
	}

	protected void handleControlPoints(ArrayList<PointND> controlPoints){
		this.controlPoints = context.createFloatBuffer(controlPoints.size()*4, Mem.READ_ONLY);
		FloatBuffer buffer= this.controlPoints.getBuffer();
		for(int i=0;i<controlPoints.size();i++) {
			buffer.put((float)controlPoints.get(i).get(0));
			buffer.put((float)controlPoints.get(i).get(1));
			buffer.put((float)controlPoints.get(i).get(2));
			buffer.put(0f);
		}
		buffer.rewind();
		texture = context.createImage2d(buffer, controlPoints.size(), 1, new CLImageFormat(CLImageFormat.ChannelOrder.RGBA, CLImageFormat.ChannelType.FLOAT), CLMemory.Mem.READ_ONLY);
		device.createCommandQueue().putWriteImage(texture, true);
	}
	
	/**
	 * Example how to read a texture at float coordinates. Note that the pixel center is at (x.5, y.5) to get the exact value.
	 * @param outputBuffer
	 */
	public void readControlPointsFromTextureInterpolate(CLBuffer<FloatBuffer> outputBuffer){
		int elementCount = getControlPoints().size(); 
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("readTextureInterp");
		kernel.putArg(texture)
		.putArg(outputBuffer)
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	/**
	 * Example how to read a texture at integer coordinates. Note that the pixel center is at (x.0, y.0) to get the exact value.
	 * @param outputBuffer
	 */
	public void readControlPointsFromTexture(CLBuffer<FloatBuffer> outputBuffer){
		int elementCount = getControlPoints().size(); 
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("readTexture");
		kernel.putArg(texture)
		.putArg(outputBuffer)
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer){
		int elementCount = samplingPoints.getBuffer().capacity();               // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  		// Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evalTextureInterp");
		kernel.putArgs(texture, samplingPoints, outputBuffer)
		.putArg(getKnots().length)
		.putArg(getControlPoints().size())
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	public void evaluateNoInterp(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer){
		int elementCount = samplingPoints.getBuffer().capacity();                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evalTexture");
		kernel.putArg(texture)
		.putArgs(samplingPoints, outputBuffer)
		.putArg(getKnots().length)
		.putArg(getControlPoints().size())
		.putArg(elementCount);
		

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	
	private static final long serialVersionUID = -3769875288043610795L;

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/