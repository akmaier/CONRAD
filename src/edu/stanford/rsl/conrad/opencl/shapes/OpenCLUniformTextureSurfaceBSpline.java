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
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLUniformTextureSurfaceBSpline extends
		OpenCLUniformSurfaceBSpline {

	CLImage2d<FloatBuffer> texture;
	
	public OpenCLUniformTextureSurfaceBSpline(String title,
			ArrayList<PointND> controlPoints, SimpleVector uKnots,
			SimpleVector vKnots, CLDevice device) {
		super(title, controlPoints, uKnots, vKnots, device);
		// TODO Auto-generated constructor stub
	}
	
	public OpenCLUniformTextureSurfaceBSpline(SurfaceBSpline spline, CLDevice device) {
		super(spline.getName(), spline.getControlPoints(), spline.getUKnots(), spline.getVKnots(), device);
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
		texture = context.createImage2d(buffer, getVKnots().getLen()-4, getUKnots().getLen()-4, new CLImageFormat(CLImageFormat.ChannelOrder.RGBA, CLImageFormat.ChannelType.FLOAT), CLMemory.Mem.READ_ONLY);
		device.createCommandQueue().putWriteImage(texture, true);
	}
	
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV){
		int elementCount = samplingPoints.getBuffer().capacity()/2;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evaluate2DTextureInterp");
		
		kernel.putArgs(texture, samplingPoints, outputBuffer)
		.putArg(elementCountU).putArg(elementCountV)
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -578366497108607026L;

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/