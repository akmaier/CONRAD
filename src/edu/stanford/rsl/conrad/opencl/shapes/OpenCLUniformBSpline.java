package edu.stanford.rsl.conrad.opencl.shapes;

import java.util.ArrayList;



import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import java.nio.FloatBuffer;

public class OpenCLUniformBSpline extends BSpline implements OpenCLEvaluatable {

	CLContext context;
	CLDevice device;
	CLBuffer<FloatBuffer> controlPoints;
	
	public OpenCLUniformBSpline(BSpline s, CLDevice device) {
		this(s.getControlPoints(), s.getKnots(), device);
	}
	
	public OpenCLUniformBSpline(ArrayList<PointND> controlPoints,
			double[] uVector, CLDevice device) {
		super(controlPoints, uVector);
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initProgram(context);	
		handleControlPoints(controlPoints);
	}
	
	protected void handleControlPoints(ArrayList<PointND> controlPoints){
		this.controlPoints = context.createFloatBuffer(controlPoints.size()*3, Mem.READ_ONLY);
		for(int i=0;i<controlPoints.size();i++) {
			this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(0));
			this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(1));
			this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(2));
		}
		this.controlPoints.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(this.controlPoints, true);
	}
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer){
		int elementCount = samplingPoints.getBuffer().capacity();                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evaluate");
		
		
		kernel.putArgs(controlPoints, samplingPoints, outputBuffer)
		.putArg(getKnots().length)
		.putArg(getControlPoints().size())
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV) {
		evaluate(samplingPoints, outputBuffer);
	}
	
	

	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5895570242152310515L;


	@Override
	public boolean isClockwise() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isTimeVariant() {
		return false;
	}

	

	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/