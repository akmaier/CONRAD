package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLUniformSurfaceBSpline extends SurfaceBSpline implements OpenCLEvaluatable{

	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> controlPoints;
	
	
	public OpenCLUniformSurfaceBSpline(SurfaceBSpline spline, CLDevice device) {
		super(spline.getName(), spline.getControlPoints(), spline.getUKnots(), spline.getVKnots());
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initProgram(context);	
		handleControlPoints(spline.getControlPoints());
	}

	public OpenCLUniformSurfaceBSpline(String title, ArrayList<PointND> controlPoints,
			double[] uKnots, double[] vKnots, CLDevice device) {
		super(title, controlPoints, uKnots, vKnots);
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initProgram(context);	
		handleControlPoints(controlPoints);
	}

	public OpenCLUniformSurfaceBSpline(String title, ArrayList<PointND> controlPoints,
			SimpleVector uKnots, SimpleVector vKnots, CLDevice device) {
		super(title, controlPoints, uKnots, vKnots);
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
	
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV){
		int elementCount = samplingPoints.getBuffer().capacity()/2;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evaluate2D");
		
		kernel.putArgs(controlPoints, samplingPoints, outputBuffer)
		.putArg(elementCountU).putArg(elementCountV)
		.putArg(elementCount);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer){
		evaluate(samplingPoints, outputBuffer, uKnots.getLen(), vKnots.getLen());
	}
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7502977258317985586L;


	@Override
	public boolean isTimeVariant() {
		return false;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/