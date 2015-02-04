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
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLTimeVariantSurfaceBSpline extends TimeVariantSurfaceBSpline implements OpenCLEvaluatable{
	
	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> controlPoints;
	
	public OpenCLTimeVariantSurfaceBSpline(
			TimeVariantSurfaceBSpline timeVariantSpline, CLDevice device) {
		this(timeVariantSpline.getSplines(), device);
		this.setTitle(timeVariantSpline.getTitle());
		this.setName(timeVariantSpline.getName());
	}
	
	public OpenCLTimeVariantSurfaceBSpline(ArrayList<SurfaceBSpline> splines, CLDevice device) {
		super(splines);
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initProgram(context);
		handleControlPoints();
	}

	protected void handleControlPoints() {
		int size = this.timeVariantShapes.size() * this.timeVariantShapes.get(0).getControlPoints().size();
		//int count = 0;
		this.controlPoints = context.createFloatBuffer(size*3, Mem.READ_ONLY);
		for (int j = 0; j < timeVariantShapes.size(); j++){
			ArrayList<PointND> controlPoints = timeVariantShapes.get(j).getControlPoints();
			for(int i=0;i<controlPoints.size();i++) {
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(0));
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(1));
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(2));
				//count ++;
			}
		}
		//System.out.println(size + " " + count);
		this.controlPoints.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(this.controlPoints, true);
	}

	
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer){
		this.evaluate(samplingPoints, outputBuffer, timeVariantShapes.get(0).getUKnots().getLen(), timeVariantShapes.get(0).getVKnots().getLen());
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2373375608698854130L;
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV) {
		int elementCount = samplingPoints.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 128);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evaluate3D");
		
		kernel.putArgs(controlPoints, samplingPoints, outputBuffer)
		.putArg(elementCountU).putArg(elementCountV).putArg(timeVariantShapes.size()+4)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
	}

	@Override
	public boolean isTimeVariant() {
		return true;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/