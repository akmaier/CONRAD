package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.shapes.simple.Cylinder;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;


public class OpenCLCylinder extends
		Cylinder implements OpenCLEvaluatable {

	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> parameter;
	
	
	public OpenCLCylinder(double dx, double dy, double dz, CLDevice device) {
		super(dx, dy, dz);
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(dx, dy, dz);
	}
	
	public OpenCLCylinder(Cylinder c, CLDevice device) {
		this(c.dx, c.dy, c.dz, device);
		this.transform = c.getTransform();
	}
	
	protected void handleParameter(double dx, double dy, double dz){
		this.parameter = context.createFloatBuffer(3, Mem.READ_ONLY);
		this.parameter.getBuffer().put((float)dx);
		this.parameter.getBuffer().put((float)dy);
		this.parameter.getBuffer().put((float)dz);
		this.parameter.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(this.parameter, true);
	}
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV){
		int elementCount = samplingPoints.getBuffer().capacity()/2;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluateCylinder");
		
		kernel.putArgs(parameter, samplingPoints, outputBuffer)
		.putArg(elementCountU).putArg(elementCountV);

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
		
		SimpleMatrix transform = SimpleMatrix.I_4.clone();
		transform.setSubMatrixValue(0, 0, this.transform.getRotation(3));
		transform.setSubColValue(0, 3, this.transform.getTranslation(3));
		OpenCLUtil.transformPoints(outputBuffer, transform, context, device);
	}
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer) {
		int elementCount = samplingPoints.getBuffer().capacity()/2;
		// assume equal length of elementCountU and elementCountV
		evaluate(samplingPoints, outputBuffer, (int) Math.sqrt(elementCount), (int) Math.sqrt(elementCount));
	}
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -578366497108607027L;


	@Override
	public boolean isClockwise() {
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