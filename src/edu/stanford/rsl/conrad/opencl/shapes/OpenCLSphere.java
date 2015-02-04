/*
 * Copyright (C) 2010-2014 Peter Fischer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.opencl.shapes;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;



import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLSphere extends Sphere implements OpenCLEvaluatable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2607437298428367616L;
	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> parameter;
	
	public OpenCLSphere(double radius, PointND surfaceOrigin, CLDevice device) {
		super(radius, surfaceOrigin);
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(radius);
	}
	public OpenCLSphere(Sphere s, CLDevice device) {
		this(s.getRadius(), s.getCenter(), device);
		this.transform = s.getTransform();
	}
	protected void handleParameter(double radius){
		this.parameter = context.createFloatBuffer(1, Mem.READ_ONLY);
		this.parameter.getBuffer().put((float)radius);
		this.parameter.getBuffer().rewind();
		device.createCommandQueue().putWriteBuffer(this.parameter, true);
	}
	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV){
		int elementCount = samplingPoints.getBuffer().capacity()/2;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluateSphere");
		
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
	@Override
	public boolean isClockwise() {
		return false;
	}
	@Override
	public boolean isTimeVariant() {
		return false;
	}
}
