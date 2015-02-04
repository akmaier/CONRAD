/*
 * Copyright (C) 2014 Zijia Guo, Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Pyramid;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLPyramid extends Pyramid implements OpenCLEvaluatable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8658765689476697398L;
	protected CLDevice device;
	protected CLContext context;
	protected CLBuffer<FloatBuffer> parameter;

	public OpenCLPyramid(CLDevice device){
		super();
	}
	
	public OpenCLPyramid(double dx, double dy, double dz, CLDevice device){
		super(dx, dy, dz);
		this.device = device;
		this.context = device.getContext();
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(dx, dy, dz);
	}
	
	public OpenCLPyramid(Pyramid p, CLDevice device){
		this(-p.getMin().get(0), -p.getMin().get(1), -p.getMin().get(2), device);
		this.transform = p.getTransform();
	}
	
	public void handleParameter(double dx, double dy, double dz){
		//double a = 0.5*(max.get(0) - min.get(0))/(max.get(2) - min.get(2));
		//double b = 0.5*(max.get(1) - min.get(1))/(max.get(2) - min.get(2));
		float a = (float)(0.5*(max.get(0) - min.get(0))/(max.get(2) - min.get(2)));
		float b = (float)(0.5*(max.get(1) - min.get(1))/(max.get(2) - min.get(2)));
		
		this.parameter = context.createFloatBuffer(5, Mem.READ_ONLY);
		this.parameter.getBuffer().put((float)dx);  // parameter[0]
		this.parameter.getBuffer().put((float)dy);  // parameter[1]
		this.parameter.getBuffer().put((float)dz);  // parameter[2]
		this.parameter.getBuffer().put(a);  // parameter[3]
		this.parameter.getBuffer().put(b);  // parameter[4]
		this.parameter.getBuffer().rewind();
		
		//device.createCommandQueue().putWriteBuffer(this.parameter, true);
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(this.parameter, true).finish();
		clc.release();

	}
	
	@Override
	public boolean isClockwise() {
		// TODO Auto-generated method stub
		return true;
	}


	@Override
	public boolean isTimeVariant() {
		// TODO Auto-generated method stub
		return false;
	}


	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints,
			CLBuffer<FloatBuffer> outputBuffer) {
		// TODO Auto-generated method stub
		int elementCount = samplingPoints.getBuffer().capacity()/2;   // capacity? 2 or 3?
		evaluate(samplingPoints, outputBuffer, (int)Math.sqrt(elementCount), (int)Math.sqrt(elementCount));  // how to calculate U and V?
	
	}

	
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints,
			CLBuffer<FloatBuffer> outputBuffer, int elementCountU,
			int elementCountV) {
		// TODO Auto-generated method stub
		int elementCount = samplingPoints.getBuffer().capacity()/2;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);  // to guaranty that global size is interger multiple of group size (local size) 
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluatePyramid");
		
		kernel.putArgs(parameter, samplingPoints, outputBuffer).putArg(elementCountU).putArg(elementCountV);
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
		kernel.release();
		clc.release();
		
		// SimpleMatrix transform
		SimpleMatrix transform = SimpleMatrix.I_4.clone();
		transform.setSubMatrixValue(0, 0, this.transform.getRotation(3));
		transform.setSubColValue(0, 3, this.transform.getTranslation(3));
		OpenCLUtil.transformPoints(outputBuffer, transform, context, device);
	}
	
	public static void main(String [] args){
		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		int u = 2;
		int v = 8;
		
		Pyramid pyramid = new Pyramid(12,12,12);
		ArrayList<PointND> cpu = pyramid.getPointCloud(u,v);
		
		int numPoints = u*v;
		
		OpenCLPyramid clpyramid = new OpenCLPyramid(pyramid, device);
		
		CLBuffer<FloatBuffer> samplingPoints = OpenCLUtil.generateSamplingPoints(u, v, context, device);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(3*numPoints, Mem.READ_WRITE);
		
		clpyramid.evaluate(samplingPoints, outputBuffer,u,v);
		
		CLCommandQueue queue = device.createCommandQueue();
		queue.putReadBuffer(outputBuffer, true);
		queue.release();
		
		ArrayList<PointND> gpu = new ArrayList<PointND>(); 
		
		double error =0;
		
		for (int i=0; i< numPoints; i++){
			PointND point = new PointND(outputBuffer.getBuffer().get(), outputBuffer.getBuffer().get(), outputBuffer.getBuffer().get());
			gpu.add(point);
			
			error += point.euclideanDistance(cpu.get(i));
		}
		
		samplingPoints.release();
		outputBuffer.release();
		
		PointCloudViewer pcv = new PointCloudViewer("gpu points with error " + error/ numPoints, gpu);
		pcv.setVisible(true);
		
	}
}


