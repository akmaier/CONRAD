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
import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLEllipsoid extends Ellipsoid implements OpenCLEvaluatable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8237752225512245291L;
	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> parameter;

	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public OpenCLEllipsoid(double dx, double dy, double dz, CLDevice device) {
		super(dx, dy, dz);
		// TODO Auto-generated constructor stub
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(dx, dy, dz);
	}

	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 * @param transform
	 */
	public OpenCLEllipsoid(double dx, double dy, double dz,
			AffineTransform transform, CLDevice device) {
		super(dx, dy, dz, transform);
		// TODO Auto-generated constructor stub
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(dx, dy, dz);
	}

	/**
	 * @param e
	 */
	public OpenCLEllipsoid(Ellipsoid e, CLDevice device) {
		super(e.dx, e.dy, e.dz, e.getTransform());
		// TODO Auto-generated constructor stub
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);
		handleParameter(dx, dy, dz);
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
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints,
			CLBuffer<FloatBuffer> outputBuffer) {
		// TODO Auto-generated method stub
		int elementCount = samplingPoints.getBuffer().capacity()/2;
		// assume equal length of elementCountU and elementCountV
		evaluate(samplingPoints, outputBuffer, (int) Math.sqrt(elementCount), (int) Math.sqrt(elementCount));
	}


	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints,
			CLBuffer<FloatBuffer> outputBuffer, int elementCountU,
			int elementCountV) {
		// TODO Auto-generated method stub
		int elementCount = samplingPoints.getBuffer().capacity()/2;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluateEllipsoid");
		kernel.putArgs(parameter, samplingPoints, outputBuffer).putArg(elementCountU).putArg(elementCountV);
		
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
	public boolean isClockwise() {
		// TODO Auto-generated method stub
		return false;
	}


	@Override
	public boolean isTimeVariant() {
		// TODO Auto-generated method stub
		return false;
	}
	
	public static void main(String [] args){
		CLContext context = OpenCLUtil.getStaticContext();
		CLDevice device = context.getMaxFlopsDevice();
		
		int u = 10;
		int v = 10;
		
		Ellipsoid ellipsoid = new Ellipsoid(1,2,2);
		ArrayList<PointND> cpu = ellipsoid.getPointCloud(u,v);
		
		int numPoints = u*v;
		
		OpenCLEllipsoid clellipsoid = new OpenCLEllipsoid(ellipsoid, device);
		
		CLBuffer<FloatBuffer> samplingPoints = OpenCLUtil.generateSamplingPoints(u, v, context, device);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(3*numPoints, Mem.READ_WRITE);
		
		clellipsoid.evaluate(samplingPoints, outputBuffer);
		
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


