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
import edu.stanford.rsl.conrad.geometry.shapes.simple.Cone;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLCone extends Cone implements OpenCLEvaluatable {

	private static final long serialVersionUID = -2146326261229488589L;

	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> parameter;
	
	public OpenCLCone(CLDevice device) {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public OpenCLCone(double dx, double dy, double dz, CLDevice device) {
		super(dx, dy, dz);
		// TODO Auto-generated constructor stub
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);     // ??
		handleParameter(dx, dy, dz);
	}

	/**
	 * @param c
	 */
	public OpenCLCone(Cone c, CLDevice device) {
		//TODO: The CPU cone still has a different radius then the GPU cone! We need to clarify which one is correct
		this(SimpleOperators.multiplyElementWise(
				SimpleOperators.subtract(c.getMax().getAbstractVector(),c.getMin().getAbstractVector()),
				new SimpleVector(0.5,0.5,1))
				, device);
		this.transform = c.getTransform();
	}
	
	/**
	 * @param c
	 */
	public OpenCLCone(SimpleVector paras, CLDevice device) {
		this(paras.getElement(0), paras.getElement(1), paras.getElement(2), device);
	}


	protected void handleParameter(double dx, double dy, double dz){
		
		
		double a = (dx)/(dz);
		double b = (dy)/(dz);
		this.parameter = context.createFloatBuffer(5, Mem.READ_ONLY);
		/*
		this.parameter = context.createFloatBuffer(5, Mem.READ_ONLY);
		
		this.parameter.getBuffer().put(-(float)min.get(0));  //parameter[0] --> dx
		this.parameter.getBuffer().put(-(float)min.get(1));  //parameter[1] --> dy
		this.parameter.getBuffer().put(-(float)min.get(2));  //parameter[2] --> dz
		*/
		this.parameter.getBuffer().put((float)dx);
		this.parameter.getBuffer().put((float)dy);
		this.parameter.getBuffer().put((float)dz);
		this.parameter.getBuffer().put((float)a);  //parameter[3]
		this.parameter.getBuffer().put((float)b);  //parameter[4]
		
		this.parameter.getBuffer().rewind();                
		device.createCommandQueue().putWriteBuffer(this.parameter, true);
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
		evaluate(samplingPoints, outputBuffer, (int)Math.sqrt(elementCount), (int)Math.sqrt(elementCount));
	}

	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints,
			CLBuffer<FloatBuffer> outputBuffer, int elementCountU,
			int elementCountV) {
		// TODO Auto-generated method stub
		int elementCount = samplingPoints.getBuffer().capacity()/2;
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);  // to guaranty that global size is interger multiple of group size (local size) 
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluateCone");
		
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
		
		int u = 3;
		int v = 10;
		
		Cone cone = new Cone(12,12,12);
		ArrayList<PointND> cpu = cone.getPointCloud(u,v);
		
		int numPoints = u*v;
		
		OpenCLCone clcone = new OpenCLCone(cone, device);
		
		CLBuffer<FloatBuffer> samplingPoints = OpenCLUtil.generateSamplingPoints(u, v, context, device);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(3*numPoints, Mem.READ_WRITE);
		
		clcone.evaluate(samplingPoints, outputBuffer, u, v);
		
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

