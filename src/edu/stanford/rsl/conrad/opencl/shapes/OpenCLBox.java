
/*
 * Copyright (C) 2014 - Zijia Guo, Andreas Maier 
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
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLBox extends Box implements OpenCLEvaluatable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3506083275052582212L;
	protected CLContext context;
	protected CLDevice device;
	protected CLBuffer<FloatBuffer> parameter;
	
	public OpenCLBox(CLDevice device){
		
	}
	
	public OpenCLBox(PointND lowerCorner, PointND upperCorner, CLDevice device){
		super(upperCorner.get(0),upperCorner.get(1),upperCorner.get(2));
		this.context = device.getContext();
		this.device = device;
		OpenCLUtil.initSimpleObjectEvaluator(context);     // ??
		handleParameter(lowerCorner, upperCorner);
	}
	

	public OpenCLBox(Box b, CLDevice device){
		this(b.lowerCorner, b.upperCorner, device);   
		this.transform = b.getTransform();
		
	}
	
	protected void handleParameter(PointND lowerCorner, PointND upperCorner) {
		// TODO Auto-generated method stub
		this.parameter = context.createFloatBuffer(6, Mem.READ_ONLY);  // "6"--> number of parameters
		
		this.parameter.getBuffer().put((float)lowerCorner.get(0));  // parameter[0]
		this.parameter.getBuffer().put((float)lowerCorner.get(1));  // parameter[1]
		this.parameter.getBuffer().put((float)lowerCorner.get(2));  // parameter[2]
		
		this.parameter.getBuffer().put((float)upperCorner.get(0));  // parameter[3]
		this.parameter.getBuffer().put((float)upperCorner.get(1));  // parameter[4]
		this.parameter.getBuffer().put((float)upperCorner.get(2));  // parameter[5]
		/*
		   4 corners on top:    (parameter[0], parameter[1], parameter[5])
		  					 	(parameter[3], parameter[1], parameter[5])
		  					 	(parameter[3], parameter[4], parameter[5])
		  					   	(parameter[0], parameter[4], parameter[5])
		   4 corners at bottom: (parameter[0], parameter[1], parameter[2])
		  					 	(parameter[3], parameter[1], parameter[2])
		  					 	(parameter[3], parameter[4], parameter[2])
		  					 	(parameter[0], parameter[4], parameter[2])
		*/
		
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
		
		CLKernel kernel = OpenCLUtil.simpleObjects.createCLKernel("evaluateBox");
		
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
		
		int u = 100;
		int v = 100;
		
		Box box = new Box(1,1,1);
		ArrayList<PointND> cpu = box.getPointCloud(u,v);
		
		int numPoints = u*v;
		
		OpenCLBox clbox = new OpenCLBox(box, device);
		
		CLBuffer<FloatBuffer> samplingPoints = OpenCLUtil.generateSamplingPoints(u, v, context, device);
		CLBuffer<FloatBuffer> outputBuffer = context.createFloatBuffer(3*numPoints, Mem.READ_WRITE);
		
		clbox.evaluate(samplingPoints, outputBuffer);
		
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

