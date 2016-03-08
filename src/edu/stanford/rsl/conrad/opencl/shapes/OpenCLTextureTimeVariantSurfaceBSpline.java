package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;
import java.util.ArrayList;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage3d;
import com.jogamp.opencl.CLImageFormat;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;

public class OpenCLTextureTimeVariantSurfaceBSpline extends
		OpenCLTimeVariantSurfaceBSpline {

	CLImage3d<FloatBuffer> texture;
	
	public OpenCLTextureTimeVariantSurfaceBSpline(
			TimeVariantSurfaceBSpline timeVariantSpline, CLDevice device) {
		super(timeVariantSpline, device);
	}
	
	public OpenCLTextureTimeVariantSurfaceBSpline(
			ArrayList<SurfaceBSpline> splines, CLDevice device) {
		super(splines, device);
	}

	protected void handleControlPoints(){
		int size = this.timeVariantShapes.size() * this.timeVariantShapes.get(0).getControlPoints().size();
		//int count = 0;
		this.controlPoints = context.createFloatBuffer(size*4, Mem.READ_ONLY);
		for (int j = 0; j < timeVariantShapes.size(); j++){
			ArrayList<PointND> controlPoints = timeVariantShapes.get(j).getControlPoints();
			for(int i=0;i<controlPoints.size();i++) {
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(0));
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(1));
				this.controlPoints.getBuffer().put((float)controlPoints.get(i).get(2));
				this.controlPoints.getBuffer().put(0f);
				//count ++;
			}
		}
		//System.out.println(size + " " + count);
		this.controlPoints.getBuffer().rewind();	
		
		texture = context.createImage3d(this.controlPoints.getBuffer(), this.timeVariantShapes.get(0).getVKnots().getLen()-4, this.timeVariantShapes.get(0).getUKnots().getLen()-4, this.timeVariantShapes.size(), new CLImageFormat(CLImageFormat.ChannelOrder.RGBA, CLImageFormat.ChannelType.FLOAT), CLMemory.Mem.READ_ONLY);
		device.createCommandQueue().putWriteImage(texture, true).finish().release();
	}
	
	/*
	 * Tessellate a 3D spline using texture interpolation.
	 * Used to be called evaluteNoTransfer, but the old evaluate function caused a memory leak
	 * @see edu.stanford.rsl.conrad.opencl.OpenCLTimeVariantSurfaceBSpline#evaluate(com.jogamp.opencl.CLBuffer, com.jogamp.opencl.CLBuffer)
	 */
	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV) {
		int elementCount = samplingPoints.getBuffer().capacity()/3;                                  // Length of arrays to process
		int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
		int globalWorkSize = OpenCLUtil.roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
		
		CLKernel kernel = OpenCLUtil.getProgramInstance().createCLKernel("evaluate3DTextureInterp");
		
		
		kernel.putArgs(texture, samplingPoints, outputBuffer)
		.putArg(elementCountU).putArg(elementCountV).putArg(timeVariantShapes.size()+4)
		.putArg(elementCount);

		// asynchronous write of data to GPU device,
		// followed by blocking read to get the computed results back.

		CLCommandQueue clc = device.createCommandQueue();
		clc.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();

		kernel.release();
		clc.release();
	}
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1682699329382911891L;

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/