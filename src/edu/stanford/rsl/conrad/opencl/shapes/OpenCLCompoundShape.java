/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.opencl.shapes;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.opencl.OpenCLEvaluatable;

public class OpenCLCompoundShape extends CompoundShape implements OpenCLEvaluatable{
	
	/**
	 */
	private static final long serialVersionUID = 1746217252255376910L;
	protected CLContext context;
	protected CLDevice device;

	public OpenCLCompoundShape(CompoundShape s, CLDevice device){
		super(s);
		this.context = device.getContext();
		this.device = device;
		
	}

	@Override
	public boolean isClockwise() {
		return false;
	}

	@Override
	public boolean isTimeVariant() {
		return false;
	}

	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer) {
		evaluate(samplingPoints,outputBuffer,0,0);		
	}

	@Override
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer,
			int elementCountU, int elementCountV) {
		int nT = this.size();
		outputBuffer.getBuffer().rewind();
		for(int i = 0; i < nT; i++){
			Triangle t = (Triangle) this.get(i);
			for(int j = 0; j < 3; j++){
				PointND p = new PointND(3);
				if(j == 0)
					p = t.getA();
				else if(j == 1)
					p = t.getB();
				else if(j == 2)
					p = t.getC();
				for(int k = 0; k < p.getDimension(); k++){
					outputBuffer.getBuffer().put((float) p.get(k));
				}
			}
		}
		outputBuffer.getBuffer().rewind();
		
		CLCommandQueue clc = device.createCommandQueue();
		clc.putWriteBuffer(outputBuffer, true).finish();
		clc.release();
		
	}
	
}
