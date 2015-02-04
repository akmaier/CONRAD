package edu.stanford.rsl.conrad.opencl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;

public interface OpenCLEvaluatable {
	
	public boolean isClockwise();
	public boolean isTimeVariant();
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer);
	public void evaluate(CLBuffer<FloatBuffer> samplingPoints, CLBuffer<FloatBuffer> outputBuffer, int elementCountU, int elementCountV);

}
/*
 * Copyright (C) 2010-2014 Peter Fischer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/