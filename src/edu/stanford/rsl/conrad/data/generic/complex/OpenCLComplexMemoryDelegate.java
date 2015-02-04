/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.complex;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;



public class OpenCLComplexMemoryDelegate extends OpenCLMemoryDelegate {

	public OpenCLComplexMemoryDelegate(Object buffer, CLContext context,
			CLDevice device) {
		if(buffer instanceof float[]) {
			fBuffer = context.createFloatBuffer(((float[])buffer).length, Mem.READ_WRITE);
			linearHostMemory = (float[])buffer;
			fBuffer.getBuffer().put((float[])buffer);
			fBuffer.getBuffer().rewind();
			hostChanged = true;
			deviceChanged = false;
			this.context = context;
			this.device = device;	
		}
	}
	
	@Override
	protected void copyToLinearHostMemory() {
	}

	@Override
	protected void copyFromLinearHostMemory() {
	}	
}
