/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.data.numeric.opencl.delegates;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;

public class OpenCLNumericMemoryDelegateLinear extends OpenCLMemoryDelegate{

	public OpenCLNumericMemoryDelegateLinear(float[] buffer, CLContext context, CLDevice device) {
		fBuffer = context.createFloatBuffer(((float[])buffer).length, Mem.READ_WRITE);
		linearHostMemory = buffer;
		fBuffer.getBuffer().put(buffer);
		fBuffer.getBuffer().rewind();
		hostChanged = true;
		deviceChanged = false;
		this.context = context;
		this.device = device;	
	}

	@Override
	protected void copyToLinearHostMemory() {
	}

	@Override
	protected void copyFromLinearHostMemory() {
	}

}
