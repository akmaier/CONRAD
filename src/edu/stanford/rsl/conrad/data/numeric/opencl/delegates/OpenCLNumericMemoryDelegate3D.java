/*
 * Copyright (C) 2014 - Anja Pohan and Stefan Nottrott 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.opencl.delegates;

import java.util.ArrayList;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;

import edu.stanford.rsl.conrad.data.OpenCLMemoryDelegate;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class OpenCLNumericMemoryDelegate3D extends OpenCLMemoryDelegate {
	
	protected ArrayList<Grid2D> hostMemory;

	public OpenCLNumericMemoryDelegate3D(ArrayList<Grid2D> buffer, CLContext context,
			CLDevice device) {
			hostMemory = buffer;
			linearHostMemory = new float[hostMemory.size()*hostMemory.get(0).getHeight()*hostMemory.get(0).getWidth()];
			copyToLinearHostMemory();
			fBuffer = context.createFloatBuffer(linearHostMemory.length, Mem.READ_WRITE);
			fBuffer.getBuffer().put(linearHostMemory);
			fBuffer.getBuffer().rewind();
			hostChanged = true;
			deviceChanged = false;
			this.context = context;
			this.device = device;
	}
	
	public OpenCLNumericMemoryDelegate3D(OpenCLNumericMemoryDelegate3D delegate) {
			hostMemory = delegate.hostMemory;
			linearHostMemory = delegate.linearHostMemory;
			this.context = delegate.context;
			this.device = delegate.device;
			fBuffer = context.createFloatBuffer(linearHostMemory.length, Mem.READ_ONLY);
			device.createCommandQueue().putCopyBuffer(delegate.fBuffer, fBuffer).finish().release();
			fBuffer.getBuffer().rewind();
			hostChanged = false;
			deviceChanged = false;

	}
	
	/**
	 * Copies the content of the ArrayList memory to the linear memory
	 */
	protected void copyToLinearHostMemory(){
		int i = 0;
		for(Grid2D grid : hostMemory){
			for(int j = 0; j < grid.getBuffer().length; j++){
				linearHostMemory[j+i] = grid.getBuffer()[j];
			}
			i += grid.getBuffer().length;
		}
	}
	
	/**
	 * Copies the content of the linear memory to the ArrayList memory
	 */
	protected void copyFromLinearHostMemory(){
		int sliceHeight = hostMemory.get(0).getHeight();
		int sliceWidth = hostMemory.get(0).getWidth();
		for(int i = 0; i < linearHostMemory.length/(sliceHeight*sliceWidth); i++){
			System.arraycopy(linearHostMemory, i*sliceHeight*sliceWidth, hostMemory.get(i).getBuffer(), 0, sliceHeight*sliceWidth);
		}
	}
}
