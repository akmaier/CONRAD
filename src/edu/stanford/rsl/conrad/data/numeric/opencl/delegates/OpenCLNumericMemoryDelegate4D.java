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
import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class OpenCLNumericMemoryDelegate4D extends OpenCLMemoryDelegate {

	ArrayList<Grid3D> hostMemory;

	public OpenCLNumericMemoryDelegate4D(ArrayList<Grid3D> buffer, CLContext context,
			CLDevice device) {
			hostMemory = buffer;
			linearHostMemory = new float[hostMemory.size()*hostMemory.get(0).getSize()[0]*hostMemory.get(0).getSize()[1]*hostMemory.get(0).getSize()[2]];
			copyToLinearHostMemory();
			fBuffer = context.createFloatBuffer(linearHostMemory.length, Mem.READ_WRITE);
			fBuffer.getBuffer().put(linearHostMemory);
			fBuffer.getBuffer().rewind();
			hostChanged = true;
			deviceChanged = false;
			this.context = context;
			this.device = device;
	}

	/**
	 * Copies the content of the ArrayList memory to the linear memory
	 */
	protected void copyToLinearHostMemory(){
		int i = 0;
		for(Grid3D grid : hostMemory){
			for (Grid2D slice : grid.getBuffer()){
				for(int j = 0; j < slice.getBuffer().length; j++){
					linearHostMemory[j+i] = slice.getBuffer()[j];
				}
				i += slice.getBuffer().length;
			}
		}
	}

	/**
	 * Copies the content of the linear memory to the ArrayList memory
	 */
	protected void copyFromLinearHostMemory(){
		int slicePitch = hostMemory.get(0).getSize()[1]*hostMemory.get(0).getSize()[0];
		int volumePitch = slicePitch*hostMemory.get(0).getSize()[2];
		for(int j = 0; j < hostMemory.get(0).getSize()[3]; j++){
			for(int i = 0; i < hostMemory.get(0).getSize()[2]; i++){
				System.arraycopy(linearHostMemory, i*slicePitch + j*volumePitch, hostMemory.get(j).getBuffer().get(i).getBuffer(), 0, slicePitch);
			}
		}
	}
}
