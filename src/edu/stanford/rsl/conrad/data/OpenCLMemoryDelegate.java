package edu.stanford.rsl.conrad.data;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;

public abstract class OpenCLMemoryDelegate implements AutoCloseable{
	
	protected CLBuffer<FloatBuffer> fBuffer;
	protected float [] linearHostMemory;
	protected CLContext context;
	protected CLDevice device;
	protected boolean hostChanged;
	protected boolean deviceChanged;
	
	boolean debug = false;
	
	
	/**
	 * Returns the pointer the the CLBuffer object.
	 * @return the CLBuffer
	 */
	public CLBuffer<FloatBuffer> getCLBuffer(){
		return fBuffer;
	}
	
	public float[] getBuffer(){
		return linearHostMemory;
	}

	/**
	 * Call this method before you want to run a host operation on the grid.
	 */
	public void prepareForHostOperation() {
		if (deviceChanged && hostChanged) throw new RuntimeException("Memory is in inconsistent state. Host and Device have changed independently.");
		if (deviceChanged){
			pull();
			deviceChanged = false;
		}
	}
	
	/**
	 * Call this method before you want to run a device operation.
	 */
	public void prepareForDeviceOperation() {
		if (deviceChanged && hostChanged) throw new RuntimeException("Memory is in inconsistent state. Host and Device have changed independently.");
		if (hostChanged){
			push();
			hostChanged = false;
		}
	}

	/**
	 * This method sends the current data from the host memory to the device memory.
	 */
	public void push() {
		if (debug) System.out.println("writing to device");
		copyToLinearHostMemory();
		fBuffer.getBuffer().put(linearHostMemory);
		fBuffer.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue();
		queue.putWriteBuffer(fBuffer, true).finish();
		queue.release();
		fBuffer.getBuffer().rewind();
	}

	/**
	 * This method fetches the current data from the device memory to the host memory.
	 */
	public void pull() {
		if (debug) System.out.println("writing to host");
		fBuffer.getBuffer().rewind();
		CLCommandQueue queue = device.createCommandQueue();
		queue.putReadBuffer(fBuffer, true).finish();
		queue.release();
		fBuffer.getBuffer().get(linearHostMemory);
		fBuffer.getBuffer().rewind();
		copyFromLinearHostMemory();
	}

	/**
	 * Call this method after you performed changes in the device memory, e.g. by calling a kernel function.
	 */
	public void notifyDeviceChange(){
		deviceChanged = true;
	}
	
	public void notifyHostChange(){
		hostChanged = true;
	}
	
	/**
	 * release the memory from the device
	 */
	public void release() {
		if (fBuffer!=null && !fBuffer.isReleased())
			fBuffer.release();
	}

	public CLDevice getCLDevice() {
		return device;
	}

	public CLContext getCLContext() {
		return context;
	}
	
	@Override
	public void close(){
		if(debug == true)
			System.out.println("Was at the autocloseable close() function!");
		this.release();
	}
	
	protected abstract void copyToLinearHostMemory();
	
	protected abstract void copyFromLinearHostMemory();

}
