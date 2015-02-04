package edu.stanford.rsl.conrad.cuda.splines;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;

/**
 * Wrapper class to create a BSpline object in the CUDAMemory
 * @author akmaier
 *
 */
public class CUDABSpline extends BSpline {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4887896401270761029L;
	private CUdeviceptr deviceX;
	
	public CUDABSpline(BSpline spline) {
		super(spline.getControlPoints(), spline.getKnots());
		float [] binary = spline.getBinaryRepresentation();
		int memorySize = binary.length;
		JCuda.cudaMemcpy(deviceX, Pointer.to(binary), memorySize, 
				cudaMemcpyKind.cudaMemcpyHostToDevice);
		deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemset(deviceX, 0, memorySize);
	}

	/**
	 * releases the memory on the device for this volume.
	 */
	public void destroy(){
		JCuda.cudaFree(deviceX);
	}
	
	/**
	 * @return the deviceX
	 */
	public CUdeviceptr getDeviceX() {
		return deviceX;
	}

	/**
	 * @param deviceX the deviceX to set
	 */
	public void setDeviceX(CUdeviceptr deviceX) {
		this.deviceX = deviceX;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
