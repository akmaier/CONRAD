package edu.stanford.rsl.conrad.cuda;

import ij.ImagePlus;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;


/**
 * CUDAVolume3D models a Volume3D in the CUDA memory. All functions of Volume3D are also available for this class, but directly implemented in CUDA. If an algorithm is implemented and tested using Volume3D, it can easily ported to CUDA by setting the VolumeOperator to CUDAVolumeOperator. The AnisotropicFilterFunction is a nice example on how to use this interface.
 * @author akmaier
 *
 * @see CUDAVolumeOperator
 * @see edu.stanford.rsl.conrad.volume3d.Volume3D
 * @see edu.stanford.rsl.conrad.volume3d.VolumeOperator
 */
public class CUDAVolume3D extends Volume3D {


	private CUdeviceptr deviceX;
	
	
	public CUDAVolume3D(int[] size2, float[] dim2, int inDim) {
		super(size2, dim2, inDim);
		int adaptedWidth = CUDAUtil.iDivUp(size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*size[0]* getInternalDimension() * Sizeof.FLOAT;
		deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemset(deviceX, 0, memorySize);
	}
	
	public CUDAVolume3D(ImagePlus image, int mirror, int cuty, boolean uneven) {
		super(image, mirror, cuty, uneven);
		int adaptedWidth = CUDAUtil.iDivUp(size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*size[0]* getInternalDimension() * Sizeof.FLOAT;
		deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemset(deviceX, 0, memorySize);
		updateOnDevice();
	}

	/**
	 * Fetches the data from the CUDA memory to the Java memory. 
	 */
	public void fetch(){
		CUDAUtil.fetchFromDevice(this, deviceX);
	}
	
	/**
	 * Moves the data from the Java memory to the CUDA memory
	 */
	public void updateOnDevice(){
		CUDAUtil.moveToDevice(this, deviceX);
	}
	
	/**
	 * releases the memory on the device for this volume.
	 */
	public void destroy(){
		super.destroy();
		JCuda.cudaFree(deviceX);
	}
	
	public CUdeviceptr getDevicePointer(){
		return deviceX;
	}
	
	public void setDevicePointer(CUdeviceptr deviceX){
		this.deviceX = deviceX;
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
