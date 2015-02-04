package edu.stanford.rsl.conrad.cuda;

import edu.stanford.rsl.conrad.volume3d.Volume3D;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUdevprop;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public abstract class CUDAUtil {
	
	// Pre-determined kernel block size
	public static int gridBlockSize[] = {32, 16};
	
	/**
	 * Returns the given (address) value, adjusted to have
	 * the given alignment. In newer versions of JCuda, this
	 * function is also available as JCudaDriver#align
	 * 
	 * @param value The address value
	 * @param alignment The desired alignment
	 * @return The aligned address value
	 */
	public static int align(int value, int alignment)
	{
		return (((value) + (alignment) - 1) & ~((alignment) - 1));		
	}

	/**
	 * copies an int array to the device and returns a pointer to the memory.
	 * @param data the int array
	 * @return the pointer to the device memory
	 */
	public static CUdeviceptr copyToDeviceMemory(int [] data){
		int memorySize = data.length * Sizeof.INT;
		
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemcpy(deviceX, Pointer.to(data), memorySize, 
				cudaMemcpyKind.cudaMemcpyHostToDevice);
		return deviceX;
	}
	
	/**
	 * copies a float array to the device and returns a pointer to the memory.
	 * @param data the float array
	 * @return the pointer to the device memory
	 */
	public static CUdeviceptr copyToDeviceMemory(float [] data){
		int memorySize = data.length * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemcpy(deviceX, Pointer.to(data), memorySize, 
				cudaMemcpyKind.cudaMemcpyHostToDevice);
		return deviceX;
	}
	
	/**
	 * fetches a float data array from the device and frees the memory on the device.
	 * @param data the float array to write to
	 * @param deviceX the pointer to the device memory
	 */
	public static void fetchFromDeviceMemory(float [] data, CUdeviceptr deviceX){
		int memorySize = data.length * Sizeof.FLOAT;
		JCuda.cudaMemcpy(Pointer.to(data), deviceX, memorySize, 
				cudaMemcpyKind.cudaMemcpyDeviceToHost);
		JCuda.cudaFree(deviceX);
	}
	
	/**
	 * Allocates space on the CUDA device for a Volume3D 
	 * @param vol the volume
	 * @return the pointer to the memory
	 */
	public static CUdeviceptr allocateSpace(Volume3D vol){
		// We allocate too much memory as we parallelize along x and y direction and the memory must be a multiple along this direction internally.
		int adaptedWidth = iDivUp(vol.size[2], gridBlockSize[0]) * gridBlockSize[0];
		int adaptedHeight = iDivUp(vol.size[1], gridBlockSize[1]) * gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*vol.size[0]* vol.getInternalDimension() * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		return deviceX;
	}

	/**
	 * Moves the volume to the device.
	 * @param vol the volume
	 * @param deviceX the memory pointer
	 */
	public static void moveToDevice(Volume3D vol, CUdeviceptr deviceX){
		// Allocate memory on the device using JCuda
		int memorySize = vol.size[2]* vol.getInternalDimension() * Sizeof.FLOAT;
		// Copy memory from host to device using JCuda
		for (int i = 0; i < vol.size[0]; i++){
			for(int j = 0; j < vol.size[1]; j++){
				AdjustablePointer offset = new AdjustablePointer(deviceX, ((vol.size[1]*i) + j) * memorySize);
				JCuda.cudaMemcpy(offset, Pointer.to(vol.data[i][j]), memorySize, 
						cudaMemcpyKind.cudaMemcpyHostToDevice);
			}		
		}
	}

	/**
	 * Fetches the volume from the device
	 * @param vol the volume object
	 * @param deviceX the pointer to the memory on the deivce.
	 */
	public static void fetchFromDevice(Volume3D vol, CUdeviceptr deviceX){
		// Allocate memory on the device using JCuda
		int memorySize = vol.size[2]* vol.getInternalDimension() * Sizeof.FLOAT;
		// Copy memory from host to device using JCuda
		for (int i = 0; i < vol.size[0]; i++){
			for(int j = 0; j < vol.size[1]; j++){
				AdjustablePointer offset = new AdjustablePointer(deviceX, (((vol.size[1]*i) + j) * memorySize));
				JCuda.cudaMemcpy(Pointer.to(vol.data[i][j]), offset, memorySize, 
						cudaMemcpyKind.cudaMemcpyDeviceToHost);
			}		
		}
	}

	/**
	 * Integral division, rounding the result to the next highest integer.
	 * 
	 * @param a Dividend
	 * @param b Divisor
	 * @return a/b rounded to the next highest integer.
	 */
	public static int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	
	
	public static CUdeviceptr copyFloatArrayToDevice(float [] array, CUmodule module, String nameInCode) {
		CUdeviceptr devptr = new CUdeviceptr();
		JCudaDriver.cuModuleGetGlobal(devptr, new int[1], module, nameInCode);
		JCudaDriver.cuMemcpyHtoD(devptr, Pointer.to(array), Sizeof.FLOAT * array.length);
		return devptr;
	}
	
	public static void updateFloatArrayOnDevice(CUdeviceptr devptr, float [] array, CUmodule module) {
		//JCudaDriver.cuModuleGetGlobal(devptr, new int[1], module, nameInCode);
		JCudaDriver.cuMemcpyHtoD(devptr, Pointer.to(array), Sizeof.FLOAT * array.length);
	}
	
	public static long correctMemoryValue(int memory){
		long mem = memory;
		if (mem < 0) {
			mem -= Integer.MIN_VALUE;
			mem += Integer.MAX_VALUE;
		}
		return mem;
	}

	public static CUdevice getBestDevice() {
		CUdevice best = null;
		long lastmem = Long.MIN_VALUE;
		int [] count = new int[1];
		JCudaDriver.cuDeviceGetCount(count);
		for (int i = 0; i < count[0]; i++) {
			CUdevice dev = new CUdevice();
			JCudaDriver.cuDeviceGet(dev, i);
			CUdevprop prop = new CUdevprop();
			JCudaDriver.cuDeviceGetProperties(prop, dev);
			//System.out.println(prop);
			int [] memory = new int [1]; 
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			long mem = correctMemoryValue(memory[0]);
			//System.out.println("Memory " + mem);
			if (mem > lastmem){
				best = dev;
				lastmem = mem;
			}
		}
		return best;
	}
	
	public static CUdevice getSmallestDevice() {
		CUdevice best = null;
		long lastmem = Long.MAX_VALUE;
		int [] count = new int[1];
		JCudaDriver.cuDeviceGetCount(count);
		for (int i = 0; i < count[0]; i++) {
			CUdevice dev = new CUdevice();
			JCudaDriver.cuDeviceGet(dev, i);
			CUdevprop prop = new CUdevprop();
			JCudaDriver.cuDeviceGetProperties(prop, dev);
			//System.out.println(prop);
			int [] memory = new int [1]; 
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			long mem = correctMemoryValue(memory[0]);
			//System.out.println("Memory " + mem);
			if (mem < lastmem){
				best = dev;
				lastmem = mem;
			}
		}
		return best;
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
