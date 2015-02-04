package edu.stanford.rsl.conrad.cuda;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.volume3d.FFTVolumeHandle;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import jcuda.runtime.JCuda;

public class CUDAFFTVolumeHandle extends FFTVolumeHandle {

	private boolean nativeCopy = false;

	public CUDAFFTVolumeHandle(VolumeOperator operator){
		super(operator);
		JCufft.setExceptionsEnabled(true);
		JCufft.initialize();
	}

	public enum CUFFTResult {
		CUFFT_SUCCESS,
		CUFFT_INVALID_PLAN,
		CUFFT_ALLOC_FAILED,
		CUFFT_INVALID_TYPE,
		CUFFT_INVALID_VALUE,
		CUFFT_INTERNAL_ERROR,
		CUFFT_EXEC_FAILED,
		CUFFT_SETUP_FAILED,
		CUFFT_INVALID_SIZE
	};

	public static CUFFTResult getResultEnum(int i){
		CUFFTResult res = null;
		if (i==0) res = CUFFTResult.CUFFT_SUCCESS;
		if (i==1) res = CUFFTResult.CUFFT_INVALID_PLAN;
		if (i==2) res = CUFFTResult.CUFFT_ALLOC_FAILED;
		if (i==3) res = CUFFTResult.CUFFT_INVALID_TYPE;
		if (i==4) res = CUFFTResult.CUFFT_INVALID_VALUE;
		if (i==5) res = CUFFTResult.CUFFT_INTERNAL_ERROR;
		if (i==6) res = CUFFTResult.CUFFT_EXEC_FAILED;
		if (i==7) res = CUFFTResult.CUFFT_SETUP_FAILED;
		if (i==8) res = CUFFTResult.CUFFT_INVALID_SIZE;
		return res;
	}

	private static void checkResult(int i) throws Exception {
		if (i != 0) {
			throw new Exception ("CUDA FFT Error: " + getResultEnum(i));
		}
	}

	public static float[][][] toHostFormat(float [] cuda, int [] size){
		float [] [] [] hostVolume = new float[size[0]][size[1]][size[2]*2];
		int sliceStride = size[2] * size[1]*2;
		int rowStride = size[2]*2;
		for (int h = 0; h < size[0]; h++){
			for (int j = 0; j < size[1]; j++){
				System.arraycopy(cuda, (sliceStride * h) + (j * rowStride), hostVolume[h][j], 0, size[2]*2);
				//for (int i = 0; i < size[2]; i++){
				//	float value = cuda[(sliceStride * h) + (j * rowStride) + (i*2)];
				//	hostVolume[h][j][i*2] =  value;
				//	value = cuda[(sliceStride * h) + (j * rowStride) + (i*2)+1];;
				//	hostVolume[h][j][i*2+1] =  value;
				//}
			}
		}
		return hostVolume;
	}

	public static float[] toCUDAFormat(float [][][] hostVolume){
		int [] size = {hostVolume.length, hostVolume[0].length, hostVolume[0][0].length/2};
		int sliceStride = size[2] * size[1]*2;
		int rowStride = size[2]*2;
		float [] cuda = new float[size[0]*size[1]*size[2]*2];
		for (int h = 0; h < size[0]; h++){
			for (int j = 0; j < size[1]; j++){
				System.arraycopy(hostVolume[h][j], 0, cuda, (sliceStride * h) + (j * rowStride), size[2]*2);
				//for (int i = 0; i < size[2]; i++){
				//	float value = hostVolume[h][j][2*i];
				//	cuda[(sliceStride * h) + (j * rowStride) + (i*2)] =  value;
				//	value = hostVolume[h][j][2*i+1];
				//	cuda[(sliceStride * h) + (j * rowStride) + (i*2)+1] =  value;
				//}
			}
		}
		return cuda;
	}

	/**
	 * Performs a forward 3-D FFT on the given volume in the CUDA memory.
	 * @param deviceX the Pointer to the device's memory
	 * @param size the sizes of the volume
	 * @throws Exception may happen.
	 */
	public void forwardTransform(Pointer deviceX, int [] size) throws Exception {
		if (debug){
			
			
			System.out.println("Planning " + size[0] +"x"+size[1]+"x"+size[2] +" Complex FFT");
			
		}
		
		cufftHandle plan = new cufftHandle();
		
		int revan = JCufft.cufftPlan3d(plan, size[0], size[1], size[2], cufftType.CUFFT_C2C);
		checkResult(revan);
		revan = JCufft.cufftExecC2C(plan, deviceX, deviceX, JCufft.CUFFT_FORWARD);
		checkResult(revan);
		revan = JCufft.cufftDestroy(plan);
		checkResult(revan);
	}

	@Override
	public void forwardTransform(Volume3D vol)
	{
		try{
			if (debug)
				System.out.println("CUDA vol_fft\n");

			operator.makeComplex(vol);  

			if (vol instanceof CUDAVolume3D){
				int [] fftsize = {vol.size[0], vol.size[1], vol.size[2]};
				
				forwardTransform(((CUDAVolume3D) vol).getDevicePointer(), fftsize);
			} else {
				CONRAD.gc();


				if(nativeCopy) {
					CUdeviceptr deviceX = CUDAUtil.allocateSpace(vol);
					CUDAUtil.moveToDevice(vol, deviceX);
					forwardTransform(deviceX, vol.size);
					CUDAUtil.fetchFromDevice(vol, deviceX);
					JCuda.cudaFree(deviceX);
				} else {
					float [] cuda = toCUDAFormat(vol.data);
					cufftHandle plan = new cufftHandle();
					
					int revan = JCufft.cufftPlan3d(plan, vol.size[0], vol.size[1], vol.size[2], cufftType.CUFFT_C2C);
					checkResult(revan);
					revan = JCufft.cufftExecC2C(plan, cuda, cuda, JCufft.CUFFT_FORWARD);
					checkResult(revan);
					//Clean up
					revan = JCufft.cufftDestroy(plan);
					checkResult(revan);
					vol.data = null;
					vol.data = toHostFormat(cuda, vol.size);
					cuda = null;
					CONRAD.gc();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}


	}

	/**
	 * Performs an inverse 3-D FFT on the CUDA device memory pointed to by deviceX.
	 * @param deviceX the device pointer
	 * @param size the sizes of the volume.
	 * @throws Exception may happen.
	 */
	public void inverseTransform(Pointer deviceX, int [] size) throws Exception{
		cufftHandle plan = new cufftHandle();
		int revan = JCufft.cufftPlan3d(plan, size[0], size[1], size[2], cufftType.CUFFT_C2C);
		checkResult(revan);
		revan = JCufft.cufftExecC2C(plan, deviceX, deviceX, JCufft.CUFFT_INVERSE);
		checkResult(revan);
		revan = JCufft.cufftDestroy(plan);
		checkResult(revan);
	}

	@Override
	public void inverseTransform(Volume3D vol)
	{

		try{
			if (debug)
				System.out.println("CUDA vol_ifft\n");

			operator.makeComplex(vol);

			if (vol instanceof CUDAVolume3D){
				int [] fftsize = {vol.size[0], vol.size[1], vol.size[2]};
				inverseTransform(((CUDAVolume3D) vol).getDevicePointer(), fftsize);
			} else {
				CONRAD.gc();

				if(nativeCopy) {
					CUdeviceptr deviceX = CUDAUtil.allocateSpace(vol);
					CUDAUtil.moveToDevice(vol, deviceX);
					inverseTransform(deviceX, vol.size);
					CUDAUtil.fetchFromDevice(vol, deviceX);
					JCuda.cudaFree(deviceX);
				} else {
					float [] cuda = toCUDAFormat(vol.data);
					cufftHandle plan = new cufftHandle();
					int revan = JCufft.cufftPlan3d(plan, vol.size[0], vol.size[1], vol.size[2], cufftType.CUFFT_C2C);
					checkResult(revan);
					revan = JCufft.cufftExecC2C(plan, cuda, cuda, JCufft.CUFFT_INVERSE);
					checkResult(revan);
					//Clean up
					revan = JCufft.cufftDestroy(plan);
					checkResult(revan);
					vol.data = null;
					vol.data = toHostFormat(cuda, vol.size);
					cuda = null;
					CONRAD.gc();
				}
			}
			operator.multiplyScalar(vol, 1.0f / (float) (vol.size[0]*vol.size[1]*vol.size[2]), 0.0f);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void cleanUp() {
		if (nativeCopy)	JCuda.cudaThreadExit();
	}

	@Override
	public void setThreadNumber(int number) {
		// not gonna happen
	}


}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
