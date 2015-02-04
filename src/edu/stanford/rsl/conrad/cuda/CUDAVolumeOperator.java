package edu.stanford.rsl.conrad.cuda;

import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.MAX_FILTERS;
import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.filt_get_filt_dirs;
import static edu.stanford.rsl.conrad.filtering.multiprojection.anisotropic.AnisotropicFilterFunction.filt_get_n_filters;
import ij.IJ;
import ij.ImagePlus;

import java.util.ArrayList;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUdevprop;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.dim3;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.volume3d.Volume3D;
import edu.stanford.rsl.conrad.volume3d.VolumeOperator;

/**
 * Class to implement all functions of VolumeOperator on CUDA hardware. Code is made CUDA compatible by replacing the VolumeOperator with CUDAVolumeOperator. As a result all code is executed on CUDA instead of the CPU. On good hardware usually a substantial speed-up is achieved.
 * Development and debugging should be done using Volume3D, as CUDA problems are usually hard to debug.
 * @author akmaier
 * @see edu.stanford.rsl.conrad.volume3d.VolumeOperator
 */
public class CUDAVolumeOperator extends VolumeOperator {
	/**
	 * The CUDA module containing the kernel
	 */
	private CUmodule module = null;
	private static boolean debug = false;
	private boolean inited = false;
	/**
	 * the context
	 */
	private CUcontext cuCtx = null;

	static long last = System.currentTimeMillis();

	private static void printTime(){
		long time = System.currentTimeMillis();
		long diff = time - last;
		System.out.println("Time passed: " + diff);
		last = time;
	}

	/**
	 * The grid size of the kernel execution
	 */
	private dim3 gridSize = null;

	private void callCUDAFunction(CUfunction function, ArrayList<Object> arguments){
		int offset = 0;
		if (debug) System.out.println("Working on Parameter set.");
		// send the parameters to the function
		for (int i = 0; i < arguments.size(); i++) {
			if (debug) System.out.print("Parameter " + i);
			Object argument = arguments.get(i);
			if (argument instanceof CUdeviceptr) {
				if (debug) System.out.println(" is CUdeviceptr");
				Pointer pointer = Pointer.to((Pointer) argument);
				offset = CUDAUtil.align(offset, Sizeof.POINTER);
				JCudaDriver.cuParamSetv(function, offset, pointer, Sizeof.POINTER);
				offset += Sizeof.POINTER;
			}
			if (argument  instanceof Integer) {
				if (debug) System.out.println(" is Integer");
				int [] integer = {((Integer) argument).intValue()};
				Pointer intPointer = Pointer.to(integer);
				offset = CUDAUtil.align(offset, Sizeof.INT);
				JCudaDriver.cuParamSetv(function, offset, intPointer, Sizeof.INT);
				offset += Sizeof.INT;
			}
			if (argument instanceof Float){
				if (debug) System.out.println(" is Float");
				float [] array = {((Float) argument).floatValue()};
				Pointer floatPointer = Pointer.to(array);
				offset = CUDAUtil.align(offset, Sizeof.FLOAT);
				JCudaDriver.cuParamSetv(function, offset, floatPointer, Sizeof.FLOAT);
				offset += Sizeof.FLOAT;
			}
		}
		// set parameter space
		JCudaDriver.cuParamSetSize(function, offset);
		if (debug) System.out.println("Parameters set.");
		// Call the CUDA kernel, writing the results into the volume which is pointed at
		if (debug) System.out.println("Setting blocks.");
		JCudaDriver.cuFuncSetBlockShape(function, CUDAUtil.gridBlockSize[0], CUDAUtil.gridBlockSize[1], 1);
		if (debug) System.out.println("Staring grid: " + gridSize.x +  "x"+ gridSize.y);
		int revan = JCudaDriver.cuLaunchGrid(function, gridSize.x, gridSize.y);
		if (debug) System.out.println("Exit code launch: "+revan);
		revan = JCudaDriver.cuCtxSynchronize();
		if (debug) System.out.println("Exit code sync: "+revan);
	}


	public void initCUDA(){
		if (!inited) {
			// Initialize the JCudaDriver. Note that this has to be done from 
			// the same thread that will later use the JCudaDriver API.
			JCudaDriver.setExceptionsEnabled(true);
			JCudaDriver.cuInit(0);
			CUdevice dev = CUDAUtil.getBestDevice();
			cuCtx = new CUcontext();
			JCudaDriver.cuCtxCreate(cuCtx, 0, dev);
			// check space on device:
			int [] memory = new int [1]; 
			JCudaDriver.cuDeviceTotalMem(memory, dev);
			int availableMemory = (int) (CUDAUtil.correctMemoryValue(memory[0]) / ((long)1024 * 1024));
			if (debug) {
				System.out.println("Total available Memory on CUDA card:" + availableMemory);
			}
			if (debug) {
				CUdevprop prop = new CUdevprop();
				JCudaDriver.cuDeviceGetProperties(prop, dev);
				System.out.println(prop.toFormattedString());
			}
			// Load the CUBIN file containing the kernel
			module = new CUmodule();
			JCudaDriver.cuModuleLoad(module, "CUDAVolumeFunctions.sm_10.cubin");

			// Obtain a function pointer to the kernel function. This function
			// will later be called. 
			// 
			if (debug) System.out.println("Initialized.");
			inited = true;
		}
	}
	
	public void cleanup(){
		//JCudaDriver.cuCtxDestroy(cuCtx);
		JCudaDriver.cuModuleUnload(module);
		
	}

	
	public void fill(Volume3D vol, float number){
		initCUDA();
		CUdeviceptr sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
		"_Z4fillPfPifi");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Float(number));
		arguments.add(new Integer(vol.getInternalDimension()));

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		if (debug) System.out.println("Calling.");
		callCUDAFunction(function, arguments);
		if (debug) System.out.println("Freeing.");

		JCuda.cudaFree(sizePointer);
		//((CUDAVolume3D) vol).fetch();
	}

	private dim3 getGrid(int [] size){
		return new dim3(
				CUDAUtil.iDivUp(size[2], CUDAUtil.gridBlockSize[0]), 
				CUDAUtil.iDivUp(size[1], CUDAUtil.gridBlockSize[1]), 
				1);
	}

	@Override
	public int divideByVolume(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;
		
		if (DEBUG_FLAG)
			fprintf("vol_div\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_div: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2) {
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2) {

			fprintf( "vol_div: Invalid dimension\n");
			return(0);

		}


		initCUDA();
		CUdeviceptr sizePointer = CUDAUtil.copyToDeviceMemory(vol1.size);
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
		"_Z6dividePfS_Pii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol1).getDevicePointer());
		arguments.add(((CUDAVolume3D) vol2).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol1.getInternalDimension()));

		// Calculate new grid size
		gridSize = getGrid(vol1.size);

		if (debug) System.out.println("Calling.");
		callCUDAFunction(function, arguments);
		if (debug) System.out.println("Freeing.");

		JCuda.cudaFree(sizePointer);

		return(0);

	}
	
	
	@Override
	public Volume3D solveMaximumEigenvalue(Volume3D [][] structureTensor)
	{
		CUDAVolume3D a11 = (CUDAVolume3D) structureTensor[0][0];
		CUDAVolume3D a12 = (CUDAVolume3D) structureTensor[0][1];
		CUDAVolume3D a13 = (CUDAVolume3D) structureTensor[0][2];
		CUDAVolume3D a22 = (CUDAVolume3D) structureTensor[1][1];
		CUDAVolume3D a23 = (CUDAVolume3D) structureTensor[1][2];
		CUDAVolume3D a33 = (CUDAVolume3D) structureTensor[2][2];
		CUDAVolume3D vol = (CUDAVolume3D) createVolume(a11.size, a11.spacing, a11.getInternalDimension());
		
		initCUDA();

		CUdeviceptr sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);
		
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z25filt_solve_max_eigenvaluePfS_S_S_S_S_PiiS_");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(a11.getDevicePointer());
		arguments.add(a12.getDevicePointer());
		arguments.add(a13.getDevicePointer());
		arguments.add(a22.getDevicePointer());
		arguments.add(a23.getDevicePointer());
		arguments.add(a33.getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(a11.getInternalDimension());
		arguments.add(vol.getDevicePointer());
		
		CUDAUtil.gridBlockSize[0] /= 2;
		
		gridSize = getGrid(vol.size);

		if (debug) System.out.println("Calling.");
		callCUDAFunction(function, arguments);
		if (debug) System.out.println("Freeing.");

		CUDAUtil.gridBlockSize[0] *= 2;
		
		JCuda.cudaFree(sizePointer);
		
		return(vol);
		
	}
	
	
	@Override
	public Volume3D createDirectionalWeights(int dimensions, int size[],
			float dim[], float dir[], int A, FILTER_TYPE t_filt)
	{
		Volume3D vol = null;
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs;
		int  dim_loop;


		if (DEBUG_FLAG)
			fprintf("filt_cos2\n"+t_filt);

		vol=createVolume(size, dim, 1);

		/* normalize filter direction */

		r_abs = 0.0f;
		r_abs += dir[0] * dir[0];
		r_abs += dir[1] * dir[1];
		r_abs += dir[2] * dir[2];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			dir[dim_loop] /= r_abs;

		if (DEBUG_FLAG) {
			fprintf("  direction = ");
			for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
				fprintf(dir[dim_loop]);
				if (dim_loop<dimensions-1)
					fprintf(", ");
			}
			fprintf("\n");
		}

		/* calculate filter boundings */

		getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		// load CUDA stuff.
		initCUDA();

		CUdeviceptr sizePointer = CUDAUtil.copyToDeviceMemory(size);
		CUdeviceptr dirPointer = CUDAUtil.copyToDeviceMemory(dir);
		CUdeviceptr fDeltaPointer = CUDAUtil.copyToDeviceMemory(f_delta);
		CUdeviceptr fMaxPointer = CUDAUtil.copyToDeviceMemory(f_max);

		Integer filt = 0;
		if (t_filt == FILTER_TYPE.QUADRATIC) filt = new Integer(1);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z9filt_cos2PfPiS_S_S_ii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(dirPointer);
		arguments.add(fDeltaPointer);
		arguments.add(fMaxPointer);
		arguments.add(new Integer(A));
		arguments.add(filt);

		gridSize = getGrid(vol.size);

		if (debug) System.out.println("Calling.");
		callCUDAFunction(function, arguments);
		if (debug) System.out.println("Freeing.");

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(dirPointer);
		JCuda.cudaFree(fDeltaPointer);
		JCuda.cudaFree(fMaxPointer);

		return(vol);
	}

	@Override
	public Volume3D createVolume(int size[],
			float dim[],
			int in_dim)
	{
		Volume3D volume;
		volume= new CUDAVolume3D(size, dim, in_dim);
		return(volume);
	}

	@Override
	public Volume3D createExponentialDirectionalHighPassFilter(int dimensions, int size[],
			float dim[], float dir[], int A, float B, float ri,
			FILTER_TYPE t_filt)
	{
		Volume3D vol;
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		float r_abs;
		int   dim_loop;

		if (DEBUG_FLAG)
			fprintf("filt_cos2_r\n"+t_filt);

		vol=createVolume(size, dim, 1);

		/* normalize filter direction */

		r_abs=0;
		for (dim_loop=0; dim_loop<dimensions; dim_loop++)
			r_abs += dir[dim_loop] * dir[dim_loop];
		r_abs = (float) Math.sqrt(r_abs);

		for (dim_loop=0; dim_loop<dimensions; dim_loop++) {
			dir[dim_loop] /= r_abs;
			if (DEBUG_FLAG)
				fprintf("  direction "+dim_loop+" = "+dir[dim_loop]+"\n");
		}

		/* calculate filter boudings */

		getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		initCUDA();	

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(size);
		Pointer dirPointer = CUDAUtil.copyToDeviceMemory(dir);
		Pointer fDeltaPointer = CUDAUtil.copyToDeviceMemory(f_delta);
		Pointer fMaxPointer = CUDAUtil.copyToDeviceMemory(f_max);

		Integer filt = 0;
		if (t_filt == FILTER_TYPE.QUADRATIC) filt = 1;

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z11filt_cos2_rPfPiS_S_S_iffi");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(dirPointer);
		arguments.add(fDeltaPointer);
		arguments.add(fMaxPointer);
		arguments.add(new Integer(A));
		arguments.add(new Float(B));
		arguments.add(new Float(ri));
		arguments.add(filt);

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(dirPointer);
		JCuda.cudaFree(fDeltaPointer);
		JCuda.cudaFree(fMaxPointer);

		//((CUDAVolume3D) vol).fetch();
		//vol.getImagePlus(dir[0] + " " +dir[1] + " " +dir[2] +" Cosine Square R").show();
		fftShift(vol);
		//vol.getImagePlus(dir[0] + " " +dir[1] + " " +dir[2] +"Cosine Square R shifted").show();

		return(vol);
	}

	@Override
	public int fftShift(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_fftshift\n");
		initCUDA();
		
		if (vol.in_dim == 1) makeComplex(vol);

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z8fftshiftPfPii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.getInternalDimension()));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);

		return(0);
	}

	@Override
	public Volume3D createHighPassFilter(int dimensions, int [] size, float [] dim, int filt_loop, float lp_upper){
		float [][] dir = new float[MAX_FILTERS][Volume3D.MAX_DIM];
		float hp_lower = (float) (10f*Math.PI);   /* was PI*10 LW 2006-01-31 */
		float hp_upper = (float) (10f*Math.PI);   /* was PI    LW 2006-01-31 */
		//float lp_upper = 1.50f;    /* was 1.5   LW 2006-01-31 */
		CUDAVolume3D vol= (CUDAVolume3D) createVolume(size, dim, 1);
		int n_filters;

		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);
		/* create HP filter */

		initCUDA();	

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(size);
		Pointer fDeltaPointer = CUDAUtil.copyToDeviceMemory(f_delta);
		Pointer fMaxPointer = CUDAUtil.copyToDeviceMemory(f_max);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z20createHighPassFilterPfPiS_S_fff");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(fDeltaPointer);
		arguments.add(fMaxPointer);
		arguments.add(new Float(lp_upper));
		arguments.add(new Float(hp_lower));
		arguments.add(new Float(hp_upper));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(fDeltaPointer);
		JCuda.cudaFree(fMaxPointer);

		//Volume3D.vol_fftshift(vol);

		filt_get_filt_dirs(vol.dimensions, dir);
		n_filters = filt_get_n_filters(vol.dimensions);

		IJ.showStatus("Computing High Pass Filters");
		IJ.showProgress((((float)(filt_loop))/n_filters));
		CUDAVolume3D filt = (CUDAVolume3D) createDirectionalWeights(vol.dimensions, vol.size, vol.spacing,
				dir[filt_loop] , 1, FILTER_TYPE.NORMAL);

		if (filt==null) {
			fprintf( "filt_make_enhance_filters: Out of memory\n");
			return(null);
		}

		multiply(filt,vol);
		vol.destroy();

		fftShift(filt);
		//filt.getImagePlus("filter"+ filt_loop).show();
		return filt;
	}

	@Override
	public int multiply(Volume3D vol1, Volume3D vol2)
	{

		int  dim_loop;
		if (debug) {
			System.out.print("Called multiply ");
			printTime();
		}
		if (DEBUG_FLAG)
			fprintf("vol_mult\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_mult: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol1.in_dim>2 || vol2.in_dim>2) {

			fprintf( "vol_mult: Invalid dimension\n");
			return(0);

		}

		if (debug) {
			System.out.print("prechecks ");
			printTime();
		}



		initCUDA();

		if (debug) {
			System.out.print("first init step ");
			printTime();
		}


		// Calculate new grid size
		gridSize = getGrid(vol1.size);

		if (debug) {
			System.out.print("get Grid");
			printTime();
		}


		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol1.size);

		if (debug) {
			System.out.print("mem cpy");
			printTime();
		}


		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
		"_Z8multiplyPfS_Pii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol1).getDevicePointer());
		arguments.add(((CUDAVolume3D) vol2).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol1.getInternalDimension()));

		if (debug) {
			System.out.print("init done ");
			printTime();
		}
		callCUDAFunction(function, arguments);

		if (debug) {
			System.out.print("CUDA done ");
			printTime();
		}
		JCuda.cudaFree(sizePointer);

		if (debug) {
			System.out.print("clean up done ");
			printTime();
		}

		return(0);
	}

	@Override
	public Volume3D createVolume(ImagePlus image, int mirror, int cuty, boolean uneven)
	{
		Volume3D volume;
		volume= new CUDAVolume3D(image, mirror, cuty, uneven);
		return(volume);
	}

	@Override
	public void makeComplex(Volume3D vol)
	{

		if (vol.getInternalDimension() == 2) return;
		if (vol.getInternalDimension() != 1) {
			fprintf("vol_make_comlex: Invalid dimension\n");
			return;
		}

		initCUDA();

		int adaptedWidth = CUDAUtil.iDivUp(vol.size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(vol.size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*vol.size[0]* 2 * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);
		JCuda.cudaMemset(deviceX, 0, memorySize);

		CUDAVolume3D cudaVol = (CUDAVolume3D) vol;

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z11makeComplexPfS_Pi");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(cudaVol.getDevicePointer());
		arguments.add(deviceX);
		arguments.add(sizePointer);

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(cudaVol.getDevicePointer());

		cudaVol.setDevicePointer(deviceX);

		float [][][] temp = new float [vol.size[0]][vol.size[1]][vol.size[2]*2];

		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 2;

		return;

	}

	@Override
	public Volume3D createLowPassFilter(int dimensions, int size[], float dim [], float lp_upper){
		float [] f_max = new float [Volume3D.MAX_DIM];
		float [] f_delta = new float [Volume3D.MAX_DIM];
		CUDAVolume3D vol = (CUDAVolume3D) createVolume(size, dim, 1);
		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		/* create LP filter */
		initCUDA();	

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(size);
		Pointer fDeltaPointer = CUDAUtil.copyToDeviceMemory(f_delta);
		Pointer fMaxPointer = CUDAUtil.copyToDeviceMemory(f_max);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z19createLowPassFilterPfPiS_S_f");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(fDeltaPointer);
		arguments.add(fMaxPointer);
		arguments.add(new Float(lp_upper));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(fDeltaPointer);
		JCuda.cudaFree(fMaxPointer);

		fftShift(vol);
		return vol;
	}

	@Override
	public Volume3D createGaussLowPassFilter(int dimensions, int size[],
			float dim[], float alpha)
	{
		Volume3D       vol;
		float []       f_max = new float [Volume3D.MAX_DIM];
		float []       f_delta = new float [Volume3D.MAX_DIM];

		if (DEBUG_FLAG)
			fprintf("filt_gauss\n");

		vol=createVolume(size, dim, 1);

		/* calculate filter boudings */

		VolumeOperator.getFrequencyBoundings(dimensions, size, dim, f_max, f_delta);

		/* create LP filter */
		initCUDA();	

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(size);
		Pointer fDeltaPointer = CUDAUtil.copyToDeviceMemory(f_delta);
		Pointer fMaxPointer = CUDAUtil.copyToDeviceMemory(f_max);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z10filt_gaussPfPiS_S_f");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(fDeltaPointer);
		arguments.add(fMaxPointer);
		arguments.add(new Float(alpha));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(fDeltaPointer);
		JCuda.cudaFree(fMaxPointer);

		fftShift(vol);

		return(vol);
	}

	@Override
	public int sigmoid(Volume3D vol,
			float smoothing, float lowValue, float highValue,
			float highPassLowerLevel,  float highPassUpperLevel)
	{

		if (DEBUG_FLAG)
			fprintf("vol_sigmoid_th\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}

		initCUDA();	

		// Calculate new grid size
		// Sigmoid is requires a lot of registers. Hence we have to reduce the block size a bit.
		CUDAUtil.gridBlockSize[0] /= 2;
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z7sigmoidPfPiifffff");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(new Float(smoothing));
		arguments.add(new Float(lowValue));
		arguments.add(new Float(highValue));
		arguments.add(new Float(highPassLowerLevel));
		arguments.add(new Float(highPassUpperLevel));

		callCUDAFunction(function, arguments);

		// restore original block size.
		CUDAUtil.gridBlockSize[0] *= 2;

		JCuda.cudaFree(sizePointer);

		return(0);
	}


	@Override
	public int addVolume(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_add\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_add: Volumes have different sizes\n");
				return(-1);

			}

		/* OBS !!! borde inte behova konvertera vol2 . komplex */

		if (vol1.in_dim==2 &&  vol2.in_dim==1) {
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2 || vol1.in_dim>2) {

			fprintf( "vol_add: Invalid dimension\n");
			return(-1);

		}

		initCUDA();	

		gridSize = getGrid(vol1.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol1.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z9addVolumePfS_Pii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol1).getDevicePointer());
		arguments.add(((CUDAVolume3D) vol2).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol1.in_dim));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);

		return(0);
	}


	@Override
	public int real(Volume3D vol)
	{

		if (debug) {
			System.out.print("Called real ");
			printTime();
		}
		if (DEBUG_FLAG)
			fprintf("vol_real\n");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "vol_real: Invalid dimension\n");
			return(-1);
		}

		int adaptedWidth = CUDAUtil.iDivUp(vol.size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(vol.size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*vol.size[0]* 1 * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);

		CUDAVolume3D cudaVol = (CUDAVolume3D) vol;

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z4realPfS_Pi");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(cudaVol.getDevicePointer());
		arguments.add(deviceX);
		arguments.add(sizePointer);
		if (debug) {
			System.out.print("Called init done ");
			printTime();
		}
		callCUDAFunction(function, arguments);
		if (debug) {
			System.out.print("CUDA done ");
			printTime();
		}

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(cudaVol.getDevicePointer());

		cudaVol.setDevicePointer(deviceX);

		float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 1;
		if (debug) {
			System.out.print("Clean up done ");
			printTime();
		}
		return(0);
	}

	@Override
	public int imag(Volume3D vol)
	{

		if (debug) {
			System.out.print("Called real ");
			printTime();
		}
		if (DEBUG_FLAG)
			fprintf("vol_real\n");

		if (vol.in_dim == 1) return(0);
		if (vol.in_dim != 2) {

			fprintf( "vol_real: Invalid dimension\n");
			return(-1);
		}

		int adaptedWidth = CUDAUtil.iDivUp(vol.size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(vol.size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*vol.size[0]* 1 * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);

		CUDAVolume3D cudaVol = (CUDAVolume3D) vol;

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z4imagPfS_Pi");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(cudaVol.getDevicePointer());
		arguments.add(deviceX);
		arguments.add(sizePointer);
		if (debug) {
			System.out.print("Called init done ");
			printTime();
		}
		callCUDAFunction(function, arguments);
		if (debug) {
			System.out.print("CUDA done ");
			printTime();
		}

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(cudaVol.getDevicePointer());

		cudaVol.setDevicePointer(deviceX);

		float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 1;
		if (debug) {
			System.out.print("Clean up done ");
			printTime();
		}
		return(0);
	}

	@Override
	public int addVolume(Volume3D vol1, Volume3D vol2, double weight)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_add\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_add: Volumes have different sizes\n");
				return(-1);

			}

		/* OBS !!! borde inte behova konvertera vol2 . komplex */

		if (vol1.in_dim==2 &&  vol2.in_dim==1){ 
			makeComplex(vol2);
			CONRAD.gc();
		}

		if (vol1.in_dim==1 &&  vol2.in_dim==2){ 
			makeComplex(vol1);
			CONRAD.gc();
		}

		if (vol2.in_dim>2 || vol1.in_dim>2) {

			fprintf( "vol_add: Invalid dimension\n");
			return(-1);

		}

		initCUDA();	

		gridSize = getGrid(vol1.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol1.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z9addVolumePfS_Piif");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol1).getDevicePointer());
		arguments.add(((CUDAVolume3D) vol2).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol1.in_dim));
		arguments.add(new Float(weight));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);


		return(0);
	}
	
	@Override
	public float min(Volume3D vol)
	{
		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max: Invalid dimension\n");
			return(0);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);
		float [] results = new float [vol.size[2] * vol.size[1]];
		CUdeviceptr resultPointer = CUDAUtil.copyToDeviceMemory(results);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z3minPfPiiS_");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(resultPointer);

		callCUDAFunction(function, arguments);

		CUDAUtil.fetchFromDeviceMemory(results, resultPointer);
		JCuda.cudaFree(sizePointer);

		float m = results[0];
		for (int i = 1; i < results.length; i++){
			if (results[i] < m) m = results[i];
		}
		
		return(m); 
	}

	@Override
	public int minOfTwoVolumes(Volume3D vol1, Volume3D vol2)
	{
		int  dim_loop;

		if (DEBUG_FLAG)
			fprintf("vol_get_min\n");

		for (dim_loop=0; dim_loop<vol1.in_dim; dim_loop++)
			if (vol1.size[dim_loop] != vol2.size[dim_loop]) {

				fprintf( "vol_get_min: Volumes have different sizes\n");
				return(-1);

			}

		if (vol1.in_dim!=1 ||  vol2.in_dim!=1) {

			fprintf( "vol_get_min: Invalid dimension\n");
			return(-1);

		}

		initCUDA();	

		gridSize = getGrid(vol1.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol1.size);
		
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z15minOfTwoVolumesPfS_Pii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol1).getDevicePointer());
		arguments.add(((CUDAVolume3D) vol2).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol1.in_dim));
		
		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);

		return(0);
	}
	
	@Override
	public float mean(Volume3D vol)
	{
		
		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max_pos: Invalid inner dimension\n");
			return(0);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);
		float [] results = new float [vol.size[2] * vol.size[1]];
		CUdeviceptr resultPointer = CUDAUtil.copyToDeviceMemory(results);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z4meanPfPiiS_");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(resultPointer);

		callCUDAFunction(function, arguments);

		CUDAUtil.fetchFromDeviceMemory(results, resultPointer);
		JCuda.cudaFree(sizePointer);

		float m = 0;
		for (int i = 0; i < results.length; i++){
			m += results[i];
		}
		return(m/results.length); 
	}
	
	
	@Override
	public float max(Volume3D vol)
	{
		

		/* defined for non-complex volumes only */

		if (vol.in_dim != 1) {
			fprintf("vol_max: Invalid dimension\n");
			return(0);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);
		float [] results = new float [vol.size[2] * vol.size[1]];
		CUdeviceptr resultPointer = CUDAUtil.copyToDeviceMemory(results);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z3maxPfPiiS_");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(resultPointer);

		callCUDAFunction(function, arguments);

		CUDAUtil.fetchFromDeviceMemory(results, resultPointer);
		JCuda.cudaFree(sizePointer);

		float m = results[0];
		for (int i = 1; i < results.length; i++){
			if (results[i] > m) m = results[i];
		}
		
		return(m); 
	}
	
	@Override
	public int addScalar(Volume3D vol,
			float realPart, 
			float imagPart )
	{

		if (DEBUG_FLAG)
			fprintf("vol_add_sc\n");


		if (imagPart!=0){
			makeComplex(vol);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z9addScalarPfPiiff");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(new Float(realPart));
		arguments.add(new Float(imagPart));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);
		
		return(0);

	}

	@Override
	public int abs(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_abs\n");

		if (vol.in_dim != 1 && vol.in_dim != 2) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}

		int adaptedWidth = CUDAUtil.iDivUp(vol.size[2], CUDAUtil.gridBlockSize[0]) * CUDAUtil.gridBlockSize[0];
		int adaptedHeight = CUDAUtil.iDivUp(vol.size[1], CUDAUtil.gridBlockSize[1]) * CUDAUtil.gridBlockSize[1];
		int memorySize = adaptedWidth*adaptedHeight*vol.size[0]* 1 * Sizeof.FLOAT;
		CUdeviceptr deviceX = new CUdeviceptr();
		JCuda.cudaMalloc(deviceX, memorySize);

		initCUDA();	
		
		CUDAVolume3D cudaVol = (CUDAVolume3D) vol;

		// Calculate new grid size
		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z3absPfS_Pii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(cudaVol.getDevicePointer());
		arguments.add(deviceX);
		arguments.add(sizePointer);
		arguments.add(cudaVol.in_dim);
		if (debug) {
			System.out.print("Called init done ");
			printTime();
		}
		callCUDAFunction(function, arguments);
		if (debug) {
			System.out.print("CUDA done ");
			printTime();
		}

		JCuda.cudaFree(sizePointer);
		JCuda.cudaFree(cudaVol.getDevicePointer());

		cudaVol.setDevicePointer(deviceX);

		float [][][] temp = new float[vol.size[0]][vol.size[1]][vol.size[2]];
		vol.data = null;
		vol.data = temp;
		temp = null;
		vol.in_dim = 1;
		if (debug) {
			System.out.print("Clean up done ");
			printTime();
		}



		return(0);
	}



	@Override
	public int multiplyScalar(Volume3D vol,
			float realPart, 
			float imagPart )
	{

		if (DEBUG_FLAG)
			fprintf("vol_mult_sc\n");

		if (vol.in_dim == 1) {

			if (imagPart!=0) {
				makeComplex(vol);
				CONRAD.gc();
			}

			initCUDA();	

			gridSize = getGrid(vol.size);

			Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

			CUfunction function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module,
					"_Z14multiplyScalarPfPiif");

			ArrayList<Object> arguments = new ArrayList<Object>();
			arguments.add(((CUDAVolume3D) vol).getDevicePointer());
			arguments.add(sizePointer);
			arguments.add(new Integer(vol.in_dim));
			arguments.add(new Float(realPart));

			callCUDAFunction(function, arguments);

			JCuda.cudaFree(sizePointer);

			// _Z14multiplyScalarPfPiif

		} else {   

			// _Z14multiplyScalarPfPiiff

			initCUDA();	

			gridSize = getGrid(vol.size);

			Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

			CUfunction function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, module,
					"_Z14multiplyScalarPfPiiff");

			ArrayList<Object> arguments = new ArrayList<Object>();
			arguments.add(((CUDAVolume3D) vol).getDevicePointer());
			arguments.add(sizePointer);
			arguments.add(new Integer(vol.in_dim));
			arguments.add(new Float(realPart));
			arguments.add(new Float(imagPart));

			callCUDAFunction(function, arguments);

			JCuda.cudaFree(sizePointer);

		}

		return(0);    
	}

	@Override
	public int sqrt(Volume3D vol)
	{

		if (DEBUG_FLAG)
			fprintf("vol_sqrt\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_sqrt: Invalid dimension\n");
			return(-1);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z4sqrtPfPii");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);

		return(0);
	}


	@Override
	public int upperLimit(Volume3D vol, float max)
	{

		if (DEBUG_FLAG)
			fprintf("vol_cut_upper\n");

		if (vol.in_dim != 1) {

			fprintf( "vol_abs: Invalid dimension\n");
			return(-1);
		}

		initCUDA();	

		gridSize = getGrid(vol.size);

		Pointer sizePointer = CUDAUtil.copyToDeviceMemory(vol.size);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module,
				"_Z10upperLimitPfPiif");

		ArrayList<Object> arguments = new ArrayList<Object>();
		arguments.add(((CUDAVolume3D) vol).getDevicePointer());
		arguments.add(sizePointer);
		arguments.add(new Integer(vol.in_dim));
		arguments.add(new Float(max));

		callCUDAFunction(function, arguments);

		JCuda.cudaFree(sizePointer);

		return(0);
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/