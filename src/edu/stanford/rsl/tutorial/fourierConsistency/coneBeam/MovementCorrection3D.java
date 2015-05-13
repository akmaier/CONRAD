package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import java.io.IOException;
import java.nio.FloatBuffer;

import edu.stanford.rsl.conrad.data.generic.GenericPointwiseOperators;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexPointwiseOperators;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

public class MovementCorrection3D {

	/**
	 * @param Grid3D data: contains original projection data
	 * @param Config conf: provides parameter of acquisition and precomputed arrays used to shift
	 * @param boolean naive: if true: naive cpu version, if false faster gpu version
	 */
	private Config m_conf;
	private ComplexGrid3D m_data;
	// length of all data
	private int m_datasize;
	// stores transposed data (projection dimension becomes first dimension) after 2D-Fouriertransform
	private ComplexGrid3D m_2dFourierTransposed = null;
	// stores data after fourier transformation on projection dimension
	private ComplexGrid3D m_3dFourier = null;
	
	// mask is a precomputed binary mask containing ones where the information after fourier on projection should be zero
	private Grid2D m_mask;
	
	// opencl members
	private CLContext context;
	private CLDevice device;
	private CLProgram program;
	private CLKernel kernel;

	private CLProgram programFFT;
	private CLKernel kernelFFT;

	private CLProgram programSumFFTEnergy;
	private CLKernel kernelSumFFTEnergy;
	
	private boolean m_naive;


	// are initialized after transposing the data
	
	// exponents of complex factors multiplied to shift in frequency space
	private OpenCLGrid1D freqU;
	private OpenCLGrid1D freqV;
	
	private OpenCLGrid1D m_shift;
	private OpenCLGrid2D m_maskCL;

	// dft and idft mask used to perform 1d fouriertransform on graphics card
	private ComplexGrid2D dftMatrix;
	private ComplexGrid2D idftMatrix;
 
	// test parameter to see how often shift is performed during optimization
	public int optimizeCounter = 0;

	// parameter used when performing fouriertransform and evaluation of energy in one step without storing 3d-transformed data
	protected final int persistentGroupSize = 128;
	protected static CLBuffer<FloatBuffer> persistentResultBuffer = null;

	protected boolean debug = false;
	protected CLBuffer<FloatBuffer> getPersistentResultBuffer(CLContext context){
		if(persistentResultBuffer==null || persistentResultBuffer.isReleased())
			persistentResultBuffer = context.createFloatBuffer(persistentGroupSize, Mem.WRITE_ONLY);
		else
			persistentResultBuffer.getBuffer().rewind();
		return persistentResultBuffer;
	}
	
	/**
	 * @param data
	 * @param conf
	 * @param naive
	 */
	public MovementCorrection3D(Grid3D data, Config conf, boolean naive){
		m_conf = conf;
		m_data = new ComplexGrid3D(data);
		m_datasize = m_data.getNumberOfElements();
		m_naive = naive;
		m_shift = null;  //new OpenCLGrid1D(2*conf.getNumberOfProjections());
		m_3dFourier = new ComplexGrid3D(conf.getNumberOfProjections(), conf.getHorizontalDim(), conf.getVerticalDim());
		m_3dFourier.activateCL();

		m_mask = conf.getMask();
		m_maskCL = new OpenCLGrid2D(m_mask);
		freqU = new OpenCLGrid1D(conf.getShiftFreqX());
		freqV = new OpenCLGrid1D(conf.getShiftFreqY());
		dftMatrix = new ComplexGrid2D(conf.getDFTMatrix());
		dftMatrix.activateCL();
		idftMatrix = new ComplexGrid2D(conf.getIDFTMatrix());
		idftMatrix.activateCL();

		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();
		
		// opencl programs should have to be compiled only once
		program = null;
		kernel = null;
		programFFT = null;
		kernelFFT = null;
		programSumFFTEnergy = null;
		kernelSumFFTEnergy = null;
	}
	
 
	/**
	 * performs FFT2 on detector dimensions on untransposed data (projection dimension is third) CPU
	 */
	public void doFFT2(){
		Fourier ft = new Fourier();
		ft.fft2(m_data);
		m_data.setSpacing(m_conf.getUSpacing(), m_conf.getVSpacing(), m_conf.getAngleIncrement() );
		m_data.setOrigin(0,0,0);
		System.out.println("2d-FFT done");
		double[] spacings = m_data.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}
	
	
	/**
	 * transposing data (3 ->1, 1->2, 2->3) CPU 
	 */
	public void transposeData(){
		int[] sizeOrig = m_data.getSize();
		m_2dFourierTransposed = new ComplexGrid3D(sizeOrig[2],sizeOrig[0], sizeOrig[1]);
		//		m_2dFourierTransposed.setSpacing(m_data.getSpacing()[2], m_data.getSpacing()[0], m_data.getSpacing()[1]);
		//		m_2dFourierTransposed.setOrigin(m_data.getOrigin());
		for(int angle = 0; angle < sizeOrig[2]; angle++){
			for(int horiz = 0; horiz < sizeOrig[0]; horiz++){
				for(int vert = 0; vert < sizeOrig[1]; vert++){
					//float value = orig.getAtIndex(horiz, vert, angle);
					//if(value > 0){
					m_2dFourierTransposed.setAtIndex(angle, horiz, vert, m_data.getAtIndex(horiz, vert, angle));
					//}

				}
			}
		}

		m_2dFourierTransposed.setSpacing(m_conf.getAngleIncrement() ,m_conf.getUSpacing(), m_conf.getVSpacing());
		m_2dFourierTransposed.setOrigin(0,0,0);

		m_2dFourierTransposed.activateCL();
		

		System.out.println("Transposing done");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}
	
 
	/**
	 * after optimization the image has to be transposed back (+ ifft2) to be displayed in a normal way
	 */
	public void backTransposeData(){
		int[] sizeOrig = m_data.getSize();
		for(int angle = 0; angle < sizeOrig[2]; angle++){
			for(int horiz = 0; horiz < sizeOrig[0]; horiz++){
				for(int vert = 0; vert < sizeOrig[1]; vert++){
					m_data.setAtIndex(horiz, vert, angle, m_2dFourierTransposed.getAtIndex(angle, horiz, vert));
				}
			}
		}
		m_data.setSpacing(m_conf.getUSpacing(), m_conf.getVSpacing(),m_conf.getAngleIncrement());
		System.out.println("Backtransposing done");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}

	/**
	 * perform the fft on angles using dft mask, values are stored in m_3dFourier, not possible for big datasets, GPU
	 */
	public void doFFTAngleCL(){
		if(m_2dFourierTransposed == null){
			return;
		}
		//m_2dFourierTransposed.show("Vor forward transformation");
		CLBuffer<FloatBuffer> dataBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> dftMatBuffer = dftMatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> resultBuffer = m_3dFourier.getDelegate().getCLBuffer();

		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		dftMatrix.getDelegate().prepareForDeviceOperation();
		m_3dFourier.getDelegate().prepareForDeviceOperation();

		if(programFFT == null || kernelFFT == null){
			try {
				programFFT = context.createProgram(MovementCorrection3D.class.getResourceAsStream("matrixMul.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			programFFT.build();
			kernelFFT = programFFT.createCLKernel("dftMatrixMul");
		}
		kernelFFT.rewind();
		kernelFFT.putArg(resultBuffer);
		kernelFFT.putArg(dftMatBuffer);
		kernelFFT.putArg(dataBuffer);
		kernelFFT.putArg(m_conf.getNumberOfProjections());
		kernelFFT.putArg(m_conf.getNumberOfProjections());
		kernelFFT.putArg(m_conf.getNumberOfProjections());
		kernelFFT.putArg(m_conf.getHorizontalDim());
		kernelFFT.putArg(m_conf.getVerticalDim());


		int localWorksize = 10;
		long globalWorksizeA = OpenCLUtil.roundUp(localWorksize, m_3dFourier.getSize()[0]);
		long globalWorksizeB = OpenCLUtil.roundUp(localWorksize, m_3dFourier.getSize()[1]);
		long globalWorksizeC = OpenCLUtil.roundUp(localWorksize, m_3dFourier.getSize()[2]);	

		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put3DRangeKernel(kernelFFT, 0, 0, 0, globalWorksizeA, globalWorksizeB, globalWorksizeC, localWorksize,localWorksize,localWorksize).finish();
		commandQueue.release();
		m_3dFourier.getDelegate().notifyDeviceChange();
	}
	
	/**
	 * computes fft on projection and sums up energies, should work on normal datasets, GPU
	 * @return sum of relevant energy
	 */
	public float getFFTandEnergy(){
		//CLBuffer<FloatBuffer> dataBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> dftMatBuffer = dftMatrix.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> maskBuffer = m_maskCL.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> dataBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();

		int elementCount = m_2dFourierTransposed.getNumberOfElements();

		int localWorkSize = 256;
		int globalWorkSize = 32768;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount, persistentGroupSize),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;


		CLBuffer<FloatBuffer> resultBuffer = getPersistentResultBuffer(context);

		long time  = System.currentTimeMillis();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		dftMatrix.getDelegate().prepareForDeviceOperation();
		m_maskCL.getDelegate().prepareForDeviceOperation();
		long time1 = System.currentTimeMillis(); 
		long difftime = time1 - time;
		System.out.println("Time needed for copying whole dataset to GPU = " + difftime);
		if(programSumFFTEnergy == null || kernelSumFFTEnergy == null){
			try {
				programSumFFTEnergy = context.createProgram(MovementCorrection3D.class.getResourceAsStream("sumFFTEnergy.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			programSumFFTEnergy.build();
			kernelSumFFTEnergy = programSumFFTEnergy.createCLKernel("sumFFTEnergy");
		}
		kernelSumFFTEnergy.rewind();
		kernelSumFFTEnergy.putArg(dataBuffer);
		kernelSumFFTEnergy.putArg(dftMatBuffer);
		kernelSumFFTEnergy.putArg(maskBuffer);
		kernelSumFFTEnergy.putArg(resultBuffer);
		kernelSumFFTEnergy.putArg(nperGroup);
		kernelSumFFTEnergy.putArg(nperWorkItem);
		kernelSumFFTEnergy.putArg(m_2dFourierTransposed.getSize()[0]);
		kernelSumFFTEnergy.putArg(m_2dFourierTransposed.getSize()[1]);
		kernelSumFFTEnergy.putArg(m_2dFourierTransposed.getSize()[2]);

		CLCommandQueue commandqueue = device.createCommandQueue();
		long time2 = System.currentTimeMillis();
		difftime = time2 - time1;
		System.out.println("Time needed copying Graphicscard = " + difftime);
		commandqueue.put1DRangeKernel(kernelSumFFTEnergy, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(resultBuffer, true)
		.finish();
		commandqueue.release();
		long time3 = System.currentTimeMillis();
		difftime =time3 - time2;
		System.out.println("Complete time on GPU = " + difftime);
		
		float sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()){
			sum += resultBuffer.getBuffer().get();
		}
		
		return sum;

	}
	
	/**
	 * problem with idft on GPU, performs backtransformation on CPU
	 */
	public void doiFFTAngleCL(){
		
		if(m_3dFourier == null){
			return;
		}
		ComplexPointwiseOperators cpo = new  ComplexPointwiseOperators();
		cpo.copy(m_2dFourierTransposed, m_3dFourier);
		doiFFTAngle();
				
//				CLBuffer<FloatBuffer> dataBuffer = m_3dFourier.getDelegate().getCLBuffer();
//				CLBuffer<FloatBuffer> idftMatBuffer = idftMatrix.getDelegate().getCLBuffer();
//				CLBuffer<FloatBuffer> resultBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();
//				
//				m_3dFourier.getDelegate().prepareForDeviceOperation();
//				dftMatrix.getDelegate().prepareForDeviceOperation();
//				m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
//				
//				if(programFFT == null || kernelFFT == null){
//					try {
//						programFFT = context.createProgram(MovementCorrection3D.class.getResourceAsStream("matrixMul.cl"));
//						} catch (IOException e) {
//						e.printStackTrace();
//						}
//						programFFT.build();
//						kernelFFT = programFFT.createCLKernel("dftMatrixMul");
//				}
//				kernelFFT.rewind();
//				
//				kernelFFT.putArg(resultBuffer);
//				kernelFFT.putArg(idftMatBuffer);
//				kernelFFT.putArg(dataBuffer);
//				kernelFFT.putArg(m_conf.getNumberOfProjections());
//				kernelFFT.putArg(m_conf.getNumberOfProjections());		
//				kernelFFT.putArg(m_conf.getNumberOfProjections());
//				kernelFFT.putArg(m_conf.getHorizontalDim());
//				kernelFFT.putArg(m_conf.getVerticalDim());
//				
//				
//				int localWorksize = 10;
//				long globalWorksizeA = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[0]);
//				long globalWorksizeB = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[1]);
//				long globalWorksizeC = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[2]);	
//				
//				CLCommandQueue commandQueue = device.createCommandQueue();
//		//		commandQueue.put2DRangeKernel(kernelFFT, 0, 0, globalWorksizeA, globalWorksizeB, localWorksize, localWorksize).finish();
//				commandQueue.put3DRangeKernel(kernelFFT, 0, 0, 0, globalWorksizeA, globalWorksizeB, globalWorksizeC, localWorksize,localWorksize,localWorksize).finish();
//				
//				
//				m_2dFourierTransposed.getDelegate().notifyDeviceChange();
//				//idftMatrix.getDelegate().notifyDeviceChange();
//				//m_3dFourier.getDelegate().notifyDeviceChange();
//				//m_2dFourierTransposed.show("zuruecktransformiert");
	}

	/**
	 * "normal" fft on projectionangle, CPU
	 */
	public void doFFTAngle(){
		long time = System.currentTimeMillis();
		if(m_2dFourierTransposed == null){
			return;
		}
		Fourier ft = new Fourier();
		ft.fft(m_2dFourierTransposed);
		m_2dFourierTransposed.setSpacing(m_conf.getKSpacing(),m_conf.getUSpacing(), m_conf.getVSpacing());
		m_2dFourierTransposed.setOrigin(0,0,0);
		System.out.println("FFT on angle done");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}

		time = System.currentTimeMillis()-time;
		System.out.println("Time for forward fft:"+ time);
	}
	
	/**
	 * "normal" ifft on projectionangle, GPU
	 */
	public void doiFFTAngle(){
		Fourier ft = new Fourier();
		ft.ifft(m_2dFourierTransposed);

		m_2dFourierTransposed.setSpacing(m_conf.getAngleIncrement(),m_conf.getUSpacing(), m_conf.getVSpacing());
		m_2dFourierTransposed.setOrigin(0,0,0);
		System.out.println("ifft on angle done");
		double[] spacings = m_2dFourierTransposed.getSpacing();

		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}

	}
	

	/**
	 * shift in frequency space on CPU
	 */
	public void applyShift(){
		long time = System.currentTimeMillis();
		//precomputed angles(-2*pi*xi/N) for u and v direction
		Grid1D shiftFreqX = m_conf.getShiftFreqX();
		Grid1D shiftFreqY = m_conf.getShiftFreqY();
	
		for(int angle = 0; angle < m_2dFourierTransposed.getSize()[0]; angle++){
			// get the shifts in both directions (in pixel)
			float shiftX = m_shift.getAtIndex(angle*2);
			float shiftY = m_shift.getAtIndex(angle*2+1);
			
			for(int u = 0; u < m_2dFourierTransposed.getSize()[1]; u++){
				//number representing phase of shift in x-direction	in complex number						
				float angleX = shiftFreqX.getAtIndex(u)*shiftX;
				//Complex expShiftX = shiftComplexFreqX.getAtIndex(u).power(shiftX);
				for(int v = 0; v < m_2dFourierTransposed.getSize()[2]; v++){
					
					// exponent of complex number representing shift in both directions					
					float sumAngles = angleX + shiftFreqY.getAtIndex(v)*shiftY;

					// complex number representing both shifts	
					// multiply at position in complex grid
					Complex shift = getComplexFromAngles(sumAngles);			
					m_2dFourierTransposed.multiplyAtIndex(angle, u, v,shift);

				}
			}
		}
		time = System.currentTimeMillis()-time;
		System.out.println("Time for complete shift:"+ time);
	}

	/**
	 * paralellized shift on GPU
	 */
	public void parallelShiftOptimized(){
		//m_2dFourierTransposed.show("Before 'shift'");
		CLBuffer<FloatBuffer> dataBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();
		//CLBuffer<FloatBuffer> bufferFourierTransposedCL = m_2dFourierTransposed.getDelegate().getCLBuffer();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		//OpenCLGrid1D shiftCL = new OpenCLGrid1D(m_shift);

		CLBuffer<FloatBuffer> bufferShifts = m_shift.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqU = freqU.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqV = freqV.getDelegate().getCLBuffer();

		m_shift.getDelegate().prepareForDeviceOperation();
		freqU.getDelegate().prepareForDeviceOperation();
		freqV.getDelegate().prepareForDeviceOperation();

		if(program == null || kernel == null){
			try {
				program = context.createProgram(MovementCorrection3D.class.getResourceAsStream("shiftInFourierSpace2DNEW.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			program.build();
			kernel = program.createCLKernel("shift");
		}


		kernel.rewind();
		kernel.putArg(dataBuffer);
		kernel.putArg(bufferFreqU);
		kernel.putArg(bufferFreqV);
		kernel.putArg(bufferShifts);
		kernel.putArg(m_2dFourierTransposed.getSize()[0]);
		kernel.putArg(m_2dFourierTransposed.getSize()[1]);
		kernel.putArg(m_2dFourierTransposed.getSize()[2]);


		int localWorksize = 16;
		long globalWorksizeU = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[1]);
		long globalWorksizeV = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[2]);

		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeU, globalWorksizeV, localWorksize, localWorksize).finish();
		commandQueue.release();
		m_2dFourierTransposed.getDelegate().notifyDeviceChange();		
	}
	
	/**
		compute iFFT2 after all other operations to get displayable data	
	*/
	public void doiFFT2(){
		Fourier ft = new Fourier();
		ft.ifft2(m_data);
		m_data.setSpacing(m_conf.getPixelXSpace(),m_conf.getPixelYSpace(), m_conf.getAngleIncrement());
		m_data.setOrigin(0,0,0);

		System.out.println("2d-iFFT done");
		double[] spacings = m_data.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}

	// get and set functions
	/**
	 * @return conf
 	 */
	public Config getConfig(){
		return m_conf;
	}
	/**
	 * 
	 * @return data
	 */
	public ComplexGrid3D getData(){
		return m_data;
	}
	
	/**
	 * 
	 * @return fouriertransformed and transposed data
	 */
	public ComplexGrid3D get2dFourierTransposedData(){
		return m_2dFourierTransposed;
	}
	
	/**
	 * 
	 * @param inputGrid (pre-fouriertransformed and transposed data)
	 */
	void set2dFourierTransposedData(ComplexGrid3D inputGrid){
		m_2dFourierTransposed = inputGrid;
	}
	
	/**
	 * 
	 * @return data after 3d fouriertransform
	 */
	public ComplexGrid3D get3dFourier(){
		return m_3dFourier;
	}
	/**
	 * sets the shift vector (2*number of projections)
	 * @param shift
	 */
	public void setShiftVector(Grid1D shift){
		if(shift.getSize()[0] != 2* m_conf.getNumberOfProjections()){
			return;
		}
		
		if(m_shift == null)
			m_shift = new OpenCLGrid1D(shift);
		else
			NumericPointwiseOperators.copy(m_shift, shift);
	}

	/**
	 * computes a complex number with real and imaginary part from angle (radius 1)
	 * @param angle
	 * @return
	 */
	private Complex getComplexFromAngles(float angle){
		//float[] result = new float[2];
		float re = (float)(Math.cos(angle));
		float im = (float)(Math.sin(angle));
		return new Complex(re,im);//result;
	}

	
	/**
	 * 
	 * @return shift vector where relevant energies are minimized
	 */
	public Grid1D computeOptimalShift(){
		EnergyToBeMinimized function = new EnergyToBeMinimized();
		FunctionOptimizer fo = new FunctionOptimizer();
		fo.setDimension(2*m_conf.getNumberOfProjections());
		fo.setOptimizationMode(OptimizationMode.Function);
		fo.setConsoleOutput(true);
		double[]min = new double[2*m_conf.getNumberOfProjections()];
		double[]max = new double[2*m_conf.getNumberOfProjections()];
		for(int i = 0; i < min.length; i++){
			min[i] = -20.0;
			max[i] = 20.0;
		}
		fo.setMaxima(max);
		fo.setMinima(min);
		double[] optimalShift = null;
		double minEnergy = Double.MAX_VALUE;
		//for(int i = 0; i < 5; i++){
		//System.out.println("Optimizer: " + i);
		double[] initialGuess = new double[2*m_conf.getNumberOfProjections()];
		fo.setInitialX(initialGuess);
		double [] result = fo.optimizeFunction(function);
		double newVal = function.evaluate(result, 0);
		if ( newVal < minEnergy) {
			optimalShift = result;
			minEnergy = newVal;
		}
		//}

		Grid1D optimalShiftGrid = new Grid1D(optimalShift.length);
		for(int i = 0; i < optimalShift.length; i++){
			optimalShiftGrid.setAtIndex(i, (float)(optimalShift[i]));
		}

		return optimalShiftGrid;
	}




	private class EnergyToBeMinimized implements GradientOptimizableFunction{

		// stores the sum of shifts made up to this point
		Grid1D m_oldGuessAbs = new Grid1D(2*m_conf.getNumberOfProjections());
		// the newest shift to be performed
		Grid1D m_guessRel = new Grid1D(2*m_conf.getNumberOfProjections());
		
		/**
		 * Sets the number of parallel processing blocks. This number should optimally equal to the number of available processors in the current machine. 
		 * Default value should be 1. 
		 * @param number
		 */
		public void setNumberOfProcessingBlocks(int number){

		}

		/**
		 * returns the number of parallel processing blocks.
		 * @return
		 */
		public int getNumberOfProcessingBlocks(){

			return 1;
		}

		/**
		 * Evaluates the function at position x.<BR> 
		 * (Note that x is a Fortran Array which starts at 1.)
		 * @param x the position
		 * @param block the block identifier. First block is 0. block is < getNumberOfProcessingBlocks().
		 * @return the function value at x
		 */
		public double evaluate(double[] x, int block){
			long timeComplete  = System.currentTimeMillis();
			long time  = System.currentTimeMillis();
			long diff = 0;
			System.out.println("Counter: " + (++optimizeCounter));
			for(int i = 0; i < x.length; i++){
				m_guessRel.setAtIndex(i, (float)(x[i]*1e3) - m_oldGuessAbs.getAtIndex(i));
				if(Math.abs(m_guessRel.getAtIndex(i)) != 0.0f){
					System.out.println("pos: " + i + ", val: " + m_guessRel.getAtIndex(i));
				}
				m_oldGuessAbs.setAtIndex(i,(float) (x[i]*1e3));
			}
			long time1 = System.currentTimeMillis();
			diff = time1 - time;
			time = time1;
			System.out.println("Zeit 1: " + diff);
			setShiftVector(m_guessRel);
			
			time1 = System.currentTimeMillis();
			diff = time1 - time;		
			time = time1;
			System.out.println("Zeit 2: " + diff);
			
			parallelShiftOptimized();
			
			time1 = System.currentTimeMillis();
			diff = time1 - time;
			time = time1;
			System.out.println("Zeit 3: " + diff);
			float sum = 0;
			
			// naive approach: do fft on angles, sum up relevant energies, difft 
			if (m_naive){
				doFFTAngleCL();
				for(int proj = 0; proj < m_3dFourier.getSize()[0]; proj++ ){
					for(int u = 0; u < m_3dFourier.getSize()[1]; u++){
						if(m_mask.getAtIndex(proj, u) == 1){
							for(int v = 0; v < m_3dFourier.getSize()[2]; v++){
								sum += m_3dFourier.getAtIndex(proj, u, v).getMagn();
							}
						}
					}
				}
				System.out.println("Berechnete Summe: " + sum);
				doiFFTAngleCL();
				//m_2dFourierTransposed.show("m_2dFourierTransposed after");
				//sum /= m_datasize;
			}
			else{
				sum = getFFTandEnergy();
				System.out.println("Berechnete Summe: " + sum);
				
			}
			time1 = System.currentTimeMillis();
			diff = time1 - time;
			System.out.println("Zeit 4: " + diff);
			
			System.out.println("Summe: " + sum);
			timeComplete = System.currentTimeMillis() - timeComplete;
			
			System.out.println("Complete Time Milis: " + timeComplete);
			return sum;
		}

		public double [] gradient(double[] x, int block){
			return null;
		}

	}
}



