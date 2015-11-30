/*
 * Copyright (C) 2015 Wolfgang Aichinger, Martin Berger, Katrin Mentl
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import ij.gui.Plot;
import ij.gui.PlotWindow;

import java.awt.Color;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.Map.Entry;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexPointwiseOperators;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;
import edu.stanford.rsl.jpop.OptimizationOutputFunction;
import edu.stanford.rsl.tutorial.fourierConsistency.coneBeam.CLFFTPlan.CLFFTDataFormat;
import edu.stanford.rsl.tutorial.fourierConsistency.coneBeam.CLFFTPlan.CLFFTDirection;
import edu.stanford.rsl.tutorial.fourierConsistency.coneBeam.CLFFTPlan.InvalidContextException;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

public class MovementCorrection3D {

	/**
	 * @param Grid3D data: contains original projection data
	 * @param Config conf: provides parameter of acquisition and precomputed arrays used to shift
	 * @param boolean naive: if true: naive cpu version, if false faster gpu version
	 */

	protected CLFFTPlan fft;
	protected Config m_conf;
	protected ComplexGrid3D m_data;
	
	// stores transposed data (projection dimension becomes first dimension) after 2D-Fouriertransform
	protected ComplexGrid3D m_2dFourierTransposed = null;

	// stores data after fourier transformation on projection dimension
	protected ComplexGrid3D m_3dFourier = null;

	// mask is a precomputed binary mask containing ones where the information after fourier on projection should be zero
	private Grid2D m_mask;
	private Double maskNormalizationSum;

	// opencl members
	protected CLContext context;
	protected CLDevice device;
	protected CLProgram program;
	protected CLKernel kernel;

	private CLProgram programFFT;
	private CLKernel kernelFFT;

	protected CLProgram programSumFFTEnergy;
	protected CLKernel kernelSumFFTEnergy;
	
	protected CLProgram programSumFFTEnergyGradient;
	protected CLKernel kernelSumFFTEnergyGradient;

	private boolean m_naive;


	// are initialized after transposing the data

	// exponents of complex factors multiplied to shift in frequency space
	protected OpenCLGrid1D freqU;
	protected OpenCLGrid1D freqV;
	protected OpenCLGrid1D freqP; 
	
	// ext_p dependent on current gradient optimization parameter alpha_i (index i)
	protected int grad_idx = 0;
	
	protected OpenCLGrid1D m_shift;
	protected OpenCLGrid2D m_maskCL;

	// dft and idft mask used to perform 1d fouriertransform on graphics card
	private ComplexGrid2D dftMatrix;
	
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
		m_naive = naive;
		m_shift = null;

		m_mask = conf.getMask();
		m_maskCL = new OpenCLGrid2D(m_mask);
		freqU = new OpenCLGrid1D(conf.getShiftFreqX());
		freqV = new OpenCLGrid1D(conf.getShiftFreqY());
		freqP = new OpenCLGrid1D(conf.getShiftFreqP());
		
		//dftMatrix = new ComplexGrid2D(conf.getDFTMatrix());
		//dftMatrix.activateCL();
		//idftMatrix = new ComplexGrid2D(conf.getIDFTMatrix());
		//idftMatrix.activateCL();

		context = OpenCLUtil.getStaticContext();
		device = context.getMaxFlopsDevice();

		// opencl programs should have to be compiled only once
		program = null;
		kernel = null;
		programFFT = null;
		kernelFFT = null;
		programSumFFTEnergy = null;
		kernelSumFFTEnergy = null;
		programSumFFTEnergyGradient = null;
		kernelSumFFTEnergyGradient = null;
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
		if(m_3dFourier==null){
			m_3dFourier = new ComplexGrid3D(m_conf.getNumberOfProjections(), m_conf.getHorizontalDim(), m_conf.getVerticalDim());
			m_3dFourier.activateCL();
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
	 * computes sum of relevant energy w.r.t. the gradient
	 * computes fft using ext_p_shift on projection and sums up energies, should work on normal datasets, GPU
	 * @return sum of relevant energy
	 */
	public float getFFTandEnergyGradient() {
		CLCommandQueue queue = device.createCommandQueue();
		//from here on we have the 3DFFT of the projection data shifted with T_s
		
		/*long time1 = System.nanoTime();
		diff = time1 - time;
		time = time1;
		CONRAD.log("Time for 4a) Angular FFT:     " + diff/1e6);*/

		int elementCount = m_3dFourier.getNumberOfElements();

		int localWorkSize = 256;
		int globalWorkSize = 32768;
		// nperGroup needs to be multiples of localWorkSize (this causes overhead for small arrays with length < globalWorkSize)
		int nperGroup = (OpenCLUtil.iDivUp(OpenCLUtil.iDivUp(elementCount, persistentGroupSize),localWorkSize))*localWorkSize;
		// should always be an exact integer, thus no div up necessary
		int nperWorkItem = nperGroup/localWorkSize;
		CLBuffer<FloatBuffer> resultBuffer = getPersistentResultBuffer(context);
		

		//long time  = System.nanoTime();
		//long time1 = System.nanoTime(); 
		//long difftime = time1 - time;
		//System.out.println("Time needed for copying whole dataset to GPU = " + difftime/1e6);
		
		/*for(int i = 0; i < freqP.getNumberOfElements(); i++){
			System.out.println("FreqP At Index i: " + i + " is " + freqP.getAtIndex(i));
		}*/
		
		if(programSumFFTEnergyGradient == null || kernelSumFFTEnergyGradient == null){
			try {
				programSumFFTEnergyGradient = context.createProgram(MovementCorrection3D.class.getResourceAsStream("sumFFTEnergyGradient.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			programSumFFTEnergyGradient.build();
			kernelSumFFTEnergyGradient = programSumFFTEnergyGradient.createCLKernel("sumEnergyGradient");
			kernelSumFFTEnergyGradient.putArg(m_3dFourier.getDelegate().getCLBuffer());
			
			
			kernelSumFFTEnergyGradient.putArg(m_2dFourierTransposed.getDelegate().getCLBuffer()); 
			kernelSumFFTEnergyGradient.putArg(freqU.getDelegate().getCLBuffer()); 
			kernelSumFFTEnergyGradient.putArg(freqV.getDelegate().getCLBuffer()); 
			kernelSumFFTEnergyGradient.putArg(freqP.getDelegate().getCLBuffer()); 
			
			kernelSumFFTEnergyGradient.putArg(m_maskCL.getDelegate().getCLBuffer());
			kernelSumFFTEnergyGradient.putArg(resultBuffer);
			kernelSumFFTEnergyGradient.putArg(nperGroup);
			kernelSumFFTEnergyGradient.putArg(nperWorkItem);
			kernelSumFFTEnergyGradient.putArg(m_maskCL.getNumberOfElements());
			kernelSumFFTEnergyGradient.putArg(m_3dFourier.getNumberOfElements());
			
			kernelSumFFTEnergyGradient.putArg(m_2dFourierTransposed.getSize()[0]);
			kernelSumFFTEnergyGradient.putArg(m_2dFourierTransposed.getSize()[1]);
			kernelSumFFTEnergyGradient.putArg(m_2dFourierTransposed.getSize()[2]);
			
			kernelSumFFTEnergyGradient.putArg(grad_idx);
			kernelSumFFTEnergyGradient.putArg((float) m_conf.getAngleIncrement());
			
		}

		//long time2 = System.nanoTime();
		//difftime = time2 - time1;
		//System.out.println("Time needed copying Graphicscard = " + difftime/1e6);
		kernelSumFFTEnergyGradient.setArg(14, grad_idx);
		m_maskCL.getDelegate().prepareForDeviceOperation();
		m_3dFourier.getDelegate().prepareForDeviceOperation();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		freqU.getDelegate().prepareForDeviceOperation();
		freqV.getDelegate().prepareForDeviceOperation();
		freqP.getDelegate().prepareForDeviceOperation();

		queue.put1DRangeKernel(kernelSumFFTEnergyGradient, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(resultBuffer, true)
		.finish();
		queue.release();

		//m_3dFourier.show("2D FFT results");
		//m_3dFourier.show("3D FFT results");
		//long time3 = System.nanoTime();
		//difftime =time3 - time2;
		//System.out.println("Complete time on GPU = " + difftime/1e6);

		float sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()){
			sum += resultBuffer.getBuffer().get();
		}

		/*time1 = System.nanoTime();
		diff = time1 - time;
		CONRAD.log("Time for 4b) Summing Mask Energy:     " + diff/1e6);*/

		return sum;
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

		long time  = System.nanoTime();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		dftMatrix.getDelegate().prepareForDeviceOperation();
		m_maskCL.getDelegate().prepareForDeviceOperation();
		long time1 = System.nanoTime(); 
		long difftime = time1 - time;
		System.out.println("Time needed for copying whole dataset to GPU = " + difftime/1e6);
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
		long time2 = System.nanoTime();
		difftime = time2 - time1;
		System.out.println("Time needed copying Graphicscard = " + difftime/1e6);
		commandqueue.put1DRangeKernel(kernelSumFFTEnergy, 0, globalWorkSize, localWorkSize)
		.putReadBuffer(resultBuffer, true)
		.finish();
		commandqueue.release();
		long time3 = System.nanoTime();
		difftime =time3 - time2;
		System.out.println("Complete time on GPU = " + difftime/1e6);

		float sum = 0;
		while (resultBuffer.getBuffer().hasRemaining()){
			sum += resultBuffer.getBuffer().get();
			//sum = resultBuffer.getBuffer().get();
			//System.out.println("resultBuffer: " + sum);
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
		long time = System.nanoTime();
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

		time = System.nanoTime()-time;
		System.out.println("Time for forward fft:"+ time/1e6);
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
		long time = System.nanoTime();
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
		time = System.nanoTime()-time;
		System.out.println("Time for complete shift:"+ time/1e6);
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
		fo.setOptimizationMode(OptimizationMode.Gradient);
		fo.setConsoleOutput(true);
		double[]min = new double[2*m_conf.getNumberOfProjections()];
		double[]max = new double[2*m_conf.getNumberOfProjections()];
		for(int i = 0; i < min.length; i++){
			min[i] = -20.0;
			max[i] = 20.0;
		}
		fo.setMaxima(max);
		fo.setMinima(min);
		fo.setItnlim(m_conf.getNumberOfIterations());
		fo.setMsg(16);
		fo.setNdigit(6);
		fo.setIexp(1);
		ArrayList<OptimizationOutputFunction> visFcts = new ArrayList<OptimizationOutputFunction>();
		visFcts.add(function);
		fo.setCallbackFunctions(visFcts);
		double[] optimalShift = null;
		double minEnergy = Double.MAX_VALUE;
		//for(int i = 0; i < 5; i++){
		//System.out.println("Optimizer: " + i);
		double[] initialGuess = new double[2*m_conf.getNumberOfProjections()];
		fo.setInitialX(initialGuess);
		double [] result = fo.optimizeFunction(function);
		
		double newVal = function.evaluate(result, 0);
		
		if (newVal < minEnergy) {
			optimalShift = result;
			minEnergy = newVal;
		}

		Grid1D optimalShiftGrid = new Grid1D(optimalShift.length);
		for(int i = 0; i < optimalShift.length; i++){
			optimalShiftGrid.setAtIndex(i, (float)(optimalShift[i]));
		}

		return optimalShiftGrid;
	}




	private class EnergyToBeMinimized implements GradientOptimizableFunction, OptimizationOutputFunction{

		// stores the sum of shifts made up to this point
		Grid1D m_oldGuessAbs = new Grid1D(2*m_conf.getNumberOfProjections());
		// the newest shift to be performed
		//OpenCLGrid1D m_guessRel = new OpenCLGrid1D(new Grid1D(2*m_conf.getNumberOfProjections()));

		private TreeMap<Integer,Double> resultVisualizer;

		private PlotWindow resultVisualizerPlot;

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
			//System.out.println("in evaluate method");
			long timeComplete  = System.nanoTime();
			long time  = System.nanoTime();
			long diff = 0;
			CONRAD.log("*******************\n*******************\nCounter: " + (++optimizeCounter));

			if (m_shift == null)
				m_shift = new OpenCLGrid1D(new Grid1D(x.length));
			m_shift.getDelegate().prepareForHostOperation();
			
			//use relative shifts if the shifts are done in place 
			for(int i = 0; i < x.length; i++){
				m_shift.setAtIndex(i, (float)(x[i]) - m_oldGuessAbs.getAtIndex(i));
				if(Math.abs(m_shift.getAtIndex(i)) != 0.0f){
					//CONRAD.log("pos: " + i + ", val: " + m_guessRel.getAtIndex(i));
				}
				m_oldGuessAbs.setAtIndex(i,(float) (x[i]));
			}
			
			m_shift.getDelegate().notifyHostChange();

			long time1 = System.nanoTime();
			diff = time1 - time;
			time = time1;
			CONRAD.log("Time for 1) Relative motion vector:     " + diff/1e6);
			//setShiftVector(m_guessRel);
			//m_shift = m_guessRel;

			//time1 = System.nanoTime();
			//diff = time1 - time;		
			//time = time1;
			//CONRAD.log("Time for 2) Setting the motion vector:  " + diff/1e6);

			applyShift(); 

			time1 = System.nanoTime();
			diff = time1 - time;
			time = time1;
			CONRAD.log("Time for 3) Performing the motion: 	    " + diff/1e6);
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
				CONRAD.log("Berechnete Summe: " + sum);
				doiFFTAngleCL();
				//m_2dFourierTransposed.show("m_2dFourierTransposed after");
				//sum /= m_datasize;
			}
			else{
				sum = getFFTandEnergy();
				//CONRAD.log("Berechnete Summe: " + sum);

			}
			time1 = System.nanoTime();
			diff = time1 - time;
			CONRAD.log("Time for 4) Performing the FTT + Sum:	" + diff/1e6);


			timeComplete = System.nanoTime() - timeComplete;
			CONRAD.log("*******************\nComplete Time in [ms]: " + timeComplete/1e6);

			if(maskNormalizationSum == null){
				maskNormalizationSum = 1.0;
				//maskNormalizationSum *= (double)(NumericPointwiseOperators.sum(m_mask)*m_conf.getNumberOfProjections());
				//maskNormalizationSum*=m_2dFourierTransposed.getNumberOfElements();
				maskNormalizationSum = sum/1e2;
			}
			sum/=maskNormalizationSum;
			CONRAD.log("Calculated Energy: " + sum);
			return sum;
		}

		
		/**
		 * Computes the Gradient at position x.
		 * (Note that x is a Fortran Array which starts at 1.)
		 * @param x the position
		 * @param block the block identifier. First block is 0. block is < getNumberOfProcessingBlocks().
		 * @return the gradient at x. (In Fortran Style)
		 */
		public double [] gradient(double[] x, int block){
			double[] gradient = new double[x.length];
			
			for(int i = 0; i < x.length; i++){ //walk over all alphas
				grad_idx = i;
				float sum = getFFTandEnergyGradient(); //fast version on GPU
				gradient[i] = 2*sum/maskNormalizationSum;
				//calculate sum by reduction (Energy in the mask)
				//System.out.println("Finished partial derivative for: " + grad_idx + " gradient: " + gradient[i]);
			}
			return gradient;
		}

		
		@Override
		public void optimizerCallbackFunction(int currIterationNumber,
				double[] x, double currFctVal, double[] gradientAtX) {

			// Visualization of parameter vector
			this.visualize(x);
			// Visualization of gradient vector
			this.visualizeGradient(gradientAtX);
			
			/*for(int i = 0; i < gradientAtX.length; i++){
				System.out.println("gradient at " + i + " is " + gradientAtX[i]);	
			}*/

			// Visualization of cost function value over time
			if (this.resultVisualizer == null)
				resultVisualizer = new TreeMap<Integer, Double>();
			resultVisualizer.put(currIterationNumber, currFctVal);
			if (resultVisualizerPlot != null)
				resultVisualizerPlot.close();

			Grid1D out = new Grid1D(resultVisualizer.size());
			Iterator<Entry<Integer,Double>> it = resultVisualizer.entrySet().iterator();
			while (it.hasNext()) {
				Entry<Integer,Double> e = it.next();
				out.setAtIndex(e.getKey(), e.getValue().floatValue());
			}
			resultVisualizerPlot = VisualizationUtil.createPlot(out.getBuffer()).show();

			if (m_2dFourierTransposed != null && currIterationNumber%5==0) {
				boolean firstShow = (outputCentralSino == null);
				if (firstShow){
					outputCentralSino = new Grid3D(m_data.getSize()[0], m_data.getSize()[1], m_data.getSize()[2]);
				}
				backTransposeData();
				doiFFT2();
				NumericPointwiseOperators.copy(outputCentralSino, m_data.getRealGrid());
				
				if (firstShow)
					outputCentralSino.show("Intermediate Projection Result");
			}
			
			// Visualization of 3D FFT
			if (currIterationNumber%5==0 && m_3dFourier!=null){
				boolean firstShow = false;
				if(outputLog3DFFT==null){
					outputLog3DFFT = new Grid3D(m_3dFourier.getSize()[0],m_3dFourier.getSize()[1],m_3dFourier.getSize()[2]);
					firstShow=true;
				}

				NumericPointwiseOperators.copy(outputLog3DFFT, m_3dFourier.getMagGrid());
				NumericPointwiseOperators.addBy(outputLog3DFFT, 1);
				NumericPointwiseOperators.log(outputLog3DFFT);

				if (firstShow)
					outputLog3DFFT.show("3D Fourier Transform Output");
			}
		}

		
		//Visualisation methods for parameters and gradients
		private Plot[] plots = null;
		private double[] oldOutSave;
		private PlotWindow[] plotWindows;
		private Plot[] plotsGradient = null;
		private double[] oldOutSaveGradient;
		private PlotWindow[] plotWindowsGradient;
		
		private Grid3D outputLog3DFFT;
		private Grid3D outputCentralSino;

		public void visualize(double[] out) {
			String[] titles = new String[] {"t_u", "t_v"};
			double[][] oldOutVecs = new double[titles.length][out.length/titles.length];
			double[][] outVecs = new double[titles.length][out.length/titles.length];
			if(plots==null)
				plots = new Plot[titles.length];
			if(plotWindows==null)
				plotWindows = new PlotWindow[titles.length];

			for (int j = 0; j < titles.length; j++) {
				for (int i = 0; i < out.length/titles.length; i++) {
					outVecs[j][i]=out[i*titles.length+j];
					if(oldOutSave!=null){
						oldOutVecs[j][i]=oldOutSave[i*titles.length+j];
					}
				}
				if(plotWindows[j]!=null)
					plotWindows[j].close();

				plots[j]=VisualizationUtil.createPlot(outVecs[j],titles[j], "Proj Idx", titles[j]);
				if(oldOutSave!=null){
					double[] xValues = new double[out.length/titles.length];
					for (int k = 0; k < xValues.length; k++) {
						xValues[k] = k + 1;
					}
					plots[j].setColor(Color.red);
					plots[j].addPoints(xValues,oldOutVecs[j], 2);
				}
				plotWindows[j] = plots[j].show();
			}

			if(oldOutSave==null)
				oldOutSave = new double[out.length];
			System.arraycopy(out, 0, oldOutSave, 0, out.length);
		}
		
		public void visualizeGradient(double[] out) {
			String[] titles = new String[] {"gradient_u", "gradient_v"};
			double[][] oldOutVecs = new double[titles.length][out.length/titles.length];
			double[][] outVecs = new double[titles.length][out.length/titles.length];
			if(plotsGradient==null)
				plotsGradient = new Plot[titles.length];
			if(plotWindowsGradient==null)
				plotWindowsGradient = new PlotWindow[titles.length];

			for (int j = 0; j < titles.length; j++) {
				for (int i = 0; i < out.length/titles.length; i++) {
					outVecs[j][i]=out[i*titles.length+j];
					if(oldOutSaveGradient!=null){
						oldOutVecs[j][i]=oldOutSaveGradient[i*titles.length+j];
					}
				}
				if(plotWindowsGradient[j]!=null)
					plotWindowsGradient[j].close();

				plotsGradient[j]=VisualizationUtil.createPlot(outVecs[j],titles[j], "Proj Idx", titles[j]);
				if(oldOutSaveGradient!=null){
					double[] xValues = new double[out.length/titles.length];
					for (int k = 0; k < xValues.length; k++) {
						xValues[k] = k + 1;
					}
					plotsGradient[j].setColor(Color.red);
					plotsGradient[j].addPoints(xValues,oldOutVecs[j], 2);
				}
				plotWindowsGradient[j] = plotsGradient[j].show();
			}

			if(oldOutSaveGradient==null)
				oldOutSaveGradient = new double[out.length];
			System.arraycopy(out, 0, oldOutSaveGradient, 0, out.length);
		}
	}
}



