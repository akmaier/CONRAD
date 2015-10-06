package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan;
import com.jogamp.opencl.demos.fft.CLFFTPlan.CLFFTDataFormat;
import com.jogamp.opencl.demos.fft.CLFFTPlan.CLFFTDirection;
import com.jogamp.opencl.demos.fft.CLFFTPlan.InvalidContextException;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class MotionCorrection3DFast extends MovementCorrection3D{

	private CLProgram programTranspose;
	private CLKernel kernelTransposeFwd;
	private CLKernel kernelTransposeBwd;


	public MotionCorrection3DFast(Grid3D data, Config conf, boolean naive) {
		super(data, conf, naive);
	}
	
	private void initTransposeProgramAndKernels(){
		if( programTranspose == null ){
			try {
				programTranspose = m_data.getDelegate().getCLContext().createProgram(MotionCorrection3DFast.class.getResourceAsStream("transpose.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			programTranspose.build();
		}
		if(kernelTransposeFwd == null){
			kernelTransposeFwd = programTranspose.createCLKernel("swapDimensionsFwd");
		}
		if(kernelTransposeBwd == null){
			kernelTransposeBwd = programTranspose.createCLKernel("swapDimensionsBwd");
		}
	}

	@Override
	public void transposeData() {
		int[] sizeOrig = m_data.getSize();
		m_data.activateCL();
		if(m_2dFourierTransposed==null){
			m_2dFourierTransposed = new ComplexGrid3D(sizeOrig[2],sizeOrig[0], sizeOrig[1]);
			m_2dFourierTransposed.setSpacing(m_conf.getAngleIncrement() , m_conf.getUSpacing(), m_conf.getVSpacing());
			m_2dFourierTransposed.setOrigin(0,0,0);
			m_2dFourierTransposed.activateCL();
		}

		m_data.getDelegate().prepareForDeviceOperation();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();

		initTransposeProgramAndKernels();

		kernelTransposeFwd.rewind();
		kernelTransposeFwd.putArg(m_data.getDelegate().getCLBuffer());
		kernelTransposeFwd.putArg(m_2dFourierTransposed.getDelegate().getCLBuffer());
		kernelTransposeFwd.putArg(m_data.getSize()[0]);
		kernelTransposeFwd.putArg(m_data.getSize()[1]);
		kernelTransposeFwd.putArg(m_data.getSize()[2]);
		
		int localWorksize = 512;
		long globalWorksize = OpenCLUtil.roundUp(localWorksize, m_data.getNumberOfElements());
		
		m_data.getDelegate().getCLDevice()
		.createCommandQueue()
		.put1DRangeKernel(kernelTransposeFwd, 0, globalWorksize, localWorksize)
		.finish()
		.release();
		
		m_2dFourierTransposed.getDelegate().notifyDeviceChange();
		
		m_data.getDelegate().prepareForHostOperation();
		m_data.deactivateCL();
		
		System.out.println("Transposing done on (GPU)");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}
	
	@Override
	public void backTransposeData() {
		m_data.activateCL();
		m_data.getDelegate().prepareForDeviceOperation();
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();

		initTransposeProgramAndKernels();

		kernelTransposeBwd.rewind();
		kernelTransposeBwd.putArg(m_2dFourierTransposed.getDelegate().getCLBuffer());
		kernelTransposeBwd.putArg(m_data.getDelegate().getCLBuffer());
		kernelTransposeBwd.putArg(m_2dFourierTransposed.getSize()[0]);
		kernelTransposeBwd.putArg(m_2dFourierTransposed.getSize()[1]);
		kernelTransposeBwd.putArg(m_2dFourierTransposed.getSize()[2]);
		
		int localWorksize = 512;
		long globalWorksize = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getNumberOfElements());
		
		m_2dFourierTransposed.getDelegate().getCLDevice()
		.createCommandQueue()
		.put1DRangeKernel(kernelTransposeBwd, 0, globalWorksize, localWorksize)
		.finish()
		.release();
		
		m_data.getDelegate().notifyDeviceChange();
		m_data.getDelegate().prepareForHostOperation();
		m_data.deactivateCL();
		
		System.out.println("Backtransposing done on (GPU)");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}


	/**
	 * paralellized shift on GPU
	 */
	@Override
	public void applyShift() {
		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		m_shift.getDelegate().prepareForDeviceOperation();
		freqU.getDelegate().prepareForDeviceOperation();
		freqV.getDelegate().prepareForDeviceOperation();

		CLBuffer<FloatBuffer> dataBuffer = m_2dFourierTransposed.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferShifts = m_shift.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqU = freqU.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqV = freqV.getDelegate().getCLBuffer();

		if(program == null || kernel == null){
			try {
				InputStream inStream = MovementCorrection3D.class.getResourceAsStream("shiftInFourierSpace2DNEW.cl");
				BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
				String thisline = null;
				String prog = "";
				while((thisline = br.readLine()) != null){
					prog+=thisline+'\n';
				}
				br.close();
				inStream.close();
				String constants = "";
				constants += "#define numProj " + m_2dFourierTransposed.getSize()[0] + 'u';
				constants += "\n#define numElementsU " + m_2dFourierTransposed.getSize()[1] + 'u';
				constants += "\n#define numElementsV " + m_2dFourierTransposed.getSize()[2] + 'u';
				constants += "\n\n";
				prog = constants + prog;
				program = context.createProgram(prog);
			} catch (IOException e) {
				e.printStackTrace();
			}
			program.build();
			kernel = program.createCLKernel("shift");

			kernel.putArg(dataBuffer);
			kernel.putArg(bufferFreqU);
			kernel.putNullArg((int)bufferFreqU.getCLSize());
			kernel.putArg(bufferFreqV);
			kernel.putNullArg((int)bufferFreqV.getCLSize());
			kernel.putArg(bufferShifts);
			kernel.putNullArg((int)bufferShifts.getCLSize());
		}
		
		int localWorksizeProj = 256;
		int localWorksizeV = 2;
		long globalWorksizeProj = OpenCLUtil.roundUp(localWorksizeProj, m_2dFourierTransposed.getSize()[0]);
		long globalWorksizeV = OpenCLUtil.roundUp(localWorksizeV, m_2dFourierTransposed.getSize()[2]);

		CLCommandQueue commandQueue = device.createCommandQueue();
		//commandQueue.put3DRangeKernel(kernel, 0, 0, 0, globalWorksizeProj, globalWorksizeU, globalWorksizeV, localWorksizeProj, localWorksizeU, localWorksizeV);
		commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeProj, globalWorksizeV, localWorksizeProj, localWorksizeV);
		commandQueue.finish();
		commandQueue.release();
		m_2dFourierTransposed.getDelegate().notifyDeviceChange();
	}




	/**
	 * computes fft on projection and sums up energies, should work on normal datasets, GPU
	 * @return sum of relevant energy
	 */
	@Override
	public float getFFTandEnergy() {
		long time = System.nanoTime();
		long diff = 0;

		if (fft == null){
			try {
				fft = new CLFFTPlan(m_2dFourierTransposed.getDelegate().getCLContext(), new int[]{m_2dFourierTransposed.getSize()[0]}, CLFFTDataFormat.InterleavedComplexFormat);
			} catch (InvalidContextException e1) {
				e1.printStackTrace();
			}
		}

		if(m_3dFourier==null){
			m_3dFourier = new ComplexGrid3D(m_conf.getNumberOfProjections(), m_conf.getHorizontalDim(), m_conf.getVerticalDim());
			m_3dFourier.activateCL();
		}

		m_2dFourierTransposed.getDelegate().prepareForDeviceOperation();
		m_3dFourier.getDelegate().prepareForDeviceOperation();
		CLCommandQueue queue = device.createCommandQueue();
		fft.executeInterleaved(queue, m_2dFourierTransposed.getNumberOfElements()/m_2dFourierTransposed.getSize()[0], 
				CLFFTDirection.Forward, m_2dFourierTransposed.getDelegate().getCLBuffer(), 
				m_3dFourier.getDelegate().getCLBuffer(), null, null);
		queue.finish();
		m_3dFourier.getDelegate().notifyDeviceChange();

		long time1 = System.nanoTime();
		diff = time1 - time;
		time = time1;
		CONRAD.log("Time for 4a) Angular FFT:     " + diff/1e6);

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
		if(programSumFFTEnergy == null || kernelSumFFTEnergy == null){
			try {
				programSumFFTEnergy = context.createProgram(MovementCorrection3D.class.getResourceAsStream("sumFFTEnergy.cl"));
			} catch (IOException e) {
				e.printStackTrace();
			}
			programSumFFTEnergy.build();
			kernelSumFFTEnergy = programSumFFTEnergy.createCLKernel("sumEnergy");
			kernelSumFFTEnergy.putArg(m_3dFourier.getDelegate().getCLBuffer());
			kernelSumFFTEnergy.putArg(m_maskCL.getDelegate().getCLBuffer());
			kernelSumFFTEnergy.putArg(resultBuffer);
			kernelSumFFTEnergy.putArg(nperGroup);
			kernelSumFFTEnergy.putArg(nperWorkItem);
			kernelSumFFTEnergy.putArg(m_maskCL.getNumberOfElements());
			kernelSumFFTEnergy.putArg(m_3dFourier.getNumberOfElements());
		}

		//long time2 = System.nanoTime();
		//difftime = time2 - time1;
		//System.out.println("Time needed copying Graphicscard = " + difftime/1e6);
		m_maskCL.getDelegate().prepareForDeviceOperation();
		m_3dFourier.getDelegate().prepareForDeviceOperation();
		queue.put1DRangeKernel(kernelSumFFTEnergy, 0, globalWorkSize, localWorkSize)
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

		time1 = System.nanoTime();
		diff = time1 - time;
		CONRAD.log("Time for 4b) Summing Mask Energy:     " + diff/1e6);

		return sum;
	}



}
