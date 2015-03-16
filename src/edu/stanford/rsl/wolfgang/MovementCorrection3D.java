package edu.stanford.rsl.wolfgang;

import java.io.IOException;
import java.nio.FloatBuffer;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid1D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid3D;
import edu.stanford.rsl.conrad.opencl.OpenCLUtil;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;


import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLImage2d;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class MovementCorrection3D {

	/**
	 * @param args
	 */
	private Config m_conf;
	private ComplexGrid3D m_data;
	private ComplexGrid3D m_2dFourierTransposed = null;
	private Grid1D m_shift;
	private Grid2D m_mask;
	public MovementCorrection3D(Grid3D data, Config conf){
		m_conf = conf;
		m_data = new ComplexGrid3D(data);
		m_shift = new Grid1D(2*conf.getNumberOfProjections());
		m_mask = conf.getMask();
	}
	
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
		
		
		System.out.println("Transposing done");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
	}
	
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
	
	public void doFFTAngle(){
		long time = System.currentTimeMillis();
		if(m_2dFourierTransposed == null){
			return;
		}
		Fourier ft = new Fourier();
		ft.fft(m_2dFourierTransposed);
		m_2dFourierTransposed.setSpacing(m_conf.getKSpacing(),m_conf.getUSpacing(), m_conf.getVSpacing());
		m_2dFourierTransposed.setOrigin(0,0,0);
		int length = m_conf.getNumberOfProjections();
//		for(int i = 0; i < m_2dFourierTransposed.getSize()[0]; i++){
//			for(int j = 0; j < m_2dFourierTransposed.getSize()[1]; j++){
//				for(int k = 0; k < m_2dFourierTransposed.getSize()[2]; k++){
//					m_2dFourierTransposed.divideAtIndex(i, j, k, length);
//				}
//			}
//		}
		System.out.println("FFT on angle done");
		double[] spacings = m_2dFourierTransposed.getSpacing();
		for(int i = 0; i <spacings.length; i++){
			System.out.println("Dimension "+ i + ": "+ spacings[i]);
		}
		
		time = System.currentTimeMillis()-time;
		System.out.println("Time for forward fft:"+ time);
		
	}
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
	public void applyShift(){
		long time = System.currentTimeMillis();
		//precomputed angles(-2*pi*xi/N) for u and v direction
		Grid1D shiftFreqX = m_conf.getShiftFreqX();
		Grid1D shiftFreqY = m_conf.getShiftFreqY();
		//ComplexGrid1D shiftComplexFreqX = m_conf.getComplexShiftFreqX();
		//ComplexGrid1D shiftComplexFreqY = m_conf.getComplexShiftFreqY();
		//float xShiftNormFactor = (float)(1.0f/(m_conf.getPixelXSpace()*m_conf.getUSpacing())); // = 640
		//float yShiftNormFactor = (float)(1.0f/(m_conf.getPixelYSpace()*m_conf.getVSpacing())); // = 480
		for(int angle = 0; angle < m_2dFourierTransposed.getSize()[0]; angle++){
			// get the shifts in both directions (in pixel)
			float shiftX = m_shift.getAtIndex(angle*2);
			float shiftY = m_shift.getAtIndex(angle*2+1);
			//System.out.println(angle+1 + " of "+ m_data.getSize()[2]);
			for(int u = 0; u < m_2dFourierTransposed.getSize()[1]; u++){
				//complex number representing shift in x-direction
				//Complex expShiftX = new Complex(Math.cos(shiftFreqX[u]*shiftX/*xShiftNormFactor/*+0.001*/),Math.sin(shiftFreqX[u]*shiftX/*xShiftNormFactor/*+0.001*/));			
				float angleX = shiftFreqX.getAtIndex(u)*shiftX;
				//Complex expShiftX = shiftComplexFreqX.getAtIndex(u).power(shiftX);
				for(int v = 0; v < m_2dFourierTransposed.getSize()[2]; v++){
					// complex number representing shift in y-direction
					//Complex expShiftY = new Complex(Math.cos(shiftFreqY[v]*shiftY/*yShiftNormFactor /*+0.001*/),Math.sin(shiftFreqY[v]*shiftY/*yShiftNormFactor/*+0.001*/));
					
					/*test */
//					float angleY = shiftFreqY[v]*shiftY;
//					System.out.println("Angle x: "+angleX+", Angle y: "+angleY);
					float sumAngles = angleX + shiftFreqY.getAtIndex(v)*shiftY;
					
					//Complex expShiftY = shiftComplexFreqY.getAtIndex(v).power(shiftY);
					// complex number representing both shifts
					//expShiftY = expShiftY.mul(expShiftX);		
					// multiply at position in complex grid
					Complex shift = getComplexFromAngles(sumAngles);
//					float newVal = m_data.getRealAtIndex(u, v, angle);
//					m_data.setAtIndex(u, v, angle, newVal*shift[0], newVal*shift[1]);//(u, v, angle, shift);
					m_2dFourierTransposed.multiplyAtIndex(angle, u, v,shift);
					
				}
			}
		}
		time = System.currentTimeMillis()-time;
		System.out.println("Time for complete shift:"+ time);
	}
	
	
	public void parallelizedApplyShift(){
		long startTime = System.currentTimeMillis();
		OpenCLGrid3D realPart = new OpenCLGrid3D(m_2dFourierTransposed.getRealGrid());
		OpenCLGrid3D imagPart = new OpenCLGrid3D(m_2dFourierTransposed.getImagGrid());
		OpenCLGrid1D freqU = new OpenCLGrid1D(m_conf.getShiftFreqX());
		OpenCLGrid1D freqV = new OpenCLGrid1D(m_conf.getShiftFreqY());
		OpenCLGrid1D shifts = new OpenCLGrid1D(m_shift);
		
		// read and write buffers
		CLBuffer<FloatBuffer> bufferRealPart = realPart.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferImagPart = imagPart.getDelegate().getCLBuffer();
		
		// only read buffer
		CLBuffer<FloatBuffer> bufferFreqU = freqU.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqV = freqV.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferShifts = shifts.getDelegate().getCLBuffer();
		
		
		realPart.getDelegate().prepareForDeviceOperation();
		imagPart.getDelegate().prepareForDeviceOperation();
		
		freqU.getDelegate().prepareForDeviceOperation();
		freqV.getDelegate().prepareForDeviceOperation();
		shifts.getDelegate().prepareForDeviceOperation();
		
		
		// create Context
		CLContext context = OpenCLUtil.getStaticContext();
		// choose fastest device
		CLDevice device = context.getMaxFlopsDevice();
		CLProgram program = null;
		try {
		program = context.createProgram(MovementCorrection3D.class.getResourceAsStream("shiftInFourierSpace.cl"));
		} catch (IOException e) {
		e.printStackTrace();
		}
		program.build();
		
		CLKernel kernel = program.createCLKernel("shiftInFourierSpace");
		kernel.putArg(bufferRealPart);
		kernel.putArg(bufferImagPart);
		kernel.putArg(bufferFreqU);
		kernel.putArg(bufferFreqV);
		kernel.putArg(bufferShifts);
		kernel.putArg(m_2dFourierTransposed.getSize()[0]);
		kernel.putArg(m_2dFourierTransposed.getSize()[1]);
		kernel.putArg(m_2dFourierTransposed.getSize()[2]);

		int localWorksize = 8;
		long globalWorksizeAngle = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[0]);
		long globalWorksizeU = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[1]);
		long globalWorksizeV = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[2]);

		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put3DRangeKernel(kernel, 0, 0, 0, globalWorksizeAngle, globalWorksizeU, globalWorksizeV, localWorksize, localWorksize, localWorksize).finish();
		
		realPart.getDelegate().notifyDeviceChange();
		imagPart.getDelegate().notifyDeviceChange();
		
		long timeAfterFunction = System.currentTimeMillis();
		long diff1 = timeAfterFunction - startTime;
		System.out.println("time for applying shift parallel: "+ diff1);
		Grid3D realCPU = new Grid3D(realPart);
		Grid3D imagCPU = new Grid3D(imagPart);
		
		m_2dFourierTransposed = new ComplexGrid3D(realCPU, imagCPU);
		m_2dFourierTransposed.setOrigin(0,0,0);
		m_2dFourierTransposed.setSpacing(m_conf.getAngleIncrement() ,m_conf.getUSpacing(), m_conf.getVSpacing());
		
		long diff2 = System.currentTimeMillis() - timeAfterFunction;
		long diff3 = System.currentTimeMillis() - startTime;
		System.out.println("time getting back to cpu: "+ diff2);
		System.out.println("complete time applying shift on graphics card" + diff3);
		
	}
	
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
	
	public Config getConfig(){
		return m_conf;
	}
	public ComplexGrid3D getData(){
		return m_data;
	}
	public ComplexGrid3D get2dFourierTransposedData(){
		return m_2dFourierTransposed;
	}
	public void setShiftVector(Grid1D shift){
		if(shift.getSize()[0] != 2* m_conf.getNumberOfProjections()){
			return;
		}
		m_shift = shift;
	}
	
	private Complex getComplexFromAngles(float angle){
		//float[] result = new float[2];
		float re = (float)(Math.cos(angle));
		float im = (float)(Math.sin(angle));
		return new Complex(re,im);//result;
	}
	
	
	
	
	private class EnergyToBeMinimized implements GradientOptimizableFunction{
		
		/**
		 * Sets the number of parallel processing blocks. This number should optimally equal to the number of available processors in the current machine. 
		 * Default value should be 1. 
		 * @param number
		 */
		Grid1D m_oldGuess = new Grid1D(2*m_conf.getNumberOfProjections());
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
			for(int i = 0; i < x.length; i++){
				m_oldGuess.setAtIndex(i, (float)(x[i]) - m_oldGuess.getAtIndex(i)); 
			}
			
			setShiftVector(m_oldGuess);
			applyShift();
			doFFTAngle();
			double sum = 0;
			for(int proj = 0; proj < m_mask.getSize()[0]; proj++ ){
				for(int u = 0; u < m_mask.getSize()[1]; u++){
					if(m_mask.getAtIndex(proj, u) == 1){
						for(int v = 0; v < m_2dFourierTransposed.getSize()[2]; v++){
							sum += Math.pow(m_2dFourierTransposed.getAtIndex(proj, u, v).getMagn(),2.0);
						}
					}
				}
			}
			doiFFTAngle();
			return sum;
		}
		
		public double [] gradient(double[] x, int block){
			return null;
		}
		
	}
}



