package edu.stanford.rsl.wolfgang;

import java.io.IOException;
import java.nio.FloatBuffer;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
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
	
	// opencl members
	
	// are initialized after transposing the data
	private OpenCLGrid3D realPart;
	private OpenCLGrid3D imagPart;
	private OpenCLGrid1D freqU;
	private OpenCLGrid1D freqV;
	
	public MovementCorrection3D(Grid3D data, Config conf){
		m_conf = conf;
		m_data = new ComplexGrid3D(data);
		m_shift = new Grid1D(2*conf.getNumberOfProjections());
		m_mask = conf.getMask();
		freqU = new OpenCLGrid1D(conf.getShiftFreqX());
		freqV = new OpenCLGrid1D(conf.getShiftFreqY());
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
		
		realPart = new OpenCLGrid3D(m_2dFourierTransposed.getRealGrid());
		imagPart = new OpenCLGrid3D(m_2dFourierTransposed.getImagGrid());
		
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
		long time1 = System.currentTimeMillis();
		long timeComplete = 0;
//		OpenCLGrid3D realPart = new OpenCLGrid3D(m_2dFourierTransposed.getRealGrid());
//		OpenCLGrid3D imagPart = new OpenCLGrid3D(m_2dFourierTransposed.getImagGrid());
//		OpenCLGrid1D freqU = new OpenCLGrid1D(m_conf.getShiftFreqX());
//		OpenCLGrid1D freqV = new OpenCLGrid1D(m_conf.getShiftFreqY());
		OpenCLGrid1D shifts = new OpenCLGrid1D(m_shift);
		
		long time2  = System.currentTimeMillis();
		long timeDiff = time2 - time1;
		time1= time2;
		System.out.println("Step 1: " + timeDiff);
		timeComplete += timeDiff;
		// read and write buffers
		CLBuffer<FloatBuffer> bufferRealPart = realPart.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferImagPart = imagPart.getDelegate().getCLBuffer();
		
		// only read buffer
		CLBuffer<FloatBuffer> bufferFreqU = freqU.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferFreqV = freqV.getDelegate().getCLBuffer();
		CLBuffer<FloatBuffer> bufferShifts = shifts.getDelegate().getCLBuffer();
		
		time2  = System.currentTimeMillis();
		timeDiff = time2 - time1;
		System.out.println("Step 2: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		
		realPart.getDelegate().prepareForDeviceOperation();
		imagPart.getDelegate().prepareForDeviceOperation();
		
		freqU.getDelegate().prepareForDeviceOperation();
		freqV.getDelegate().prepareForDeviceOperation();
		shifts.getDelegate().prepareForDeviceOperation();
		
		time2  = System.currentTimeMillis();
		timeDiff = time2 -time1;
		System.out.println("Step 3: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		
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

		int localWorksize = 10;
		long globalWorksizeAngle = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[0]);
		long globalWorksizeU = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[1]);
		long globalWorksizeV = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[2]);

		time2  = System.currentTimeMillis();
		timeDiff = time2 -time1;
		System.out.println("Step 4: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		
		CLCommandQueue commandQueue = device.createCommandQueue();
		commandQueue.put3DRangeKernel(kernel, 0, 0, 0, globalWorksizeAngle, globalWorksizeU, globalWorksizeV, localWorksize, localWorksize, localWorksize).finish();
		
		realPart.getDelegate().notifyDeviceChange();
		imagPart.getDelegate().notifyDeviceChange();
		
		time2  = System.currentTimeMillis();
		timeDiff = time2 -time1;
		System.out.println("Step 5: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		
		Grid3D realCPU = new Grid3D(realPart);
		Grid3D imagCPU = new Grid3D(imagPart);
		
		time2  = System.currentTimeMillis();
		timeDiff = time2 -time1;
		System.out.println("Step 6: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		
		for(int i = 0; i < shifts.getSize()[0]; i = i+2){
			if(shifts.getAtIndex(i) != 0 || shifts.getAtIndex(i+1) != 0){
				for(int detCol = 0; detCol < m_2dFourierTransposed.getSize()[1]; detCol++){
					for(int detRow = 0; detRow < m_2dFourierTransposed.getSize()[2]; detRow++){
						m_2dFourierTransposed.setAtIndex(i/2, detCol, detRow, realCPU.getAtIndex(i/2, detCol, detRow), imagCPU.getAtIndex(i/2, detCol, detRow));
					}
				}
			}
		}
//		m_2dFourierTransposed = new ComplexGrid3D(realCPU, imagCPU);
//		m_2dFourierTransposed.setOrigin(0,0,0);
//		m_2dFourierTransposed.setSpacing(m_conf.getAngleIncrement() ,m_conf.getUSpacing(), m_conf.getVSpacing());
		
		time2  = System.currentTimeMillis();
		timeDiff = time2 -time1;
		System.out.println("Step 7: " + timeDiff);
		time1 = time2;
		timeComplete += timeDiff;
		System.out.println("Complete Time:" + timeComplete);
	}
	
	public void parallelizedApplyShift2D(){
		long time1 = System.currentTimeMillis();
		long timeComplete = time1;

		for(int shiftc = 0; shiftc < m_shift.getNumberOfElements(); shiftc = shiftc + 2){
			if(m_shift.getAtIndex(shiftc) != 0 || m_shift.getAtIndex(shiftc +1) != 0 ){
				
				Grid2D realSlice2Shift = new Grid2D(m_2dFourierTransposed.getSize()[1], m_2dFourierTransposed.getSize()[2]);
				Grid2D imagSlice2Shift = new Grid2D(m_2dFourierTransposed.getSize()[1], m_2dFourierTransposed.getSize()[2]);
				for(int detCol = 0; detCol < realSlice2Shift.getSize()[0]; detCol++){
					for(int detRow = 0; detRow < realSlice2Shift.getSize()[1]; detRow++){
						Complex temp = m_2dFourierTransposed.getAtIndex(shiftc/2, detCol, detRow);
						realSlice2Shift.setAtIndex(detCol, detRow, (float)(temp.getReal()));
						imagSlice2Shift.setAtIndex(detCol, detRow, (float)(temp.getImag()));
					}
				}
				

				OpenCLGrid2D realSliceCL = new OpenCLGrid2D(realSlice2Shift);
				OpenCLGrid2D imagSliceCL = new OpenCLGrid2D(imagSlice2Shift);
				
				long time2 = System.currentTimeMillis();
				long timeDiff = time2 - time1;
				System.out.println("Time to get to graphics card" + timeDiff);
				time1 = time2;
				
				
				CLBuffer<FloatBuffer> bufferRealPart = realSliceCL.getDelegate().getCLBuffer();
				CLBuffer<FloatBuffer> bufferImagPart = imagSliceCL.getDelegate().getCLBuffer();
				
				CLBuffer<FloatBuffer> bufferFreqU = freqU.getDelegate().getCLBuffer();
				CLBuffer<FloatBuffer> bufferFreqV = freqV.getDelegate().getCLBuffer();
				
				realSliceCL.getDelegate().prepareForDeviceOperation();
				imagSliceCL.getDelegate().prepareForDeviceOperation();
				
				freqU.getDelegate().prepareForDeviceOperation();
				freqV.getDelegate().prepareForDeviceOperation();
				
				// create Context
				CLContext context = OpenCLUtil.getStaticContext();
				// choose fastest device
				CLDevice device = context.getMaxFlopsDevice();
				CLProgram program = null;
				try {
				program = context.createProgram(MovementCorrection3D.class.getResourceAsStream("shiftInFourierSpace2D.cl"));
				} catch (IOException e) {
				e.printStackTrace();
				}
				program.build();
				
				CLKernel kernel = program.createCLKernel("shiftInFourierSpace2D");
				
				kernel.putArg(bufferRealPart);
				kernel.putArg(bufferImagPart);
				kernel.putArg(bufferFreqU);
				kernel.putArg(bufferFreqV);
				kernel.putArg(m_shift.getAtIndex(shiftc));
				kernel.putArg(m_shift.getAtIndex(shiftc + 1));
				kernel.putArg(m_2dFourierTransposed.getSize()[1]);
				kernel.putArg(m_2dFourierTransposed.getSize()[2]);
				
				int localWorksize = 16;
				long globalWorksizeU = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[1]);
				long globalWorksizeV = OpenCLUtil.roundUp(localWorksize, m_2dFourierTransposed.getSize()[2]);
				
				CLCommandQueue commandQueue = device.createCommandQueue();
				commandQueue.put2DRangeKernel(kernel, 0, 0, globalWorksizeU, globalWorksizeV, localWorksize, localWorksize).finish();
				
				realSliceCL.getDelegate().notifyDeviceChange();
				imagSliceCL.getDelegate().notifyDeviceChange();
				
				time2 = System.currentTimeMillis();
				timeDiff = time2 - time1;
				System.out.println("time on graphics card" + timeDiff);
				time1 = time2;
				
				for(int detCol = 0; detCol < m_2dFourierTransposed.getSize()[1]; detCol++){
					for(int detRow = 0; detRow < m_2dFourierTransposed.getSize()[2]; detRow++){
						m_2dFourierTransposed.setAtIndex(shiftc/2, detCol, detRow, realSliceCL.getAtIndex(detCol, detRow), imagSliceCL.getAtIndex(detCol, detRow));
					}
				}
				
				time2 = System.currentTimeMillis();
				timeDiff = time2 - time1;
				System.out.println("time to get back" + timeDiff);
				
			}
		}
		timeComplete = System.currentTimeMillis() - timeComplete;
		System.out.println("time for complete shift: " + timeComplete);
		
		
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
	
	public Grid1D computeOptimalShift(){
		EnergyToBeMinimized function = new EnergyToBeMinimized();
		FunctionOptimizer fo = new FunctionOptimizer();
		fo.setDimension(2*m_conf.getNumberOfProjections());
		fo.setOptimizationMode(OptimizationMode.Function);
		fo.setConsoleOutput(true);
		double[] optimalShift = null;
		double min = Double.MAX_VALUE;
		for(int i = 0; i < 5; i++){
			System.out.println("Optimizer: " + i);
			double[] initialGuess = new double[2*m_conf.getNumberOfProjections()];
			fo.setInitialX(initialGuess);
			double [] result = fo.optimizeFunction(function);
			double newVal = function.evaluate(result, 0);
			if ( newVal < min) {
				optimalShift = result;
				min = newVal;
			}
		}
		
		Grid1D optimalShiftGrid = new Grid1D(optimalShift.length);
		for(int i = 0; i < optimalShift.length; i++){
			optimalShiftGrid.setAtIndex(i, (float)(optimalShift[i]));
		}
			
		return optimalShiftGrid;
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
			parallelizedApplyShift2D();
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



