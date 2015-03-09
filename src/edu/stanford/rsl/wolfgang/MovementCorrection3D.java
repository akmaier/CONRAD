package edu.stanford.rsl.wolfgang;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid3D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class MovementCorrection3D {

	/**
	 * @param args
	 */
	private Config m_conf;
	private ComplexGrid3D m_data;
	private ComplexGrid3D m_2dFourierTransposed = null;
	private float[] m_shift;
	public MovementCorrection3D(Grid3D data, Config conf){
		m_conf = conf;
		m_data = new ComplexGrid3D(data);
		m_shift = new float[2*conf.getNumberOfProjections()];
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
		float[] shiftFreqX = m_conf.getShiftFreqX();
		float[] shiftFreqY = m_conf.getShiftFreqY();
		//ComplexGrid1D shiftComplexFreqX = m_conf.getComplexShiftFreqX();
		//ComplexGrid1D shiftComplexFreqY = m_conf.getComplexShiftFreqY();
		//float xShiftNormFactor = (float)(1.0f/(m_conf.getPixelXSpace()*m_conf.getUSpacing())); // = 640
		//float yShiftNormFactor = (float)(1.0f/(m_conf.getPixelYSpace()*m_conf.getVSpacing())); // = 480
		for(int angle = 0; angle < m_data.getSize()[2]; angle++){
			// get the shifts in both directions (in pixel)
			float shiftX = m_shift[angle*2];
			float shiftY = m_shift[angle*2+1];
			//System.out.println(angle+1 + " of "+ m_data.getSize()[2]);
			for(int u = 0; u < m_data.getSize()[0]; u++){
				//complex number representing shift in x-direction
				//Complex expShiftX = new Complex(Math.cos(shiftFreqX[u]*shiftX/*xShiftNormFactor/*+0.001*/),Math.sin(shiftFreqX[u]*shiftX/*xShiftNormFactor/*+0.001*/));			
				float angleX = shiftFreqX[u]*shiftX;
				//Complex expShiftX = shiftComplexFreqX.getAtIndex(u).power(shiftX);
				for(int v = 0; v < m_data.getSize()[1]; v++){
					// complex number representing shift in y-direction
					//Complex expShiftY = new Complex(Math.cos(shiftFreqY[v]*shiftY/*yShiftNormFactor /*+0.001*/),Math.sin(shiftFreqY[v]*shiftY/*yShiftNormFactor/*+0.001*/));
					
					/*test */
//					float angleY = shiftFreqY[v]*shiftY;
//					System.out.println("Angle x: "+angleX+", Angle y: "+angleY);
					float sumAngles = angleX + shiftFreqY[v]*shiftY;
					
					//Complex expShiftY = shiftComplexFreqY.getAtIndex(v).power(shiftY);
					// complex number representing both shifts
					//expShiftY = expShiftY.mul(expShiftX);		
					// multiply at position in complex grid
					Complex shift = getComplexFromAngles(sumAngles);
//					float newVal = m_data.getRealAtIndex(u, v, angle);
//					m_data.setAtIndex(u, v, angle, newVal*shift[0], newVal*shift[1]);//(u, v, angle, shift);
					m_data.multiplyAtIndex(u, v, angle, shift);
					
				}
			}
		}
		time = System.currentTimeMillis()-time;
		System.out.println("Time for complete shift:"+ time);
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
	public void setShiftVector(float[]shift){
		if(shift.length != 2* m_conf.getNumberOfProjections()){
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
}

