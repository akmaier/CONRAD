package edu.stanford.rsl.wolfgang;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

public class Config {
	
	// dimensions projections
	private int N; // horizontal
	private int M; // vertical
	private int K; // angles
	
	// source to patient
	private double L;
	// detector to patient
	private double D;
	// max object distance to center of rotation
	private double rp;
	
	private double spacingX;
	private double spacingY;
	private double angleInc;
	
	private double wuSpacing;
	private double wvSpacing;
	private double kSpacing;
	
	// spacing arrays in frequency space
	private Grid1D wuSpacingVec;
	private Grid1D wvSpacingVec;
	private Grid1D kSpacingVec;
	
	
	private Grid1D shiftFreqX;
	private Grid1D shiftFreqY;
	
	
	// translation-grid
	private int[] translation;
	
	private int m_erosionFactor;
	private Grid2D mask;

	
	public Config(String xmlFilename, int erosionFactor){
		/*int[] dims = data.getSize();
		N = dims[0];
		M = dims[1];
		K = dims[2];*/
		
		//hard coded
		m_erosionFactor = erosionFactor;
		rp = 125.0f;
		getGeometry(xmlFilename);
		wuSpacing = 1.0/(N*spacingX);
		wvSpacing = 1.0/(M*spacingY);
		kSpacing = 1.0/(K*angleInc);
		wuSpacingVec = createFrequArray(N, (float)(wuSpacing));
		wvSpacingVec = createFrequArray(M, (float)(wvSpacing));
		kSpacingVec = createFrequArray(K, (float)(kSpacing));
		
		// construct a shift vector
		shiftFreqX = constructShiftFreq(wuSpacingVec, spacingX);
		shiftFreqY = constructShiftFreq(wvSpacingVec, spacingY);
		fillMask();
//		createShift(0.0f);
		
			
	}
	
	private void getGeometry(String filename){
		Configuration config = Configuration.loadConfiguration(filename);
		Trajectory geom = config.getGeometry();
		// convert angle to radians
		angleInc = geom.getAverageAngularIncrement()*Math.PI/180.0;
		spacingX = geom.getPixelDimensionX();
		spacingY = geom.getPixelDimensionY();
		
		N = geom.getDetectorWidth();
		M = geom.getDetectorHeight();
		K = geom.getProjectionStackSize();
		
		
		D = geom.getSourceToDetectorDistance() - geom.getSourceToAxisDistance();
		L = geom.getSourceToAxisDistance();
		
		
		
	}
	public double getSourceToPatientDist(){
		return L;
	}
	public double getDetectorToPatientDist(){
		return D;
	}
	public double getMaxObjectToDetectorDist(){
		return rp;
	}
	
	// as this parameter is approximated at the moment it is adjustable from outside
	public void setMaxObjectToDetectorDist(double dist){
		if(dist < 0){
			return;
		}
		rp = dist;
	}
	public int getHorizontalDim(){
		return N;
	}
	public int getVerticalDim(){
		return M;
	}
	public int getNumberOfProjections(){
		return K;
	}
	public double getPixelXSpace(){
		return spacingX;
	}
	public double getPixelYSpace(){
		return spacingY;
	}
	public double getAngleIncrement(){
		return angleInc;
	}
	public double getUSpacing(){
		return wuSpacing;
	}
	public double getVSpacing(){
		return wvSpacing;
	}
	public double getKSpacing(){
		return kSpacing;
	}
	public Grid1D getUSpacingVec(){
		return wuSpacingVec;
	}
	public Grid1D getVSpacingVec(){
		return wvSpacingVec;
	}
	public Grid1D getKSpacingVec(){
		return kSpacingVec;
	}
	public Grid1D getShiftFreqX(){
		return shiftFreqX;
	}
	public Grid1D getShiftFreqY(){
		return shiftFreqY;
	}
	public Grid2D getMask(){
		return mask;
	}
	
	
	private Grid1D createFrequArray(int dim, float freqSpacing){
		Grid1D frequArray = new Grid1D(dim);
//		freqSpacing = 1.0f;//(dim*spacing);
		for(int i = 0; i <= (dim -1)/2; i++){
			frequArray.setAtIndex(i, i*freqSpacing);
		}
		int i = -dim/2;
		for(int pos = dim/2; pos < dim; pos++, i++ ){
			frequArray.setAtIndex(pos, i*freqSpacing);
		}
		return frequArray;
	}
	
	
	
	public void setTrans(int[] t){
		translation = t;
	}
	
	private Grid1D constructShiftFreq(Grid1D spacingVec, double spacingLocal){
		int dim = spacingVec.getSize()[0];
		Grid1D frequencies = new Grid1D(dim);
		for(int i = 0; i < frequencies.getSize()[0]; i++){
			frequencies.setAtIndex(i, (float)(-2*Math.PI*spacingVec.getAtIndex(i)*spacingLocal/*/(dim*spacingLocal)*/));
		}
		return frequencies;
	}
	
	private void fillMask(){
		mask = new Grid2D(K,N);
		Grid2D helpMask = new Grid2D(K,N);
		// filling mask according to formula
		for(int proj = 0; proj < helpMask.getSize()[0]; proj++){
			for(int uPixel = 0; uPixel < helpMask.getSize()[1]; uPixel++){
				if(Math.abs(kSpacingVec.getAtIndex(proj)/(kSpacingVec.getAtIndex(proj)-wuSpacingVec.getAtIndex(uPixel)*(L+D))) > rp/(float)(L)){
					helpMask.setAtIndex(proj, uPixel, 1);
				}
			}
			
		}
		// using erosion
		int shift = (int)(m_erosionFactor/2);
		for(int proj = 0; proj < mask.getSize()[0]; proj++){
			for(int uPixel = 0; uPixel < helpMask.getSize()[1]; uPixel++){
				boolean foundZero = false;
				
				for(int horiErosion = proj-shift; horiErosion <= proj+shift; horiErosion++){
					int horiErosionValid =  horiErosion;
					if (horiErosion < 0){
						horiErosionValid = K+horiErosion;
					}
					if(horiErosion >= K){
						horiErosionValid = horiErosion - K;
					}
					
					// we are at a boundary ((N-1)/2 and -(N/2))
					
					if(Math.abs(kSpacingVec.getAtIndex(proj) - kSpacingVec.getAtIndex(horiErosionValid))> (shift+1)*kSpacing){
						continue;
					}
					for(int vertErosion = uPixel-shift; vertErosion <= uPixel+shift; vertErosion++){
						int vertErosionValid = vertErosion;
						if(vertErosion < 0){
							vertErosionValid = N+vertErosion;
						}
						if(vertErosion >= N){
							vertErosionValid = vertErosion - N;
						}
						if(Math.abs(wuSpacingVec.getAtIndex(uPixel) - wuSpacingVec.getAtIndex(vertErosionValid))> (shift+1)*wuSpacing){
							continue;
						}
						
						if (helpMask.getAtIndex(horiErosionValid, vertErosionValid) == 0){
							foundZero = true;
							break;
						}
						
					}
					if(foundZero){
						break;
					}
					
				}
				if(!foundZero){
					mask.setAtIndex(proj, uPixel, 1);
				}
				foundZero = false;
				
			}
		}
	}
//	private void createShift(float shift){
//		frequencyShift = new ComplexGrid1D(frequencies.length);
//		for(int i = 0; i < frequencyShift.getSize()[0]; i++){
//			float angle =  (float)(frequencies[i]*Math.PI/2);
//			frequencyShift.setRealAtIndex((float)(Math.cos(angle)), i);
//			frequencyShift.setImagAtIndex((float)(Math.sin(angle)), i);
//			
//		}
//	}
	
	

}
