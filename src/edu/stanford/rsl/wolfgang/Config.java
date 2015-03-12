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
	
	
	private float[] shiftFreqX;
	private float[] shiftFreqY;
	
	
	// translation-grid
	private int[] translation;
	private Grid2D mask;

	
	public Config(String xmlFilename){
		/*int[] dims = data.getSize();
		N = dims[0];
		M = dims[1];
		K = dims[2];*/
		
		//hard coded
		rp = 125.0f;
		getGeometry(xmlFilename);
		wuSpacing = 1.0/(N*spacingX);
		wvSpacing = 1.0/(M*spacingY);
		kSpacing = 1.0/(K*angleInc);
		wuSpacingVec = createFrequArray(N, (float)(wuSpacing));
		wvSpacingVec = createFrequArray(M, (float)(wvSpacing));
		kSpacingVec = createFrequArray(K, (float)(angleInc));
		
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
	public float[] getShiftFreqX(){
		return shiftFreqX;
	}
	public float[] getShiftFreqY(){
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
	
	private float[] constructShiftFreq(Grid1D spacingVec, double spacingLocal){
		int dim = spacingVec.getSize()[0];
		float[] frequencies = new float[dim];
		for(int i = 0; i < frequencies.length; i++){
			frequencies[i] = (float)(-2*Math.PI*spacingVec.getAtIndex(i)*spacingLocal/*/(dim*spacingLocal)*/);
		}
		return frequencies;
	}
	
	private void fillMask(){
		mask = new Grid2D(K,N);
		for(int proj = 0; proj < mask.getSize()[0]; proj++){
			float angle = kSpacingVec.getAtIndex(proj);
			float xi = (float)(angle*(1-L/rp)/(L+D));
			float value = (float)(angle/(angle-xi*(L+D)));
			//System.out.println("angle: "+ angle + ", xi: "+xi+ ", absolute: "+ value);
			if(angle/(angle-xi*(L+D)) > 0){
				if(xi >= 0){
					xi /= wuSpacing;
				}
				else{
					xi = (float)(xi/wuSpacing + N/2.0f);
				}
				System.out.println("angle: "+ angle + ", xi: "+xi+ ", absolute: "+ value);
				mask.setAtIndex(proj, (int)xi, 1.0f);
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
