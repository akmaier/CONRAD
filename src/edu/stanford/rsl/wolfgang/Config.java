package edu.stanford.rsl.wolfgang;

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
	
	// spacing arrays in frequency space
	private double[] wuSpacing;
	private double[] wvSpacing;
	private double[] kSpacing;
	
	public Config(String xmlFilename){
		/*int[] dims = data.getSize();
		N = dims[0];
		M = dims[1];
		K = dims[2];*/
		
		//hard coded
		rp = 125.0f;
		getGeometry(xmlFilename);
		wuSpacing = createFrequArray(N, spacingX);
		wvSpacing = createFrequArray(M, spacingY);
		kSpacing = createFrequArray(K, angleInc);
		
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
	
	private double[] createFrequArray(int dim, double spacing){
		double[] frequArray = new double[dim];
		for(int i = 0; i < (dim -1)/2; i++){
			frequArray[i] = i*spacing;
		}
		int i = -dim/2;
		for(int pos = dim/2; pos < dim; pos++, i++ ){
			frequArray[pos] = i*spacing;
		}
		return frequArray;
	}
	

}
