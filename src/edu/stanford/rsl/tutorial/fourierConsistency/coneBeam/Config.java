/*
 /*
 * Copyright (C) 2015 Wolfgang Aichinger, Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.fourierConsistency.coneBeam;

import edu.stanford.rsl.conrad.data.generic.complex.ComplexGrid2D;
import edu.stanford.rsl.conrad.data.generic.complex.Fourier;
import edu.stanford.rsl.conrad.data.generic.datatypes.Complex;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

public class Config {

	// scaling factor indicating smaller data than original for testing purposes
	private int m_scalingFactor;
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
	// spacing in spatial domain
	private double spacingX;
	private double spacingY;
	private double angleInc;

	//spacing in frequency domain
	private double wuSpacing;
	private double wvSpacing;
	private double kSpacing;

	// spacing arrays in frequency space
	private Grid1D wuSpacingVec;
	private Grid1D wvSpacingVec;
	private Grid1D kSpacingVec;

	// precomputed arrays to perform shift in frequencydomain
	private Grid1D shiftFreqX;
	private Grid1D shiftFreqY;
	private Grid1D shiftFreqP;

	// Size of mask used for erosion
	private int m_erosionFactor;
	// contains a one for all positions which should be zero in ideal fouriertransformed sinogram
	private Grid2D mask;

	// matrices to perform a dft and idft for use on graphicscard
	private ComplexGrid2D dftMatrix;
	private ComplexGrid2D idftMatrix;

	private Integer numberOfIterations;



	public Config(String xmlFilename, int erosionFactor, int scalingFactor, double maxObjectRadius, int nrOfIterations){
		/*int[] dims = data.getSize();
		N = dims[0];
		M = dims[1];
		K = dims[2];*/

		//hard coded
		numberOfIterations = nrOfIterations;
		m_erosionFactor = erosionFactor;
		m_scalingFactor = scalingFactor;
		rp = maxObjectRadius;
		getGeometry(xmlFilename);
		wuSpacing = 1.0/(N*spacingX);
		wvSpacing = 1.0/(M*spacingY);
		kSpacing = 1.0/(K*angleInc);
		wuSpacingVec = createFrequArray(N, (float)(wuSpacing));
		wvSpacingVec = createFrequArray(M, (float)(wvSpacing));
		kSpacingVec = createFrequArray(K, (float)(kSpacing));

		// construct a shift vector
		shiftFreqX = constructShiftFreq(wuSpacingVec);
		shiftFreqY = constructShiftFreq(wvSpacingVec);
		shiftFreqP = constructShiftFreq(kSpacingVec);
		fillMask();
		createDFTMatrix();
		createIDFTMatrix();
		//		createShift(0.0f);


	}
	/**
	 * loads all data from a configfile, uses scalingfactor if original data is smaller for testing purpose
	 * @param filename
	 */
	private void getGeometry(String filename){
		Configuration config = Configuration.loadConfiguration(filename);
		Trajectory geom = config.getGeometry();
		// convert angle to radians
		angleInc = geom.getAverageAngularIncrement()*Math.PI/180.0 * m_scalingFactor;

		//TODO return back to regular, factor 2 only test with smaller database
		spacingX = geom.getPixelDimensionX() * m_scalingFactor;
		spacingY = geom.getPixelDimensionY() * m_scalingFactor;

		N = geom.getDetectorWidth()/m_scalingFactor;
		M = geom.getDetectorHeight()/m_scalingFactor;
		K = geom.getProjectionStackSize()/m_scalingFactor;


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
	public Grid1D getShiftFreqP(){
		return shiftFreqP;
	}
	public Grid2D getMask(){
		return mask;
	}

	public ComplexGrid2D getDFTMatrix(){
		return dftMatrix;
	}
	public ComplexGrid2D getIDFTMatrix(){
		return idftMatrix;
	}

	/**
	 * 
	 * @param dim: size of array
	 * @param freqSpacing: the spacing
	 * @return equally spaced array using freqSpacing 
	 */
	private Grid1D createFrequArray(int dim, float freqSpacing){
		Grid1D frequArray = new Grid1D(dim);
		for(int i = 0; i <= (dim -1)/2; i++){
			frequArray.setAtIndex(i, i*freqSpacing);
		}
		int i = -dim/2;
		for(int pos = dim/2; pos < dim; pos++, i++ ){
			frequArray.setAtIndex(pos, i*freqSpacing);
		}
		return frequArray;
	}



	private Grid1D constructShiftFreq(Grid1D spacingVec){
		int dim = spacingVec.getSize()[0];
		Grid1D frequencies = new Grid1D(dim);
		for(int i = 0; i < frequencies.getSize()[0]; i++){
			frequencies.setAtIndex(i, (float)(-2*Math.PI*spacingVec.getAtIndex(i)));
		}
		return frequencies;
	}

	/**
	 * computes ideal mask according to equation (paper M. Berger), than filtering with erosion mask
	 */
	private void fillMask(){
		mask = new Grid2D(K,N);
		Grid2D helpMask = new Grid2D(K,N);

		boolean preserveSymmetry = false;

		// filling mask according to formula
		int counter = 0;
		if(preserveSymmetry){
			for(int proj = 0; proj < K/2+1; proj++){
				for(int uPixel = 0; uPixel < N; uPixel++){
					if(Math.abs(kSpacingVec.getAtIndex(proj)/(kSpacingVec.getAtIndex(proj)-wuSpacingVec.getAtIndex(uPixel)*(L+D))) > rp/(float)(L)){
						if((proj == 0 && uPixel == 0) || (proj == K/2 && uPixel == 0) 
								|| (proj == 0 && uPixel == N/2) || (proj == K/2 && uPixel == N/2)){
							helpMask.setAtIndex(proj, uPixel, 0);
							counter++;
						}
						else{
							helpMask.setAtIndex(proj, uPixel, 1);
							counter++;
							if (proj == 0){
								helpMask.setAtIndex(proj, N - uPixel, 1);
								counter++;
							}
							else if (uPixel == 0){
								helpMask.setAtIndex(K - proj, uPixel, 1);
								counter++;
							}
							else{
								helpMask.setAtIndex(K - proj, N - uPixel, 1);
								counter++;
							}
						}
					}
				}
			}
		}
		else{
			for(int proj = 0; proj < K; proj++){
				for(int uPixel = 0; uPixel < N; uPixel++){
					if(Math.abs(kSpacingVec.getAtIndex(proj)/(kSpacingVec.getAtIndex(proj)-wuSpacingVec.getAtIndex(uPixel)*(L+D))) > rp/(float)(L)){
						helpMask.setAtIndex(proj, uPixel, 1);
						counter++;
					}
				}
			}
		}

		System.out.println("Anzahl 1 in Maske " + counter);

		// using erosion
		if (m_erosionFactor > 0){
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
		else {
			NumericPointwiseOperators.copy(mask, helpMask);
		}
	}


	private void createDFTMatrix(){
		dftMatrix = new ComplexGrid2D(K,K);
		float angle = (float)(-2*Math.PI/K);
		double normalFactor = 1.0f;//Math.sqrt(K);
		// dft matrix is symmetric which allows simplifications
		for(int horz = 0; horz < K; horz++){
			for(int vert = 0; vert <= horz; vert++){
				float tmpAngle = angle*horz*vert;
				Complex tmp = Complex.fromPolar(1.0f, tmpAngle);
				tmp = tmp.mul(normalFactor);
				dftMatrix.setAtIndex(horz, vert, tmp);
				dftMatrix.setAtIndex(vert, horz, tmp);
			}
		}
	}
	private void createIDFTMatrix(){
		idftMatrix = new ComplexGrid2D(K,K);
		float angle = (float)(2*Math.PI/K);
		double normalFactor = 1.0;///Math.sqrt(K);
		// dft matrix is symmetric which allows simplifications
		for(int horz = 0; horz < K; horz++){
			for(int vert = 0; vert <= horz; vert++){

				float tmpAngle = angle*horz*vert;
				Complex tmp = Complex.fromPolar(1.0f, tmpAngle);
				tmp = tmp.mul(normalFactor);
				idftMatrix.setAtIndex(horz, vert, tmp);
				idftMatrix.setAtIndex(vert, horz, tmp);
			}
		}
	}
	public Integer getNumberOfIterations() {
		return this.numberOfIterations;
	}



}
