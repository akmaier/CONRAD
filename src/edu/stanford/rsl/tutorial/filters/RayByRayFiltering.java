/**
 * 
 */
package edu.stanford.rsl.tutorial.filters;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * @author Salaheldin Saleh. me@s-saleh.com
 * The implementation of this filter is based on the paper:" A 
 * filtered backprojection algorithm with ray-by-ray noise weighting"
 * The idea of the ray by ray filter is to combine the highpass
 * ramp filter with a lowpass prefilter whose strength is 
 * determined by the noise model. This would help in removing 
 * the noise while preserving the edges. The algorithms suggests 
 * implementing 11 versions of lowpass prefilters that are combined
 * with highpass ramp filter. For each ray in the projections, only
 * one of those 11 filters is selected. As previously mentioned this
 * will depend on the strength of the noise. 
 */
public class RayByRayFiltering implements GridKernel{


	/**
	 * Both parameter k and parameter β can be used for noise
	 * regularization. A smaller k or a larger β can blur the image
	 * more. It is not clear how these two parameters interact with
	 * each other.
	 */
	// corresponds to the step-size in an iterative algorithm
	private double alpha =  0.5;
	// The contributing factor of the regularization term in the objective function. 
	// Should be very small E-3 to E-6 for good smoothing
	private double beta =  0.000200;
	// The filter function in the regularization term
	private double R =  1.;
	// Corresponds to the iteration number in iterative algorithms.
	private double k =  1000000; 
	// The maximum projection value. Should be greater than 1 and less than 7-8 
	// for convenient lowpass filters. Actual sinogram's maximum value should be
	// normalized to this one. Sometimes when using fake phantoms, the
	// sinogram maximum is much larger than pmax value. In this case set 
	// sinoMaxOrg to the original sinogram max value for auto normalization.
	private int pmax = 6;
	// The actual maximum of the sinogram. 
	private int sinoMaxOrg = 6;
	// Used for to get the omega coeff
	RamLakKernel ramLak;
	// The number of filters
	private final int numF = 11;
	// Create numF filters
	Grid1DComplex[] Hn  = new Grid1DComplex[numF];

	// Constructor with default ramlak filter values
	public RayByRayFiltering(final int size, double deltaS) {
		ramLak = new RamLakKernel(size, deltaS);
		generateRayFilters();
	}

	// Constructor with the ability to assign different ramLak values
	public RayByRayFiltering(RamLakKernel newRamlak) {
		ramLak = newRamlak;
		generateRayFilters();
	}

	// Constructor with ability to choose filter's parameters
	public RayByRayFiltering(final int size, double deltaS, double alpha, 
			double beta, double r, double k, int pmax, int sinoMaxOrg) {
		this(size, deltaS); // Call the default constructor
		this.alpha = alpha;
		this.beta = beta;
		R = r;
		this.k = k;
		this.pmax = pmax;
		this.sinoMaxOrg = sinoMaxOrg;
		generateRayFilters();
	}

	// Constructor with ability to choose filter's parameters and other ramlak coeff
	public RayByRayFiltering(RamLakKernel newRamlak, double alpha, 
			double beta, double r, double k, int pmax, int sinoMaxOrg) {
		ramLak = newRamlak; // Call the default constructor
		this.alpha = alpha;
		this.beta = beta;
		R = r;
		this.k = k;
		this.pmax = pmax;
		this.sinoMaxOrg = sinoMaxOrg;
		generateRayFilters();
	}


	// Generate the coeff of the numF RbR filters
	private void generateRayFilters() {
		// For each of the numF filters
		for(int i=0; i<numF; i++) 
		{	
			// Initialize the filter size
			Hn[i] = new Grid1DComplex(ramLak.getSize()[0], false);
			// Inititialize Wn. The exp coeff can be between 0.1-0.3
			// 0.1 is the standard value for noise modeling but it 
			// introduces more artifacts.
			float Wn = (float) Math.exp(-0.2 * i * pmax) ;
			// On each Coeff
			for(int j=0; j<Hn[i].getSize()[0]; j ++) 
			{
				double absW = (ramLak.getRealAtIndex(j)); // Get the ramLak coeff
				double aWnw = (alpha * Wn / absW); // Computes aWn/|w|
				double abr = alpha * beta * R; // Compute axbxr
				double innerpart = 1 -aWnw - abr;
				double pK = Math.pow((innerpart), k); if((-1>pK) || (pK>1)) pK=0; // set pK to zero if out of range
				double brww1 = (double) (1. + beta * absW / Wn);  
				// Set the filters coeff
				Hn[i].setRealAtIndex(j,(float) ( (1. - pK) * absW / brww1  ));//1. / brww1 * absW 
				Hn[i].setImagAtIndex(j,(float) ( 0 ));
			}
		}	
	}
	/**
	 * Show the filters
	 */
	public void showFilters(){
		for(int i=0; i<numF; i++){
			Hn[i].show();
		}
	}

	/**
	 * @return the ramLak
	 */
	public RamLakKernel getRamLak() {
		return ramLak;
	}

	/**
	 * @param ramLak the ramLak to set
	 */
	public void setRamLak(RamLakKernel ramLak) {
		this.ramLak = ramLak;
	}

	/**
	 * @return the hn
	 */
	public Grid1DComplex[] getHn() {
		return Hn;
	}

	/**
	 * @param hn the hn to set
	 */
	public void setHn(Grid1DComplex[] hn) {
		Hn = hn;
	}

	/**
	 * @return the pmax
	 */
	public double getPmax() {
		return pmax;
	}

	/**
	 * @param pmax the pmax to set
	 */
	public void setPmax(int pmax) {
		this.pmax = pmax;
	}

	/**
	 * @return the sinoMaxOrg
	 */
	public int getSinoMaxOrg() {
		return sinoMaxOrg;
	}

	/**
	 * @param sinoMaxOrg the sinoMaxOrg to set
	 */
	public void setSinoMaxOrg(int sinoMaxOrg) {
		this.sinoMaxOrg = sinoMaxOrg;
	}

	/**
	 * @return the alpha
	 */
	public double getAlpha() {
		return alpha;
	}

	/**
	 * @param alpha the alpha to set
	 */
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * @return the beta
	 */
	public double getBeta() {
		return beta;
	}

	/**
	 * @param beta the beta to set
	 */
	public void setBeta(double beta) {
		this.beta = beta;
	}

	/**
	 * @return the r
	 */
	public double getR() {
		return R;
	}

	/**
	 * @param r the r to set
	 */
	public void setR(double r) {
		R = r;
	}

	/**
	 * @return the k
	 */
	public double getK() {
		return k;
	}

	/**
	 * @param k the k to set
	 */
	public void setK(double k) {
		this.k = k;
	}


	public String getBibtexCitation() {
		String bibtex = "@BOOK{zeng2013filtered,\n" +
				"  author = {Zeng, Gengsheng L and Zamyatin, Alex},\n" +
				"  title = {{A filtered backprojection algorithm with ray-by-ray noise weighting}},\n" +
				"  journal = {Medical physics},\n" +
				"  volume = {40},\n" +
				"  number = {3},\n" +
				"  pages = {031113},\n" +
				"  publisher = {American Association of Physicists in Medicine},\n" +
				"  year = {2013}\n" +
				"}";
		return bibtex;
	}

	public String getMedlineCitation() {
		return "Zeng, Gengsheng L., and Alex Zamyatin. A filtered backprojection algorithm with ray-by-ray noise weighting. Medical physics 40.3 (2013): 031113.";
	}

	/**
	 * Used to correct for the over-sampling of the projection center in Fourier domain by back projection
	 * Not device dependent. 
	 */
	public boolean isDeviceDependent() {
		return false;
	}


	public String getToolName() {
		return "Ray by ray filtering. Applied in the frequency domain.";
	}

	// Apply the filters to the grid
	public void applyToGrid(Grid1D sino) {

		// Will hold the filtered sinogram
		Grid1DComplex[] FsinoCPLX = new Grid1DComplex [numF];
		for(int i=0; i<numF; i++) // For each of the 11 filters
		{	
			FsinoCPLX[i] = new Grid1DComplex(sino);
		}

		for (int i =0; i < numF;  i ++) { // go on the 11 version
			// fft of the sinogram
			FsinoCPLX[i].transformForward();
			for(int j =0; j < FsinoCPLX[i].getSize()[0];  j ++){ // go on every ray
				// Apply the numF filters
				FsinoCPLX[i].multiplyAtIndex(j, Hn[i].getRealAtIndex(j), Hn[i].getImagAtIndex(j));
			}
			// ifft for the filtered projections
			FsinoCPLX[i].transformInverse();
		}

		// Select the right filter to apply for each ray
		// on every ray
		for(int j =0; j < sino.getSize()[0];  j ++){ 
			// on the numF filter
			for(int i =0; i<numF; i++){ 
				// Check the range
				if(sino.getAtIndex(j)/(sinoMaxOrg/pmax)<0.5*((0.1 * i * pmax)+(0.1 * (i+1) * pmax))){
					sino.setAtIndex(j, FsinoCPLX[i].getRealAtIndex(j));
					break;
				}
				if(i == 10){ // If non selected, select 10
					sino.setAtIndex(j, FsinoCPLX[10].getRealAtIndex(j));
				}
			}

		}


	}

}
/*
 * Copyright (C) 2010-2014 Salaheldin Saleh, me@s-saleh.com
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
