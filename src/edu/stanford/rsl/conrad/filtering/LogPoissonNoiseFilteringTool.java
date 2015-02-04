package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.StatisticsUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Applies Poisson noise to the input image as it would happen in the physical process (based on a monochromatic x-ray source).
 * The relation given the emitted photons by the source {@latex.inline $N_0$}, the observed photons {@latex.inline $N$}, and the absorption 
 * coefficients {@latex.inline $\\mu_i$} in {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^3}]$} 
 * along the ray segments {@latex.inline $x_i$} in {@latex.inline $[\\textnormal{cm}]$} is given by Lambert-Beer's law:<BR>
 * {@latex.inline $N = N_0 \\cdot e^{-\\sum_i\\mu_i \\cdot x_i}$}.<BR>
 * We assume that the input data is in log domain, i.e. given as 
 * {@latex.inline $\\sum_i\\mu_i \\cdot x_i$} in 
 * {@latex.inline $[\\frac{\\textnormal{g}}{\\textnormal{cm}^2}]$} 
 * as returned by a forward projection.<BR>
 * The data is converted to photon count domain
 * via application of the minus exponent transform.<BR> 
 * Photon noise statistics are computed according to a Poisson distribution<BR>
 * {@latex.inline $$P(\\mathcal{N} = n) = \\frac{\\lambda^n}{n!} \\cdot e^{-\\lambda}$$}<BR>
 * where {@latex.inline $\\lambda$} is assumed to be {@latex.inline $N$} for each pixel.
 * With the mean and standard deviation of the Poisson distribution being {@latex.inline $\\lambda$} and {@latex.inline $\\sqrt{\\lambda}$} respectively, the Signal-to-Noise-Ration (SNR) is <BR>
 * {@latex.inline $$\\textnormal{SNR} = \\frac{\\textnormal{Signal}}{\\textnormal{Noise}} = \\frac{\\lambda}{\\sqrt{\\lambda}} = \\sqrt{\\lambda} = \\sqrt{N} $$}.<BR>
 * Hence, the more photons were observed, the better the SNR for the respective pixel.
 * <BR><BR>
 * The user
 * has to specify a photon count {@latex.inline $N_0$} (e.g. 75000). This value will correspond to 0 in the log domain image.<br> 
 * Then the photon statistics are applied according to Poisson noise statistics.
 * In the final step the data is then transformed back to minus log domain to match the scaling of the original input data.
 * 
 * @see edu.stanford.rsl.conrad.cuda.CUDAForwardProjector
 * @author Andreas Maier
 *
 */
public class LogPoissonNoiseFilteringTool extends PoissonNoiseFilteringTool {

	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6025566244585199099L;
	/**
	 * 
	 */
	private double photonCountMax = 75000;
	
	/**
	 * @return the photonCountMax
	 */
	public double getPhotonCountMax() {
		return photonCountMax;
	}

	/**
	 * @param photonCountMax the photonCountMax to set
	 */
	public void setPhotonCountMax(double photonCountMax) {
		this.photonCountMax = photonCountMax;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		LogPoissonNoiseFilteringTool filter = new LogPoissonNoiseFilteringTool();
		filter.configured = configured;
		filter.photonCountMax = photonCountMax;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Poisson Noise in Log Domain";
	}

	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		Grid2D imp = new Grid2D(imageProcessor.getWidth(), imageProcessor.getHeight());
		for (int k = 0; k < imageProcessor.getWidth(); k++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				double scaled = imageProcessor.getPixelValue(k, j) * -1.0 ;
				double expdomain = Math.exp(scaled) * photonCountMax;
				double noiseAdded = StatisticsUtil.poissonRandomNumber(expdomain);
				double log = noiseAdded / photonCountMax;
				double value =  -1.0 * Math.log(log);
				//double test = Math.exp(imageProcessor.getPixelValue(k, j) * -1.0);
				//if (test == 0){
				//	throw new RuntimeException("test was 0!");
				//}
				imp.putPixelValue(k, j, value);
				//if (scaled < -3) {
					//System.out.println("test");
				//};
			}
		}
		return imp;
	}
	
	

	public void prepareForSerialization(){
		super.prepareForSerialization();
	}
	
	
	@Override
	public void configure() throws Exception {
		photonCountMax = UserUtil.queryDouble("Initial photon count:", photonCountMax);
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@book{buzug08-CTF,\nauthor={Buzug T},\ntitle={Computed Tomography - From Photon Statistics to Modern Cone-Beam CT},\npublisher={Springer Verlag},\nlocation={Berlin, Heidelberg},\nyear={2008}\n}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "T. M. Buzug. Computed Tomography - From Photon Statistics to Modern Cone-Beam CT. Springer Berlin Heidelberg. 2008";
	}

	/**
	 * Cosine filtering depends on the projection geometry and is hence not device depdendent.
	 */
	@Override
	public boolean isDeviceDependent() {
		return false;
	}
	
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
