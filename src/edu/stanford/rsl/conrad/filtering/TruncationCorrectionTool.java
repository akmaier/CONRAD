package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.opencl.OpenCLGrid2D;

/**
 * Class implements a simple truncation correction algorithm after 
 * Ohnesorge, Flohr, Schwarz, Heiken, and Bae
 * "Efficient correction for CT image artifacts caused by objects extending outside the scan field of view"
 * Med. Phys. 27 (1), Jan 2000
 * 
 * @author Andreas Maier
 *
 */

public class TruncationCorrectionTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7424197576259159222L;
	protected static int EXTENSION_FRACTION = 3;
	protected static int offset =3;
	
	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new TruncationCorrectionTool();
		clone.setConfigured(configured);
		return clone;
	}

	@Override
	public String getToolName() {
		return "Truncation Correction";
	}

	private int n_ext = 0;
	private int n_S = 0;
	private int k_SA;
	private int k_SE;
	private double tau_cos = 0.75;

	/**
	 * Cosine type weight w_{K_{S,A}} from Eq. 5a
	 * @param kprime as input
	 * @return the weight
	 */
	private double w_K_SA(int kprime){
		double revan = 0;
		if ((kprime > n_ext - k_SA - 1) && (kprime < n_ext)){
			double nominator = (kprime - n_ext + k_SA) * Math.PI;
			double denominator = 2 * (k_SA - 1);
			if (denominator != 0) {
				revan = Math.pow(Math.sin(nominator / denominator), tau_cos);
			} else {
				revan = 0;
			}
		}
		if (Double.isNaN(revan)){
			revan = 0;
		}
		return revan;
	}

	/**
	 * Cosine type weight w_{K_{S,E}} from Eq. 5b
	 * @param kprime as input
	 * @return the weight
	 */
	private double w_K_SE(int kprime){
		double revan = 0;
		if ((kprime > n_ext + n_S - 1 && (kprime < n_ext + (2 * n_S) - k_SE))){
			double nominator = (kprime - n_ext - n_S) * Math.PI;
			double denominator = 2 * (n_S - k_SE - 1);
			if (denominator != 0) {
				revan = Math.pow(Math.cos(nominator / denominator), tau_cos);
			} else {
				revan = 0;
			}
		}
		if (Double.isNaN(revan)){
			revan = 0;
		}
		return revan;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		n_S = imageProcessor.getWidth()-offset-offset;
		n_ext = (n_S / EXTENSION_FRACTION) + offset;
		
		Grid2D theFilteredProcessor = new Grid2D(n_S+ (2 * n_ext), imageProcessor.getHeight());
		// Zero padding (Eq. (2))
		for (int j = 0; j < theFilteredProcessor.getHeight(); j++){
			for (int kprime = 0; kprime < theFilteredProcessor.getWidth(); kprime++){
				if (kprime < n_ext) theFilteredProcessor.putPixelValue(kprime, j, 0);
				if ((kprime >= n_ext) && (kprime < n_ext + n_S)) {
					theFilteredProcessor.putPixelValue(kprime, j, imageProcessor.getPixelValue(kprime - n_ext+offset, j));
				}
				if ((kprime > n_ext + n_S -1)) theFilteredProcessor.putPixelValue(kprime, j, 0);
			}
			// values at the end of the detector
			double s_A = imageProcessor.getPixelValue(offset, j);
			double s_E = imageProcessor.getPixelValue(n_S - 1+offset, j);
			// Compute k_SA (K_{S,A} in the paper)
			k_SA = n_ext;
			for (int i = 0; i < n_S; i++){
				if (imageProcessor.getPixelValue(i+offset, j) > (2 * s_A)) {
					k_SA = i - 1;
					break;
				}
			}
			// Compute k_SE (K_{S,E} in the paper)
			k_SE =  n_S - n_ext - 1;
			for (int i = n_S - 1; i > 0; i--){
				if (imageProcessor.getPixelValue(i+offset, j) > (2 * s_E)) {
					k_SE = i + 1;
					break;
				}
			}
			//System.out.println(k_SA + " " + n_ext + " " + k_SE);
			// Mirror left Eq. (3a)
			for (int k = 1; k < Math.min(k_SA, n_ext) + 1; k++){
				theFilteredProcessor.putPixelValue(n_ext - k, j, (2 * s_A) - imageProcessor.getPixelValue(k+offset, j));
			}
			// Mirror right Eq. (3b)
			for (int k = n_S - 2; k > Math.max(k_SE, n_S - n_ext - 1) + 1; k--){
				theFilteredProcessor.putPixelValue((2 * n_S) + n_ext - 2 - k, j, (2 * s_E) - imageProcessor.getPixelValue(k+offset, j));
			}
			// Weight left side Eq. (4a)
			for (int kprime = 0; kprime < n_ext; kprime++){
				theFilteredProcessor.putPixelValue(kprime, j, theFilteredProcessor.getPixelValue(kprime, j) * w_K_SA(kprime));
			}
			// Weight right Eq. (4b)
			for (int kprime = n_S + n_ext; kprime < n_S + (2 * n_ext); kprime++){
				theFilteredProcessor.putPixelValue(kprime, j, theFilteredProcessor.getPixelValue(kprime, j) * w_K_SE(kprime));
			}
		}
		return theFilteredProcessor;
	}

	public OpenCLGrid2D applyToolToImage(OpenCLGrid2D imageProcessor) {
		n_S = imageProcessor.getWidth()-offset-offset;
		n_ext = (n_S / EXTENSION_FRACTION) + offset;
		
		OpenCLGrid2D theFilteredProcessor = new OpenCLGrid2D(new Grid2D(n_S+ (2 * n_ext), imageProcessor.getHeight()));
		// Zero padding (Eq. (2))
		for (int j = 0; j < theFilteredProcessor.getHeight(); j++){
			for (int kprime = 0; kprime < theFilteredProcessor.getWidth(); kprime++){
				if (kprime < n_ext) theFilteredProcessor.putPixelValue(kprime, j, 0);
				if ((kprime >= n_ext) && (kprime < n_ext + n_S)) {
					theFilteredProcessor.putPixelValue(kprime, j, imageProcessor.getPixelValue(kprime - n_ext+offset, j));
				}
				if ((kprime > n_ext + n_S -1)) theFilteredProcessor.putPixelValue(kprime, j, 0);
			}
			// values at the end of the detector
			double s_A = imageProcessor.getPixelValue(offset, j);
			double s_E = imageProcessor.getPixelValue(n_S - 1+offset, j);
			// Compute k_SA (K_{S,A} in the paper)
			k_SA = n_ext;
			for (int i = 0; i < n_S; i++){
				if (imageProcessor.getPixelValue(i+offset, j) > (2 * s_A)) {
					k_SA = i - 1;
					break;
				}
			}
			// Compute k_SE (K_{S,E} in the paper)
			k_SE =  n_S - n_ext - 1;
			for (int i = n_S - 1; i > 0; i--){
				if (imageProcessor.getPixelValue(i+offset, j) > (2 * s_E)) {
					k_SE = i + 1;
					break;
				}
			}
			//System.out.println(k_SA + " " + n_ext + " " + k_SE);
			// Mirror left Eq. (3a)
			for (int k = 1; k < Math.min(k_SA, n_ext) + 1; k++){
				theFilteredProcessor.putPixelValue(n_ext - k, j, (2 * s_A) - imageProcessor.getPixelValue(k+offset, j));
			}
			// Mirror right Eq. (3b)
			for (int k = n_S - 2; k > Math.max(k_SE, n_S - n_ext - 1) + 1; k--){
				theFilteredProcessor.putPixelValue((2 * n_S) + n_ext - 2 - k, j, (2 * s_E) - imageProcessor.getPixelValue(k+offset, j));
			}
			// Weight left side Eq. (4a)
			for (int kprime = 0; kprime < n_ext; kprime++){
				theFilteredProcessor.putPixelValue(kprime, j, theFilteredProcessor.getPixelValue(kprime, j) * w_K_SA(kprime));
			}
			// Weight right Eq. (4b)
			for (int kprime = n_S + n_ext; kprime < n_S + (2 * n_ext); kprime++){
				theFilteredProcessor.putPixelValue(kprime, j, theFilteredProcessor.getPixelValue(kprime, j) * w_K_SE(kprime));
			}
		}
		return theFilteredProcessor;
	}
	
	@Override
	public void configure() {
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Ohnesorge00-ECF,\n" +
		"  author = {{Ohnesorge}, B. and {Flohr}, T. and {Schwarz}, K. and {Heiken}, J.~P. and {Bae}, K.~T.},\n" +
		"  title = \"{{Efficient correction for CT image artifacts caused by objects extending outside the scan field of view}}\",\n" +
		"  journal = {Medical Physics},\n" +
		"  year = 2000,\n" +
		"  volume = 27,\n" +
		"  number = 1,\n" +
		"  pages = {39-46}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		String medline = "Ohnesorge B, Flohr T, Schwarz K, Heiken JP, Bae KT. " +
		"Efficient correction for CT image artifacts caused by objects extending outside the scan field of view. " +
		"Med Phys. 2000 Jan;27(1):39-46.";
		return medline;
	}

	/**
	 * Used to compensate truncation. Not device dependent.
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