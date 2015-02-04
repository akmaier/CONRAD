package edu.stanford.rsl.conrad.metric;

public class NormalizedImprovement extends NormalizedImageMetric {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9211171034028752308L;

	@Override
	public double evaluate() {
		ImageMetric uncorrected = new RootMeanSquareErrorMetric();
		uncorrected.setReferenceImage(referenceImage);
		uncorrected.setTestImage(normalizationImage);
		ImageMetric corrected = new RootMeanSquareErrorMetric();
		corrected.setReferenceImage(referenceImage);
		corrected.setTestImage(testImage);
		double uncor = uncorrected.evaluate();
		double cor = corrected.evaluate();
		return (uncor - cor)/uncor;
	}

	@Override
	public String toString() {
		return "Normalized Improvement (0=no improvement; 1=ideal recon)";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Meyer10-AFA,\n" +
		"  author = {{Meyer}, M. and {Kalender}, W. A. and {Kyriakou}, Y. A.},\n" +
		"  title = \"{{A fast and pragmatic approach for scatter correction in flat-detector CT using elliptic modeling and iterative optimization}}\",\n" +
		"  journal = {Physics in Medicine and Biology},\n" +
		"  year = 2010,\n" +
		"  volume = 55,\n"+
		"  number = 1,\n" +
		"  pages = {99-120}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Meyer M, Kalender WA, Kyriakou Y. A fast and pragmatic approach for scatter correction in flat-detector CT using elliptic modeling and iterative optimization. Phys Med Biol 55(1):99-120. 2010.";
	}


	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/