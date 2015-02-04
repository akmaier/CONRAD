package edu.stanford.rsl.conrad.filtering;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FFTUtil;

/**
 * Class to apply a ramp filter to an image before 3D reconstruction. After instantiation a
 * RampFilter has to be set. After that the ImageFilteringTool can be used as any other ImageFilteringTool.
 * 
 * @author Andreas Maier
 *
 */
public class RampFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2575773201054615567L;
	RampFilter ramp = null;
	
	@Override
	public IndividualImageFilteringTool clone() {
		RampFilteringTool clone = new RampFilteringTool();
		if (ramp != null) {
			clone.setRamp(ramp.clone());
		} else {
			clone.setRamp(null);
		}
		clone.setConfigured(configured);
		return clone;
	}

	public RampFilter getRamp() {
		return ramp;
	}

	/**
	 * Method to set the desired ramp for filtering.
	 * @param ramp the RampFilter to be applied in Fourier domain.
	 */
	public void setRamp(RampFilter ramp) {
		this.ramp = ramp;
	}

	@Override
	public String getToolName() {
		String revan = "Ramp Filtering";
		if (ramp != null) revan += " " + ramp.getRampName();
		return revan;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		return FFTUtil.applyRampFilter(imageProcessor, ramp.clone());
	}

	
	@Override
	public void configure() throws Exception {
		RampFilter [] ramps = RampFilter.getAvailableRamps();
		for (RampFilter ramp : ramps){
			ramp.setConfiguration(Configuration.getGlobalConfiguration());
		}
		RampFilter ramp = (RampFilter) JOptionPane.showInputDialog(null, "Please select the Ramp Filter", "Ramp Filter Selection", JOptionPane.DEFAULT_OPTION, null, ramps, this.ramp);
		ramp.configure();
		setRamp(ramp);
		setConfigured(true);
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@BOOK{Kak88-POC,\n" +
				"  author = {{Kak}, A. C. and {Slaney}, M.},\n" +
				"  title = {{Principles of Computerized Tomographic Imaging}},\n" +
				"  publisher = {IEEE Service Center},\n" +
				"  address = {Piscataway, NJ, United States},\n" +
				"  year = {1988}\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Kak AC, Slaney M, Principles of Computerized Tomographic Imaging, IEEE Service Center, Piscataway, NJ, United States 1988.";
	}

	/**
	 * Used to correct for the over-sampling of the projection center in Fourier domain by back projection
	 * Not device dependent. 
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