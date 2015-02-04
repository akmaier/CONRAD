package edu.stanford.rsl.conrad.filtering;

import ij.ImagePlus;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Filtering tool to apply a 2D filter.
 * 
 * @author Andreas Maier
 *
 */
public class Filtering2DTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6298941333447055742L;
	private Grid2D filter2D = null;
	private boolean deviceDependent = false;
	
	
	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		Filtering2DTool filter = new Filtering2DTool();
		filter.setFilter2D(filter2D);
		filter.configured = configured;
		return filter;
	}

	@Override
	public String getToolName() {
		return "2D Filtering";
	}

	public void setFilter2D(Grid2D filter2D) {
		this.filter2D = filter2D;
	}

	public Grid2D getFilter2D() {
		return filter2D;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		return FFTUtil.apply2DFilter(imageProcessor, filter2D);
	}

	@Override
	public void configure() throws Exception {
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		ImagePlus filter = (ImagePlus) JOptionPane.showInputDialog(null, "Select filter: ", "Filter Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		deviceDependent = UserUtil.queryBoolean("Filter models device dependent behaviour?");
		setFilter2D(ImageUtil.wrapImageProcessor(filter.getChannelProcessor()));
	}

	@Override
	public boolean isDeviceDependent() {
		return deviceDependent;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void prepareForSerialization() {
		configured = false;
		filter2D = null;
	}
	
}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
