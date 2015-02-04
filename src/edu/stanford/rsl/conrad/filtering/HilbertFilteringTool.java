package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;

/**
 * Tool to apply a 1-D Hilbert filter.
 * 
 * @author Andreas Maier
 *
 */
public class HilbertFilteringTool extends IndividualImageFilteringTool {


	/**
	 * 
	 */
	private static final long serialVersionUID = -8601705506734490146L;
	private int nTimes = 1;
	
	/**
	 * @return the nTimes
	 */
	public int getNTimes() {
		return nTimes;
	}

	/**
	 * @param nTimes the nTimes to set
	 */
	public void setNTimes(int nTimes) {
		this.nTimes = nTimes;
	}

	public HilbertFilteringTool (){
		configured = true;
	}
	
	@Override
	public IndividualImageFilteringTool clone() {
		// TODO Auto-generated method stub
		HilbertFilteringTool filter = new HilbertFilteringTool();
		filter.configured = configured;
		filter.nTimes = nTimes;
		return filter;
	}

	@Override
	public String getToolName() {
		return "Hilbert Filtering (Hilbert space size = " + nTimes + ")";
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		int width = imageProcessor.getWidth();
		int height = imageProcessor.getHeight();
		Grid2D revan = new Grid2D(width, height);
		for (int j = 0;  j < imageProcessor.getHeight(); j++){
			double [] filter = new double [width];
			for (int i=0;i<width;i++){
				filter[i] = imageProcessor.getPixelValue(i, j);
			}
			filter = FFTUtil.hilbertTransform(filter, nTimes);
			for (int i=0;i<width;i++){
				revan.putPixelValue(i,j,filter[i]);
			}
		}
		return revan;
	}

	@Override
	public void configure() throws Exception {
		nTimes = UserUtil.queryInt("Enter size of Hilbert space: ", nTimes);
		configured = true;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return "@article{Kak70-TDH, \nauthor={Kak SC},\ntitle={The discrete Hilbert transform.},\njournal={Proc IEEE},\nvolume={58},\nnumber={4},\npages={585-586},\nyear={1970}\n}";
	}

	@Override
	public String getMedlineCitation() {
		return "Kak SC. The discrete Hilbert transform. Proc IEEE 58(4):585-6. 1970.";
	}
	

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/