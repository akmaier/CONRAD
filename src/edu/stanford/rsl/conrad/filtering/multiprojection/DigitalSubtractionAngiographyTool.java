/*
 * Copyright (C) 2016 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class DigitalSubtractionAngiographyTool extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9047172880007368459L;
	/**
	 * 
	 */
	int refImage =0;
	
	@Override
	public void configure() throws Exception {
		refImage = UserUtil.queryInt("Select Reference Image: ", refImage);
		configured = true;
	}

	@Override
	public String getBibtexCitation() {
		return "see medline.";
	}

	@Override
	public String getMedlineCitation() {
		return "Kerstin Müller, Moiz Ahmad, Martin Spahn, Jang-Hwan Choi, Silke Reitz, Niko Köster, Yanye Lu, Rebecca Fahrig, and Andreas Maier. Towards Material Decomposition on Large Field-of-View Flat Panel Photon-Counting Detectors — First In-vivo Results. CT Meeting 2016. submitted";
	}

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		// Nothing to do here
	}
	
	@Override
	protected void cleanup() {
		int numberOfImages= inputQueue.size();
		Grid2D refernceImage = (Grid2D) inputQueue.get(refImage).clone();
		for (int i = 0; i < numberOfImages; i++){
			inputQueue.get(i).getGridOperator().subtractBy(inputQueue.get(i), refernceImage);;
			try {
				sink.process(inputQueue.get(i), i);
				inputQueue.remove(i);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		super.cleanup();
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		String nameString = "Digital Subtraction Angiography Tool";
		if (configured) nameString += " (Reference Image " + refImage +")";
		return nameString;
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	public int getRefImage() {
		return refImage;
	}

	public void setRefImage(int refImage) {
		this.refImage = refImage;
	}

}
