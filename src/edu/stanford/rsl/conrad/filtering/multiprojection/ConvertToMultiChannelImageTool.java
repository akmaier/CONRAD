/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.multiprojection;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class ConvertToMultiChannelImageTool extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1698240314046019080L;
	int channelNumber = 1;
	
	@Override
	public void configure() throws Exception {
		channelNumber = UserUtil.queryInt("Enter number of channels:", channelNumber);
		configured = true;
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
	protected void processProjectionData(int projectionNumber) throws Exception {
		// Nothing to do here
	}
	
	@Override
	protected void cleanup() {
		int numberOfImages= inputQueue.size() / channelNumber;
		Grid2D refernceImage = inputQueue.get(0);
		for (int i = 0; i < numberOfImages; i++){
			MultiChannelGrid2D multi = new MultiChannelGrid2D(refernceImage.getWidth(), refernceImage.getHeight(), channelNumber);
			for (int c = 0; c < channelNumber; c++){
				multi.setChannel(c, inputQueue.get(numberOfImages*c + i));
			}
			try {
				sink.process(multi, i);
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
		String nameString = "Convert to Multi Channel Data";
		if (configured) nameString += " (" + channelNumber +" channels)";
		return nameString;
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	/**
	 * @return the channelNumber
	 */
	public int getChannelNumber() {
		return channelNumber;
	}

	/**
	 * @param channelNumber the channelNumber to set
	 */
	public void setChannelNumber(int channelNumber) {
		this.channelNumber = channelNumber;
	}

}
