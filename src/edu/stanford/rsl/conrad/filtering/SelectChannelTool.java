/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Class to select only a single channel from a multi channel detector grid.
 * @author akmaier
 *
 */
public class SelectChannelTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2407399250396805087L;
	int channel = 0;
	
	@Override
	public void configure() throws Exception {
		channel = UserUtil.queryInt("Channel to select:", channel);
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
	public IndividualImageFilteringTool clone() {
		return this;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		if (imageProcessor instanceof MultiChannelGrid2D){
			MultiChannelGrid2D multi = (MultiChannelGrid2D) imageProcessor;
			return multi.getChannel(channel);
		}
		return imageProcessor;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getToolName() {
		String nameString = "Select Channel Tool";
		if (configured) nameString += " (channel " + channel + ")";
		return nameString;
	}

	/**
	 * @return the channel
	 */
	public int getChannel() {
		return channel;
	}

	/**
	 * @param channel the channel to set
	 */
	public void setChannel(int channel) {
		this.channel = channel;
	}

}
