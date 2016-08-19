/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric;

/**
 * Grid class to model multi channel data. It can be used to model vector valued voxels, but will still fit into
 * the image streaming concept.
 * 
 * @author akmaier
 *
 */
public class MultiChannelGrid2D extends Grid2D {

	String [] channelNames;
	Grid3D multichannelData;
	
	public MultiChannelGrid2D(int width, int height, int channels) {
		super(0,0);
		multichannelData = new Grid3D(width, height, channels);
		buffer = multichannelData.getSubGrid(0).getBuffer();
		initialize(width, height);
	}
	
	public MultiChannelGrid2D(MultiChannelGrid2D multiChannelGrid2D) {
		super(0,0);
		multichannelData = new Grid3D(multiChannelGrid2D.getWidth(), multiChannelGrid2D.getHeight(), multiChannelGrid2D.getNumberOfChannels());
		buffer = multichannelData.getSubGrid(0).getBuffer();
		// Deep copy channel names and image data...
		if( multiChannelGrid2D.getChannelNames()!= null){
			channelNames = new String[multiChannelGrid2D.getNumberOfChannels()];
		};
		for (int c= 0; c < multiChannelGrid2D.getNumberOfChannels(); c++){
			multichannelData.setSubGrid(c, (Grid2D) multiChannelGrid2D.getChannel(c).clone());
			if (channelNames!=null) channelNames[c] = multiChannelGrid2D.getChannelNames()[c];
		}
		initialize(multiChannelGrid2D.getWidth(), multiChannelGrid2D.getHeight());
	}

	/**
	 * Returns the Grid2D of the respective Channel
	 * @param c the channel number
	 * @return the Grid2D
	 */
	public Grid2D getChannel(int c){
		return multichannelData.getSubGrid(c);
	}
	
	/**
	 * Sets the channel in the image to the given grid 2D
	 * @param c the channel number
	 * @param channel the grid 2D
	 */
	public void setChannel(int c, Grid2D channel){
		if (c == 0){
			buffer = channel.getBuffer();
			for (int i = 0; i < columnOffsets.length; ++i) {
				subGrids[i] = new Grid1D(this.buffer, columnOffsets[i], size[0]);
			}
		}
		multichannelData.setSubGrid(c, channel);
	}
	
	/**
	 * Reports the number of channels
	 * @return the number of channels
	 */
	public int getNumberOfChannels(){
		return multichannelData.getSize()[2];
	}
	
	
	
	/**
	 * Set a pixel value at position (x,y)
	 * @param x The value's x position
	 * @param y The value's y position
	 * @param c The value's channel
	 * @param value The value to set
	 */
	public void putPixelValue(int x, int y, int c, float value) {
		multichannelData.setAtIndex(x, y, c, value);
	}
	
	
	/**
	 * Set a pixel value at position (x,y)
	 * @param x The value's x position
	 * @param y The value's y position
	 * @param c The value's channel
	 * @param value The value to set
	 */
	public void putPixelValue(int x, int y, int c, double value) {
		multichannelData.setAtIndex(x, y, c, (float)value);
	}
	
	@Override
	public void putPixelValue(int x, int y, double value) {
		multichannelData.setAtIndex(x, y, 0, (float)value);
	}
	
		
	/**
	 * Get the pixel value at position (x,y)
	 * @param x The value's x position
	 * @param y The value's y position
	 * @param c The value's channel
	 * @return the value of the pixel
	 */
	public float getPixelValue(int x, int y, int c) {
		return multichannelData.getAtIndex(x, y, c);
	}
	
	@Override
	public float getPixelValue(int x, int y) {
		return multichannelData.getAtIndex(x, y, 0);
	}
	
	@Override
	public float getAtIndex(int x, int y) {
		return multichannelData.getAtIndex(x, y, 0);
	}

	@Override
	public NumericGrid clone() {
		return (new Grid2D(this));
	}

	/**
	 * @return the channelNames
	 */
	public String[] getChannelNames() {
		return channelNames;
	}

	/**
	 * @param channelNames the channelNames to set
	 */
	public void setChannelNames(String[] channelNames) {
		this.channelNames = channelNames;
	}

}
