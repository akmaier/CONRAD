/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.util;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;

public class CondatTools {

	public ArrayList<MultiChannelGrid3D> newEmptyMultiChannelGridList(int listSize,
			int[] gridSize, int gridChannels, double[] gridSpacing, double[] gridOrigin){
		ArrayList<MultiChannelGrid3D> otherList = new ArrayList<MultiChannelGrid3D>();
		for(int i = 0; i < listSize; i++){
			MultiChannelGrid3D newInstance = new MultiChannelGrid3D(gridSize[0],gridSize[1],gridSize[2],gridChannels);
			newInstance.setSpacing(gridSpacing);
			newInstance.setOrigin(gridOrigin);
			otherList.add(newInstance);			
		}
		return otherList;
	}
	
}
