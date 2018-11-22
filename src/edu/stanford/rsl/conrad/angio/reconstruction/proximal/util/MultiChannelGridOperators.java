/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.util;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;

public class MultiChannelGridOperators {
	
	/**
	 * Computes element-wise weighted sum of two {@link MultiChannelGrid3D}
	 * a*X + b*Y
	 * @param g1 first MultiChannelGrid3D X
	 * @param g2 second MultiCHannelGrid3D Y
	 * @param mult1 first multiplier a
	 * @param mult2 second multiplier b
	 * @return
	 */
	public static MultiChannelGrid3D aXplusbY(MultiChannelGrid3D g1, MultiChannelGrid3D g2, float mult1, float mult2){
		int[] gridSize = g1.getSize();
		int gridChannels = g1.getNumberOfChannels();
		double[] gridOrigin = g1.getOrigin();
		double[] gridSpacing = g1.getSpacing();
		
		MultiChannelGrid3D result = new MultiChannelGrid3D(gridSize[0],gridSize[1],gridSize[2],gridChannels);
		result.setSpacing(gridSpacing);
		result.setOrigin(gridOrigin);
		for(int c = 0; c < gridChannels; c++){
			for(int i = 0; i < gridSize[0]; i++){
				for(int j = 0; j < gridSize[1]; j++){
					for(int k = 0; k < gridSize[2]; k++){
						float val = g1.getPixelValue(i, j, k, c)*mult1 + g2.getPixelValue(i, j, k, c)*mult2;
						result.putPixelValue(i, j, k, c, val);
					}
				}
			}
		}
		return result;
	}
	
	/**
	 * Computes element-wise sum of two {@link MultiChannelGrid3D}
	 * X + Y
	 * @param g1 first MultiChannelGrid3D X
	 * @param g2 second MultiCHannelGrid3D Y
	 * @return
	 */
	public static void apb(MultiChannelGrid3D g1, MultiChannelGrid3D g2){
		int[] gridSize = g1.getSize();
		int gridChannels = g1.getNumberOfChannels();
		
		for(int c = 0; c < gridChannels; c++){
			for(int i = 0; i < gridSize[0]; i++){
				for(int j = 0; j < gridSize[1]; j++){
					for(int k = 0; k < gridSize[2]; k++){
						float val = g1.getPixelValue(i, j, k, c) + g2.getPixelValue(i, j, k, c);
						g1.putPixelValue(i, j, k, c, val);
					}
				}
			}
		}
	}

	/**
	 * Computes pair-wise, element-wise weighted sum of two lists of {@link MultiChannelGrid3D}
	 * X(i) = a*X(i) + b*Y(i), where i is the list index.
	 * @param g1 first MultiChannelGrid3D X
	 * @param g2 second MultiCHannelGrid3D Y
	 * @param mult1 first multiplier a
	 * @param mult2 second multiplier b
	 * @return Result stored in g1List
	 */
	public static ArrayList<MultiChannelGrid3D> axpbyList(ArrayList<MultiChannelGrid3D> g1List, ArrayList<MultiChannelGrid3D> g2List,
			float mult1, float mult2){
		int[] gridSize = g1List.get(0).getSize();
		int gridChannels = g1List.get(0).getNumberOfChannels();
		double[] gridOrigin = g1List.get(0).getOrigin();
		double[] gridSpacing = g1List.get(0).getSpacing();
		
		ArrayList<MultiChannelGrid3D> resultList = new ArrayList<MultiChannelGrid3D>();
		for(int s = 0; s < g1List.size(); s++){
			MultiChannelGrid3D result = new MultiChannelGrid3D(gridSize[0],gridSize[1],gridSize[2],gridChannels);
			result.setSpacing(gridSpacing);
			result.setOrigin(gridOrigin);
			for(int c = 0; c < gridChannels; c++){
				for(int i = 0; i < gridSize[0]; i++){
					for(int j = 0; j < gridSize[1]; j++){
						for(int k = 0; k < gridSize[2]; k++){
							float val = g1List.get(s).getPixelValue(i, j, k, c)*mult1 
										+ g2List.get(s).getPixelValue(i, j, k, c)*mult2;
							result.putPixelValue(i, j, k, c, val);
						}
					}
				}
			}
			resultList.add(result);
		}
		return resultList;
	}
	
	/**
	 * Computes pair-wise, element-wise sum of two lists of {@link MultiChannelGrid3D}
	 * X(i) = X(i) + Y(i), where i is the list index.
	 * @param g1List first list of MultiChannelGrid3D
	 * @param g2 second list of MultiCHannelGrid3D
	 * @return Result stored in g1List
	 */
	public static void apbList(ArrayList<MultiChannelGrid3D> g1List, ArrayList<MultiChannelGrid3D> g2List){
		int[] gridSize = g1List.get(0).getSize();
		int gridChannels = g1List.get(0).getNumberOfChannels();
		
		for(int s = 0; s < g1List.size(); s++){
			for(int c = 0; c < gridChannels; c++){
				for(int i = 0; i < gridSize[0]; i++){
					for(int j = 0; j < gridSize[1]; j++){
						for(int k = 0; k < gridSize[2]; k++){
							float val = g1List.get(s).getPixelValue(i, j, k, c) + g2List.get(s).getPixelValue(i, j, k, c);
							g1List.get(s).putPixelValue(i, j, k, c, val);
						}
					}
				}
			}
		}
	}
	
}
