package edu.stanford.rsl.conrad.angio.reconstruction.proximal.util;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;

public class MultiChannelGridOperators {
	
	
	//TODO make this guy fast
	public static MultiChannelGrid3D axpby(MultiChannelGrid3D g1, MultiChannelGrid3D g2, float mult1, float mult2){
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
