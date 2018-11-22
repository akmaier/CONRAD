/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.autofocus;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

public class PointsAboveThresholdAutofocus {

	public static float evaluateAutoFocus(Grid3D img, float threshold){
		int above = 0;
		for(int k = 0; k < img.getSize()[2]; k++){
			for(int i = 0; i < img.getSize()[0]; i++){
				for(int j = 0; j < img.getSize()[1]; j++){
					if(img.getAtIndex(i, j, k) > threshold){
						above++;
					}
				}
			}
		}
		return above;	
	}
	
}
