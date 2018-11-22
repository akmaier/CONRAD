/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.proximableFunctions;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.ProximableFunction;

public class CharacteristicFunctionRplus extends ProximableFunction{

	@Override
	public float evaluate(ArrayList<MultiChannelGrid3D> x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void evaluateProx(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xProx, float tau) {
		
		int[] gSize = x.get(0).getSize();
		for(int p = 0; p < x.size(); p++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					for(int k = 0; k < gSize[2]; k++){
						float val = x.get(p).getPixelValue(i, j, k, 0);
						xProx.get(p).putPixelValue(i, j, k, 0, (val<0)?0:val);
					}
				}
			}		
		}
		
	}

	@Override
	public void evaluateConjugateProx(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xProx, float tau) {
		// TODO Auto-generated method stub
		
	}

}
