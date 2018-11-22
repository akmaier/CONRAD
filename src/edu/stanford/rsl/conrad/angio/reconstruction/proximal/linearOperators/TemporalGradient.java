/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.proximal.linearOperators;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.LinearOperator;

public class TemporalGradient extends LinearOperator{

	@Override
	public float getForwBackOperatorNorm() {
		// such obvious, much clear. wow
		return 4;
	}

	@Override
	public int getNumberOfComponents() {
		return 1;
	}

	@Override
	public void apply(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xForw) {

		int[] gSize = x.get(0).getSize();
		for(int p = 0; p < x.size(); p++){
			for(int i = 1; i < gSize[0]; i++){
				for(int j = 1; j < gSize[1]; j++){
					for(int k = 1; k < gSize[2]; k++){
						float val = x.get(p).getPixelValue(i, j, k, 0);
						int nextP = (p+1)%x.size();
						float valNextP = x.get(nextP).getPixelValue(i, j, k, 0);
						xForw.get(p).putPixelValue(i, j, k, 0, valNextP-val);
					}
				}
			}		
		}
		
	}

	@Override
	public void applyAdjoint(ArrayList<MultiChannelGrid3D> u,
			ArrayList<MultiChannelGrid3D> uBack) {

		int[] gSize = u.get(0).getSize();
		for(int p = 0; p < u.size(); p++){
			for(int i = 1; i < gSize[0]; i++){
				for(int j = 1; j < gSize[1]; j++){
					for(int k = 1; k < gSize[2]; k++){
						float val = u.get(p).getPixelValue(i, j, k, 0);
						int prevP = ((p)>0)?(p-1):(u.size()-1);
						float valPrevP = u.get(prevP).getPixelValue(i, j, k, 0);
						uBack.get(p).putPixelValue(i, j, k, 0, -(val-valPrevP));
					}
				}
			}		
		}
		
	}

}
