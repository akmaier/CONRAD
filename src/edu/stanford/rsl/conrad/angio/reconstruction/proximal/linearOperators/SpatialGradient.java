package edu.stanford.rsl.conrad.angio.reconstruction.proximal.linearOperators;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.LinearOperator;

public class SpatialGradient extends LinearOperator{

	@Override
	public int getNumberOfComponents() {		
		return 3;
	}

	@Override
	public void apply(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xForw) {
		
		int[] gSize = x.get(0).getSize();
		for(int p = 0; p < x.size(); p++){
			for(int i = 0; i < gSize[0]-1; i++){
				for(int j = 0; j < gSize[1]-1; j++){
					for(int k = 0; k < gSize[2]-1; k++){
						float val = x.get(p).getPixelValue(i, j, k, 0);
						xForw.get(p).putPixelValue(i, j, k, 0, x.get(p).getPixelValue(i+1, j, k, 0)-val);
						xForw.get(p).putPixelValue(i, j, k, 1, x.get(p).getPixelValue(i, j+1, k, 0)-val);
						xForw.get(p).putPixelValue(i, j, k, 2, x.get(p).getPixelValue(i, j, k+1, 0)-val);
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
						float valX = u.get(p).getPixelValue(i, j, k, 0);
						float valY = u.get(p).getPixelValue(i, j, k, 1);
						float valZ = u.get(p).getPixelValue(i, j, k, 2);
						// negative backwards difference
						float gx = -(valX - u.get(p).getPixelValue(i-1, j, k, 0));
						float gy = -(valY - u.get(p).getPixelValue(i, j-1, k, 1));
						float gz = -(valZ - u.get(p).getPixelValue(i, j, k-1, 2));
						uBack.get(p).putPixelValue(i, j, k, 0, gx+gy+gz);
					}
				}
			}		
		}
		
	}

	@Override
	public float getForwBackOperatorNorm() {
		// this is really obvious tho
		return 12;
	}
	

}
