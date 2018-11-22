package edu.stanford.rsl.conrad.angio.reconstruction.proximal.proximableFunctions;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid3D;
import edu.stanford.rsl.conrad.angio.reconstruction.proximal.ProximableFunction;

public class OneTwoNorm extends ProximableFunction{

	private float lambda = 1f;
	
	public OneTwoNorm(float lambda) {
		this.lambda = lambda;
	}
	
	
	@Override
	public float evaluate(ArrayList<MultiChannelGrid3D> x) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void evaluateProx(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xProx, float tau) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void evaluateConjugateProx(ArrayList<MultiChannelGrid3D> x,
			ArrayList<MultiChannelGrid3D> xProx, float tau) {
		
		int[] gSize = x.get(0).getSize();
		for(int p = 0; p < x.size(); p++){
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					for(int k = 0; k < gSize[2]; k++){
						float norm = 0;
						for(int c = 0; c < x.get(p).getNumberOfChannels(); c++){
							float val = x.get(p).getPixelValue(i, j, k, c);
							norm += val*val;
						}
						norm = (float) Math.sqrt(norm)/lambda;
						for(int c = 0; c < x.get(p).getNumberOfChannels(); c++){
							float val = x.get(p).getPixelValue(i, j, k, c);
							if(lambda == 0f){
								xProx.get(p).putPixelValue(i, j, k, c, 0f);
							}else if(norm > 1){
								xProx.get(p).putPixelValue(i, j, k, c, val/norm);
							}else{
								xProx.get(p).putPixelValue(i, j, k, c, val);
							}
						}
					}
				}
			}		
		}
	}

}
