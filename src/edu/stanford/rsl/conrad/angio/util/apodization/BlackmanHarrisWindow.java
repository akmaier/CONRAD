package edu.stanford.rsl.conrad.angio.util.apodization;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class BlackmanHarrisWindow extends ApodizationWindow {

	private static final double a0 = 0.35875;
	private static final double a1 = 0.48829;
	private static final double a2 = 0.14128;
	private static final double a3 = 0.001168;
	
	public static void main(String[] args){
		int width = 64;
		int height = 64;
		
		BlackmanHarrisWindow bmw = new BlackmanHarrisWindow(width, height);
		Grid2D window = bmw.getWindow();
		
		new ImageJ();
		window.show();
		
	}
	
	
	public BlackmanHarrisWindow(int w, int h) {
		super(w, h);
	}

	@Override
	protected Grid2D setupWindow() {
		Grid2D window = new Grid2D(width, height);
		for(int i = 0; i < width; i++){
			double valU = a0 - a1*Math.cos((2*Math.PI*i)/(width-1)) + a2*Math.cos((4*Math.PI*i)/(width-1)) 
					- a3*Math.cos((6*Math.PI*i)/(width-1));
			for(int j = 0; j < height; j++){
				double valV = a0 - a1*Math.cos((2*Math.PI*j)/(height-1)) + a2*Math.cos((4*Math.PI*j)/(height-1))
						- a3*Math.cos((6*Math.PI*j)/(height-1));
				window.setAtIndex(i, j, (float)(valU*valV));
			}
		}
		return window;
	}


}
