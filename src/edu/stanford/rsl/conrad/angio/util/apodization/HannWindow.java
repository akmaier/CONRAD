package edu.stanford.rsl.conrad.angio.util.apodization;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public class HannWindow extends ApodizationWindow {

	private static final double alpha = 0.5;
	private static final double beta = 0.5;
	
	public static void main(String[] args){
		int width = 64;
		int height = 64;
		
		HannWindow hw = new HannWindow(width, height);
		Grid2D window = hw.getWindow();
		
		new ImageJ();
		window.show();
		
	}
	
	public HannWindow(int w, int h) {
		super(w, h);
	}
	
	@Override
	protected Grid2D setupWindow() {
		Grid2D window = new Grid2D(width, height);
		for(int i = 0; i < width; i++){
			double valU = alpha - beta*Math.cos((2*Math.PI*i)/(width-1));
			for(int j = 0; j < height; j++){
				double valV = alpha - beta*Math.cos((2*Math.PI*j)/(height-1));
				window.setAtIndex(i, j, (float)(valU*valV));
			}
		}
		return window;
	}

}
