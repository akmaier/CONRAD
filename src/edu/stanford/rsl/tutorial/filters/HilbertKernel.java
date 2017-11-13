package edu.stanford.rsl.tutorial.filters;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;


/**
 * Second part of the decomposition of the ramp filter into a first derivative and a Hilbert filter.
 * See L. Zeng. "Medical Image Reconstruction: A Conceptual tutorial". 2009, page 28
 * @author akmaier
 *
 */
public class HilbertKernel implements GridKernel{
	
	private double deltaS =0.d;
	
	public HilbertKernel(double deltaS) {
		this.deltaS = deltaS;
	}
	
	public void applyToGrid(Grid1D input) {
		
		int nTimes = 16;
		double[] dArray = new double[input.getSize()[0]];

		for(int i = 0; i < dArray.length; i++) {
			dArray[i] = input.getAtIndex(i);
		}
		double[] res = FFTUtil.hilbertTransform(dArray, nTimes);
		
		for(int i = 0; i < res.length; i++) {
			input.setAtIndex(i, (float) (-res[i]/(Math.PI*deltaS*2)));
		}
		
				
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Grid1D data = new Grid1D(new float[100]);
		
//		for (int i=0;i<data.getSize()[0];++i){
//			if( i >data.getSize()[0]/4 && i < data.getSize()[0]/2) {
//				data.setAtIndex(i, i*i);
//			}
//		}
		data.setAtIndex(51,1);
		HilbertKernel hKern = new HilbertKernel(1);
		new DerivativeKernel().applyToGrid(data);
		
		hKern.applyToGrid(data);
		Grid1DComplex compData = new Grid1DComplex(data);
		compData.transformForward();
//		hKern.applyToGrid(data);
		VisualizationUtil.createPlot(compData.getSubGrid(0, compData.getSize()[0]/2).getBuffer()).show();
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/