/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.tools.VesselnessProcessor;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.HysteresisThresholding;

public class Frangi2D {
	Grid3D stack = null;
	int[] gSize = null;
	double[] gSpace =  null;
	
	private Roi roi = null;
	
	private Grid3D vesselnessImage = null;
	
	/** Scales are to be given in mm */
	private double[] scales = new double[]{0.25, 0.5, 1, 2, 2.5};
	
	private double structurenessPercentile = 1.0;
	
	
	public static void main(String[] args) {
		
		String myDir = "ExampleDir";
		Roi myRoi = null;
		double[] myScales = null;
		double myVesselnessScalingParameter = 0.99;
		double myLowThr = 0;
		double myHighThr = 1;
		
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(myDir));
		
		Frangi2D vness = new Frangi2D(img);
		vness.setScales(myScales);
		
		vness.setRoi(myRoi);
		vness.setStructurenessPercentile(myVesselnessScalingParameter);
		vness.evaluate(new double[]{0.02,0.0085});
		//vness.evaluate();
		Grid3D tubenessImg = vness.getResult();
		
		HysteresisThresholding thresh = new HysteresisThresholding(myLowThr,myHighThr);	
		ImagePlus hyst = thresh.run(ImageUtil.wrapGrid3D(tubenessImg,""));
		
		new ImageJ();
		
		tubenessImg.show();
		hyst.show();
	}
	
	public Frangi2D(Grid3D g){
		this.stack = g;		
	}
	
	public void setScales(double... sc){
		this.scales = sc;
	}
	
	public void setRoi(Roi r){
		this.roi = r;
	}
	
	public void evaluate(){
		gSize = this.stack.getSize();
		gSpace = this.stack.getSpacing();
		
		if(roi == null){
			this.roi = new Roi(0,0,stack.getSize()[0],stack.getSize()[1]);
		}
		evaluateInternal(new double[]{Double.MAX_VALUE,1});
	}
	
	public void evaluate(double[] gammaVals){
		gSize = this.stack.getSize();
		gSpace = this.stack.getSpacing();
		
		if(roi == null){
			this.roi = new Roi(0,0,stack.getSize()[0],stack.getSize()[1]);
		}
		evaluateInternal(gammaVals);
	}
	
	private void evaluateInternal(double[] gammaVals) {
		Grid3D filtered = new Grid3D(gSize[0], gSize[1], gSize[2]);
		filtered.setSpacing(gSpace);
		
		for(int k = 0; k < gSize[2]; k++){
//		for(int k = 0; k < 1; k++){
			System.out.println("Frangi on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
			ImagePlus[] tubes = new ImagePlus[scales.length];
			ImagePlus imp = ImageUtil.wrapGrid(stack.getSubGrid(k), "");
			for(int s = 0; s < scales.length; s++){
				VesselnessProcessor tp = new VesselnessProcessor(scales[s],structurenessPercentile,true);
				tp.setGammaThreshold(gammaVals);
				tp.setRoi(roi);
				tubes[s] = (ImagePlus)tp.generateImage(imp).clone();
			}			
			for(int i = 0; i < gSize[0]; i++){
				for(int j = 0; j < gSize[1]; j++){
					if(roi.contains(i, j)){
						float val = filtered.getAtIndex(i, j, k);
						for(int s = 0; s < scales.length; s++){
							val = Math.max(val, tubes[s].getProcessor().getPixelValue(i, j));
						}
						filtered.setAtIndex(i, j, k, val);
					}
				}
			}
		}
		this.vesselnessImage = filtered;
	}

	public Grid3D getResult() {
		return this.vesselnessImage;
	}

	public double getStructurenessPercentile() {
		return structurenessPercentile;
	}

	public void setStructurenessPercentile(double gammaMult) {
		this.structurenessPercentile = gammaMult;
	}
}
