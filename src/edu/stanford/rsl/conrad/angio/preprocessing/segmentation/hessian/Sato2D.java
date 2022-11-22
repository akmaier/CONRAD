/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.tools.TubenessProcessor;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;

public class Sato2D {
	
	Grid3D stack = null;
	int[] gSize = null;
	double[] gSpace =  null;
	
	private Roi roi = null;
	
	private Grid3D tubenessImage = null;
	
	/** Scales are to be given in mm */
	private double[] scales = new double[]{0.25, 0.5, 1, 2, 2.5};
	
	
	public static void main(String[] args) {
		String testFile = ".../test.tif";
		
		Grid3D img = ImageUtil.wrapImagePlus(IJ.openImage(testFile));
		
		Sato2D tubeness = new Sato2D(img);
		tubeness.setScales(new double[]{1.2,1.5,2.0});
		tubeness.evaluate();
		Grid3D tubenessImg = tubeness.getResult();
		
		new ImageJ();
		
		tubenessImg.show();
	}
	
	public Sato2D(Grid3D g){
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
		evaluateInternal();
	}
	
	private void evaluateInternal() {
		Grid3D filtered = new Grid3D(gSize[0], gSize[1], gSize[2]);
		filtered.setSpacing(gSpace);
		
		for(int k = 0; k < gSize[2]; k++){
//		for(int k = 0; k < 1; k++){
			System.out.println("Sato filtering on slice "+String.valueOf(k+1)+" of "+gSize[2]+".");
			ImagePlus[] tubes = new ImagePlus[scales.length];
			ImagePlus imp = ImageUtil.wrapGrid(stack.getSubGrid(k), "");
			for(int s = 0; s < scales.length; s++){
				TubenessProcessor tp = new TubenessProcessor(scales[s],true);
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
		this.tubenessImage = filtered;
	}

	public Grid3D getResult() {
		return this.tubenessImage;
	}
}
