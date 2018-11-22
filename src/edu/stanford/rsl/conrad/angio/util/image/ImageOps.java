/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.util.image;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

public class ImageOps {

	public static float getMeanInMask(Grid2D img, Grid2D mask){
		float mean = 0; 
		float maskSize = 0;
		for(int i = 0; i < img.getSize()[0]; i++){
			for(int j = 0; j < img.getSize()[1]; j++){
				if(mask.getAtIndex(i, j) > 0){
					mean += img.getAtIndex(i, j);
					maskSize++;
				}
			}				
		}
		return (mean/maskSize);
	}
	
	
	public static Grid2D normalizeOutsideMask(Grid2D img, Grid2D mask, double max, double min){
		Grid2D normalized = new Grid2D(img);
		for(int i = 0; i < img.getSize()[0]; i++){
			for(int j = 0; j < img.getSize()[1]; j++){
				if(mask.getAtIndex(i, j) > 0){
					normalized.setAtIndex(i, j, 1.0f);
				}else{
					float val = Math.min(1.0f, (float)((img.getAtIndex(i, j)-min) / (max-min)));
					normalized.setAtIndex(i, j, val);
				}
			}				
		}
		return normalized;
	}
	
	
	public static Grid3D normalizeMaskVals(Grid3D img, float newMin, float newMax, float min, float max){
		Grid3D normalized = new Grid3D(img);
		for(int k = 0; k < img.getSize()[2]; k++){
			for(int i = 0; i < img.getSize()[0]; i++){
				for(int j = 0; j < img.getSize()[1]; j++){
					float val = img.getAtIndex(i, j, k);
					val = (val-min)/(max-min)*(newMax-newMin)+newMin;
					val = Math.min(val, 1f);
					normalized.setAtIndex(i, j, k, val);
				}				
			}
		}
		return normalized;
	}
	
	public static Grid2D normalizeOutsideMask(Grid2D img, Grid2D mask, double max, double min, boolean clampToMin){
		Grid2D normalized = new Grid2D(img);
		for(int i = 0; i < img.getSize()[0]; i++){
			for(int j = 0; j < img.getSize()[1]; j++){
				if(mask.getAtIndex(i, j) > 0){
					normalized.setAtIndex(i, j, 1.0f);
				}else{
					float imgVal = img.getAtIndex(i, j);
					if(clampToMin){
						imgVal = (float)Math.max(imgVal, min);
					}
					float val = Math.min(1.0f, (float)((imgVal-min) / (max-min)));
					normalized.setAtIndex(i, j, val);
				}
			}				
		}
		return normalized;
	}

	
	public static Grid2D thresholdImage(Grid2D img, double threshold) {
		Grid2D thresh = new Grid2D(img);
		for(int i = 0; i < img.getSize()[0]; i++){
			for(int j = 0; j < img.getSize()[1]; j++){
				float getVal = img.getAtIndex(i, j);
				float setVal = (getVal>threshold)?1:0;
				thresh.setAtIndex(i, j, setVal);
			}
		}
			
		return thresh;
	}

	public static ArrayList<PointND> thresholdedPointList(Grid2D img, double threshold) {
		ArrayList<PointND> points = new ArrayList<PointND>();
		for(int i = 0; i < img.getSize()[0]; i++){
			for(int j = 0; j < img.getSize()[1]; j++){
				float getVal = img.getAtIndex(i, j);
				if(getVal > threshold){
					points.add(new PointND(i,j,0));
				}
			}
		}
			
		return points;
	}
	
}
