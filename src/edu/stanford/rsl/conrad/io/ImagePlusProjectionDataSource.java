package edu.stanford.rsl.conrad.io;

import ij.IJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class ImagePlusProjectionDataSource extends FileProjectionSource {

	private Grid3D projections;
	
	@Override
	public void initStream(String filename) {
		projections = ImageUtil.wrapImagePlus(IJ.getImage(), true);
		super.currentIndex = -1;
	}
	
	public void setImage(Grid3D image){
		projections = image;
		super.currentIndex = -1;
	}

	@Override
	public Grid2D getNextProjection(){
		currentIndex ++;
		if (currentIndex < projections.getSize()[2]) {
			return projections.getSubGrid(currentIndex);
		} else {
			return null;
		}
	}
	
	public String getTitle(){
		return "Projection Source";
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/