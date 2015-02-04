package edu.stanford.rsl.conrad.metric;

import ij.ImagePlus;

public abstract class NormalizedImageMetric extends ImageMetric {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6441213178752251933L;

	
	protected ImagePlus normalizationImage;

	/**
	 * @return the normalizationImage
	 */
	public ImagePlus getNormalizationImage() {
		return normalizationImage;
	}

	/**
	 * @param normalizationImage the normalizationImage to set
	 */
	public void setNormalizationImage(ImagePlus normalizationImage) {
		this.normalizationImage = normalizationImage;
	}
	
	public void prepareForSerialization() {
		super.prepareForSerialization();
		normalizationImage = null;
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/