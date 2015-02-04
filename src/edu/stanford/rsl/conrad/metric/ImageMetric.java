package edu.stanford.rsl.conrad.metric;

import ij.ImagePlus;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import edu.stanford.rsl.apps.gui.Citeable;

public abstract class ImageMetric implements Serializable, Citeable {
	/**
	 * 
	 */
	protected ImagePlus testImage = null;
	protected ImagePlus referenceImage = null;

	
	public ImagePlus getTestImage() {
		return testImage;
	}

	public void setTestImage(ImagePlus recon) {
		this.testImage = recon;
	}

	public ImagePlus getReferenceImage() {
		return referenceImage;
	}

	public void setReferenceImage(ImagePlus idealRecon) {
		this.referenceImage = idealRecon;
	}

	private static final long serialVersionUID = 6626577275910593598L;

	public abstract double evaluate();
	
	public void writeObject(ObjectOutputStream ois){
		
	}
	
	public abstract String toString();
	
	public void readObject(ObjectInputStream oos){
		
	}
	
	public static ImageMetric [] getMetrics(){
		ImageMetric [] metrics = {new MeanSquareErrorMetric(), new RootMeanSquareErrorMetric(), new NormalizedImprovement()};
		return metrics;
	}
	
	@Override
	public boolean equals (Object another){
		return another.toString().equals(toString());
	}

	public void prepareForSerialization() {
		testImage = null;
		referenceImage = null;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/