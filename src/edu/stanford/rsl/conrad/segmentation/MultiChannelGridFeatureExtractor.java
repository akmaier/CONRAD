/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;

public class MultiChannelGridFeatureExtractor extends GridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = -58855970888618830L;
	MultiChannelGrid2D multiChannelGrid;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		multiChannelGrid = null;
	}
	
	@Override
	public void setDataGrid(Grid2D dataGrid){
		super.setDataGrid(dataGrid);
		multiChannelGrid = (MultiChannelGrid2D) dataGrid;
	}
	
	@Override
	public void configure() throws Exception {
		multiChannelGrid = (MultiChannelGrid2D) dataGrid;
		int numFeatures = multiChannelGrid.getNumberOfChannels();
		// attribs entspricht features
		attribs = new ArrayList<Attribute>(numFeatures+1);
		for (int i =0;i<numFeatures;i++){
			String nameString = multiChannelGrid.getChannelNames()[i];
			attribs.add(new weka.core.Attribute(nameString));
		}
		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);
		//leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);
		configured = true;
	}

	@Override
	public double [] extractFeatureAtIndex(int x, int y) {
		//array mit Werten der Features
		double attValues [] = new double [attribs.size()];
		
		attValues [attribs.size()-1] = labelGrid.getPixelValue(x, y);
		//Channels entsprechen anzahl der verschiedenen bins
		int numFeatures = multiChannelGrid.getNumberOfChannels();
		//extrahiere feature f�r jeden bin
		for (int c = 0; c <numFeatures;c++){
			double value = multiChannelGrid.getChannel(c).getPixelValue(x, y);
			attValues[c] = value;
			if (Double.isInfinite(attValues[c])){
				attValues[c] = Utils.missingValue();
			}

		}
		return attValues;
	}

	@Override
	public String getName() {
		return "Channels as Features";
	}
}
