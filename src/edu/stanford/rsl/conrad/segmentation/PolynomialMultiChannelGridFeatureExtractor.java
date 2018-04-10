/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.utils.UserUtil;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

public class PolynomialMultiChannelGridFeatureExtractor extends
		MultiChannelGridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1119850657741445026L;
	int polydegree = 3;
	
	@Override
	public void configure() throws Exception {
		polydegree = UserUtil.queryInt("Enter degree of polynomial transformation:", polydegree);
		multiChannelGrid = (MultiChannelGrid2D) dataGrid;
		int numFeatures = multiChannelGrid.getNumberOfChannels();
		attribs = new ArrayList<Attribute>(numFeatures+1);
		int combinations = (int) Math.pow(polydegree+1, numFeatures);
		int [] index = new int [numFeatures];
		for (int i =1;i<combinations;i++){
			int current = i;
			String indexString = "";
			for (int j=0;j<numFeatures; j++){
				index[j]=current % (polydegree+1);
				current /= (polydegree+1);
				indexString += " " +index[j];
			}
			String nameString = "Feature "+i + " degree " + indexString;
			attribs.add(new weka.core.Attribute(nameString));
		}
		Attribute classAttribute  = generateClassAttribute();
		attribs.add(classAttribute);
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);
		configured = true;
	}

	@Override
	public double [] extractFeatureAtIndex(int x, int y) {
		int numFeatures = multiChannelGrid.getNumberOfChannels();
		double attValues [] = new double [attribs.size()];
		attValues [attribs.size()-1] = labelGrid.getPixelValue(x, y); 
		int combinations = (int) Math.pow(polydegree+1, numFeatures);
		int [] index = new int [numFeatures];
		for (int count =1;count<combinations;count++){
			int current = count;
			double value = 1.0;
			for (int feat=0;feat<numFeatures; feat++){
				index[feat]=current % (polydegree+1);
				current /= (polydegree+1);
				double fromImage = multiChannelGrid.getChannel(feat).getPixelValue(x, y);
				value *= Math.pow(fromImage, index[feat]);
			}				
			attValues[count-1] = value;
			if (Double.isInfinite(value)){
				attValues[count-1] = Utils.missingValue();
			}
		}
		return attValues;
	}

	/**
	 * @return the polydegree
	 */
	public int getPolydegree() {
		return polydegree;
	}

	/**
	 * @param polydegree the polydegree to set
	 */
	public void setPolydegree(int polydegree) {
		this.polydegree = polydegree;
	}
	
	public String getName(){
		return "Polynomial Combination of Channels as Features";
	}

}
