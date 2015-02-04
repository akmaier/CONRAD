/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.weka;

import java.util.ArrayList;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This is a short example on how to use the Weka API. We will generate a training set and a test set.
 * Then we Train a Linear Regression Classifier and test it on the test feature vector set.
 * <br><br>
 * This is a general introduction to Weka API. Do not use this for image feature extraction.<br>
 * 
 * @author akmaier
 *
 */
public class RegressionExample {

	/**
	 * This is the main routine
	 * @param args
	 */
	public static void main(String[] args) {
		// a set of feature vectors is stored in an Instances object in weka.
		// in oder to create such an object, we first have to create a list of features called attributes in weks.
		// in this example we generate 10 random real valued features plus a real valued class attribute. 
		int numFeatures = 10;
		ArrayList<Attribute> attribs = new ArrayList<Attribute>(numFeatures+1);
		// generate 10 features and add them to the list of features.
		for (int i =0;i<numFeatures;i++){
			String nameString = "Feature " + i;
			attribs.add(new weka.core.Attribute(nameString));
		}
		// generate a real valued class attribute.
		Attribute classAttribute = new Attribute("Class");
		// add to the list of features.
		attribs.add(classAttribute);
		

		// create 10000 random training vectors
		int vectors = 10000;
		Instances trainingSet = new Instances("Training Set", attribs, vectors);
		for (int j = 0; j < vectors; j++){
			double [] vector = new double [numFeatures+1];
			for (int i =0;i<numFeatures;i++){
				vector[i]=Math.random();
			}
			vector [numFeatures] = (int) (Math.random() *1.99);
			trainingSet.add(new DenseInstance(1.0, vector));
		}
		trainingSet.setClass(classAttribute);
		
		// create 10000 random test vectors
		Instances testSet = new Instances("Test Set", attribs, vectors);
		for (int j = 0; j < vectors; j++){
			double [] vector = new double [numFeatures+1];
			for (int i =0;i<numFeatures;i++){
				vector[i]=Math.random();
			}
			vector [numFeatures] = (int) (Math.random() *1.99);
			testSet.add(new DenseInstance(1.0, vector));
		}
		testSet.setClass(classAttribute);
		
		

		try {
			// Train Classifier
			LinearRegression frf = new LinearRegression();
			frf.buildClassifier(trainingSet);
			
			// Evaluate Classifier:
			double mse = 0;
			for (int j = 0; j < vectors; j++){
				double prediction = frf.classifyInstance(testSet.instance(j));
				mse += Math.pow(prediction - testSet.instance(j).value(numFeatures),2);
			}
			
			// Report recognition rate
			System.out.println("Mean square error (Should be about 0.5): " + (double)Math.sqrt(mse/vectors));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
