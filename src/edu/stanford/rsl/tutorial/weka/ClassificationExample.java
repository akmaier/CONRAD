/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.tutorial.weka;

import java.util.ArrayList;

import hr.irb.fastRandomForest.FastRandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 * This is a short example on how to use the Weka API. We will generate a training set and a test set.
 * Then we Train a Fast Random Forest Classifier and test it on the test feature vector set.
 * <br><br>
 * This is a general introduction to Weka API. Do not use this for image feature extraction.<br>
 * 
 * @author akmaier
 *
 */
public class ClassificationExample {

	/**
	 * This is the main routine
	 * @param args
	 */
	public static void main(String[] args) {
		// a set of feature vectors is stored in an Instances object in weka.
		// in oder to create such an object, we first have to create a list of features called attributes in weks.
		// in this example we generate 10 random real valued features plus a nominal class attribute with the classe "Class One" and "Class Two". 
		int numFeatures = 10;
		ArrayList<Attribute> attribs = new ArrayList<Attribute>(numFeatures+1);
		// generate 10 features and add them to the list of features.
		for (int i =0;i<numFeatures;i++){
			String nameString = "Feature " + i;
			attribs.add(new weka.core.Attribute(nameString));
		}
		// generate a list of nominal classes for the class attribute.
		ArrayList<String> classValues = new ArrayList<String>(2);
		classValues.add("Class one");
		classValues.add("Class two");
		// create the class attribute.
		Attribute classAttribute = new Attribute("Class", classValues);
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
			FastRandomForest frf = new FastRandomForest();
			frf.buildClassifier(trainingSet);
			
			// Evaluate Classifier:
			int hit = 0;
			for (int j = 0; j < vectors; j++){
				double prediction = frf.classifyInstance(testSet.instance(j));
				if (prediction == testSet.instance(j).value(numFeatures)) hit++;
			}
			
			// Report recognition rate
			System.out.println("Correct classifications (Should be about 50 %): " + (double)hit*100/vectors + " %");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
