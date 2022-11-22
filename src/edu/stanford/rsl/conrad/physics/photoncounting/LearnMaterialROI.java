/*
 * Copyright (C) 2010-2019 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.photoncounting;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import javax.swing.JOptionPane;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.SelectedTag;
import ij.ImagePlus;
import ij.gui.Plot;
import ij.process.FloatProcessor;
import edu.stanford.rsl.apps.gui.roi.EvaluateROI;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.segmentation.GridFeatureExtractor;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

public class LearnMaterialROI extends EvaluateROI {

	ImagePlus trainingImage;
	AbstractClassifier classifier;
	GridFeatureExtractor featureExtractor;
	String classifiername;


	@Override
	public void configure() throws Exception {
		ImagePlus[] images = ImageUtil.getAvailableImagePlusAsArray();
		ImagePlus image = (ImagePlus) JOptionPane.showInputDialog(null, "Select class image: ", "Image Selection",
				JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		ImagePlus trainingImage = (ImagePlus) JOptionPane.showInputDialog(null, "Select select training image: ",
				"Image Selection", JOptionPane.PLAIN_MESSAGE, null, images, images[0]);

		Classifier[] selection = { new LinearRegression(), new Bagging(), new REPTree(), new MultilayerPerceptron() };
		AbstractClassifier classifier = (AbstractClassifier) JOptionPane.showInputDialog(null,
				"Select select training method: ", "Classifier Selection", JOptionPane.PLAIN_MESSAGE, null, selection,
				selection[0]);

		GridFeatureExtractor featureExtractor = GridFeatureExtractor.loadDefaultGridFeatureExtractor();

		configure(image, trainingImage, classifier, featureExtractor);
	}

	public void configure(ImagePlus image, ImagePlus trainingImage, AbstractClassifier classifier,
			GridFeatureExtractor extractor) throws Exception {
		this.image = image;
		this.trainingImage = trainingImage;
		roi = image.getRoi();

		featureExtractor = extractor;

		int dimZ = trainingImage.getDimensions()[3];
		Grid3D dataGrid = ImageUtil.wrapImagePlus(trainingImage, false);// Load
		// images
		// to be
		// trained.
		Grid3D labelGrid = ImageUtil.wrapImagePlus(image, false);// Load
		// GroundTruth
		// Images.
		int currentSliceLabel = (image.getCurrentSlice() - 1) % dimZ;
		int currentChannelLabel = (image.getCurrentSlice() - 1) / dimZ;
		int currentSliceData = (trainingImage.getCurrentSlice() - 1) % dimZ;
		featureExtractor.setDataGrid(dataGrid.getSubGrid(currentSliceData));
		if (labelGrid.getSubGrid(currentSliceLabel) instanceof MultiChannelGrid2D) {
			MultiChannelGrid2D labelGridSlice = (MultiChannelGrid2D) labelGrid.getSubGrid(currentSliceLabel);
			featureExtractor.setLabelGrid(labelGridSlice.getChannel(currentChannelLabel));
			featureExtractor.setClassName(labelGridSlice.getChannelNames()[currentChannelLabel]);
		} else {
			featureExtractor.setLabelGrid(labelGrid.getSubGrid(currentSliceLabel));
			featureExtractor.setClassName("reference");
		}
		if (!featureExtractor.isConfigured())
			featureExtractor.configure();
		GridFeatureExtractor.setDefaultGridFeatureExtractor(featureExtractor);

		this.classifier = classifier;
		if (classifier instanceof LinearRegression) {
			LinearRegression linearRegression = (LinearRegression) classifier;
			linearRegression.setAttributeSelectionMethod(
					new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
			linearRegression.setEliminateColinearAttributes(false);
		}

		if (classifier instanceof REPTree) {
			REPTree repTree = (REPTree) classifier;
		}

		if (classifier instanceof Bagging) {
			Bagging bag = new Bagging();
			bag.setNumExecutionSlots(14);
		}

		configured = true;
	}

	@Override
	public Object evaluate() {

		int width = image.getWidth();
		int height = image.getHeight();

		Instances data = null;
		ImagePlus returnValue = null;
		int currentChannelLabel = (image.getCurrentSlice() - 1) / trainingImage.getDimensions()[3];

		boolean dataloadSuccess = false;
		if (Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CLASSIFIER_DATA_LOCATION) != null) {
			File f = new File(
					Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CLASSIFIER_DATA_LOCATION));
			if (f.exists()) {

				try {
					data = featureExtractor.loadInstances();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				data.setClassIndex(data.numAttributes() - 1);
				System.out.println("Data loaded");
				dataloadSuccess = true;
				// if no text.arff file exists calculate instances
			}
		}
		// if features already extracted and an test.arff exist no need for
		// extra calculation of instances
		if (!dataloadSuccess) {
			if (roi == null) {
				// no roi -> whole projection
				featureExtractor.extractAllFeatures();
			} else {

				width = roi.getBounds().width;
				height = roi.getBounds().height;


				for (int j = 0; j < roi.getBounds().height; j++) {
					for (int i = 0; i < roi.getBounds().width; i++) {
						int x = roi.getBounds().x + i;
						int y = roi.getBounds().y + j;
						featureExtractor.addInstance(featureExtractor.extractFeatureAtIndex(x, y));
					}
				}


				data = featureExtractor.getInstances();
				System.out.println("Data extracted");
				try {
					featureExtractor.saveInstances();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		// data loaded or extracted -> build classifier
		try {

			classifier.buildClassifier(data);
			System.out.println("Classifier built");
			double[] xCoord = new double[data.numInstances()];
			double[] yCoord = new double[data.numInstances()];
			// Creates a blank FloatProcessor using the default grayscale
			// LUT
			// that displays zero as black.
			FloatProcessor fl = new FloatProcessor(width, height);
			FloatProcessor classImage = new FloatProcessor(width, height);
			for (int i = 0; i < data.numInstances(); i++) {
				xCoord[i] = data.instance(i).classValue();
				yCoord[i] = classifier.classifyInstance(data.instance(i));
				// Returns a reference to the float array containing this
				// image's pixel data.
				((float[]) fl.getPixels())[i] = (float) classifier.classifyInstance(data.instance(i));
				((float[]) classImage.getPixels())[i] = (float) data.instance(i).classValue();
			}


			Plot a = VisualizationUtil.createScatterPlot("Prediction ", xCoord, yCoord, new LinearFunction()); a.draw(); a.show();

			returnValue = VisualizationUtil.showImageProcessor(fl,"Predicted Image (Train=Test)");
			VisualizationUtil.showImageProcessor(classImage, "Class Image");

			// featureExtractor.saveInstances();
			if (Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CLASSIFIER_MODEL_LOCATION) != null) {
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(
						Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CLASSIFIER_MODEL_LOCATION)
						+ "classifier.model"));
				oos.writeObject(classifier);
				oos.flush();
				oos.close();
			} else {
				System.out.println(
						"Warning: Classifier could not be written to disk. Please set CLASSIFIER_MODEL_LOCATION in the Registry correctly.");
			}
			// System.out.println(classifier);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return returnValue;
	}

	@Override
	public String toString() {
		return "Learn Material Predictor - Andreas";
	}

}
