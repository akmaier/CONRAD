/*
 * Copyright (C) 2010-2019 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.physics.photoncounting;

import ij.ImagePlus;

import ij.process.FloatProcessor;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import javax.swing.JOptionPane;

import weka.classifiers.Classifier;
import weka.core.Instances;
import edu.stanford.rsl.apps.gui.roi.EvaluateROI;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.segmentation.GridFeatureExtractor;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;

public class PredictMaterial extends EvaluateROI {

	Classifier classifier = null;
	GridFeatureExtractor featureExtractor;

	public void configure(ImagePlus testImage, Classifier classif, GridFeatureExtractor extractor) throws Exception {
		image = testImage;
		// image.show("testImage");//debug
		roi = image.getRoi();
		featureExtractor = extractor;
		int dimZ = image.getDimensions()[3];
		Grid3D dataGrid = ImageUtil.wrapImagePlus(image, false);
		// dataGrid.show("dataGrid");//debug
		Grid3D labelGrid = ImageUtil.wrapImagePlus(image, false);
		// labelGrid.show("labelGrid");//debug
		// int testValue=image.getCurrentSlice();//debug
		int currentSliceLabel = (image.getCurrentSlice() - 1) % dimZ;
		int currentChannelLabel = (image.getCurrentSlice() - 1) / dimZ;
		int currentSliceData = (image.getCurrentSlice() - 1) % dimZ;
		featureExtractor.setDataGrid(dataGrid.getSubGrid(currentSliceData));
		MultiChannelGrid2D labelGridSlice = (MultiChannelGrid2D) labelGrid.getSubGrid(currentSliceLabel);
		// labelGridSlice.show("labelGridSlice");//debug
		featureExtractor.setLabelGrid(labelGridSlice.getChannel(currentChannelLabel));
		if (!featureExtractor.isConfigured()) {
			featureExtractor.setClassName("class");
			featureExtractor.configure();
		}
		if (classif == null) {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
					Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.CLASSIFIER_MODEL_LOCATION)
							+ "classifier.model"));
			classifier = (Classifier) ois.readObject();
			ois.close();
		} else {
			classifier = classif;
		}

		configured = true;
	}

	@Override
	public void configure() throws Exception {
		ImagePlus[] images = ImageUtil.getAvailableImagePlusAsArray();
		ImagePlus image = (ImagePlus) JOptionPane.showInputDialog(null, "Select image: ", "Image Selection",
				JOptionPane.PLAIN_MESSAGE, null, images, images[0]);
		GridFeatureExtractor featureExtractor = GridFeatureExtractor.loadDefaultGridFeatureExtractor();
		configure(image, null, featureExtractor);
	}

	@Override
	public Object evaluate() {
		Instances data = null;
		int width = image.getWidth();
		int height = image.getHeight();

		/*
		 * Get the instance first, then delete it for fixing the bug that the
		 * vector length of the first instance("data.numInstances()") is larger
		 * than the input image(floatprocessor) in multiple images learning
		 * cases. --Yanye
		 */
		data = featureExtractor.getInstances();
		data.delete();

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
		}

		data = featureExtractor.getInstances();
		ImagePlus returnValue = null;
		try {

			FloatProcessor fl = new FloatProcessor(width, height);
			for (int i = 0; i < data.numInstances(); i++) {

				((float[]) fl.getPixels())[i] = (float) classifier.classifyInstance(data.instance(i));

			}

			// fl.min(0);
			// returnValue = VisualizationUtil.showImageProcessor(fl,
			// "Predicted Image");
			ImagePlus PredictedImage = new ImagePlus("Predicted Image", fl);
			returnValue = PredictedImage;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return returnValue;
	}

	@Override
	public String toString() {
		return "Predict Material";
	}

}
