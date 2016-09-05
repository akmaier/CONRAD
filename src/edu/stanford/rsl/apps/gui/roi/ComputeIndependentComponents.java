/*
 * Copyright (C) 2016 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.apps.gui.roi;

import ij.IJ;
import ij.process.ByteProcessor;

import org.fastica.FastICAException;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.filtering.PatchwiseComponentComputationTool;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * Very useful tool to explore the components of small patches. It features PCA, SVD, and ICA.
 * @author akmaier
 *
 */
public class ComputeIndependentComponents extends EvaluateROI {

	private Grid3D multiGrid = null;
	private int currentImage = -1;
	private int numChannels = -1;
	private String operation = null;
	public static final String SVD = " SVD ";
	public static final String PCA = " PCA ";
	public static final String ICA = " ICA ";
	
	public void configure() throws Exception {
		image = IJ.getImage();
		roi = image.getRoi();
		if (roi != null){
			configured = true;
		}
		multiGrid = ImageUtil.wrapImagePlus(image);
		currentImage = image.getCurrentSlice() -1;
		numChannels = ((MultiChannelGrid2D) multiGrid.getSubGrid(0)).getNumberOfChannels();
		String [] operations = {SVD, PCA, ICA};
		operation = (String) UserUtil.chooseObject("Select operation: ", "Operation Selection", operations, operation);
		
	}

	@Override
	public Object evaluate() {
		double [][] signals = null;
		if (roi.getMask() == null){
			signals = new double [numChannels][roi.getBounds().height*roi.getBounds().width];
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					int x = roi.getBounds().x + i;
					int y = roi.getBounds().y + j;
					for (int k=0;k<numChannels;k++){
						signals[k][(j*roi.getBounds().width)+i] = ((MultiChannelGrid2D) multiGrid.getSubGrid(currentImage)).getChannel(k).getPixelValue(x, y);	
					}
				}
			}

		} else {
			// Count pixels in mask
			int count = 0;
			ByteProcessor mask = (ByteProcessor)roi.getMask();
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					if (mask.getPixel(i, j) == 255){
						count++;
					}
				}
			}
			signals = new double [numChannels][count];
			int index = 0;
			for (int j=0; j < roi.getBounds().height; j++){
				for (int i=0; i < roi.getBounds().width; i++){
					int x = roi.getBounds().x + i;
					int y = roi.getBounds().y + j;
					if (mask.getPixel(i, j) == 255){
						for (int k=0;k<numChannels;k++){
							signals[k][index] = ((MultiChannelGrid2D) multiGrid.getSubGrid(currentImage)).getChannel(k).getPixelValue(x, y);
						}
						index++;
					}
				}
			}
		}
		try {
			
			double [][] vectors = PatchwiseComponentComputationTool.getComponents(signals, numChannels, operation);

			ByteProcessor mask = (ByteProcessor)roi.getMask();
			MultiChannelGrid2D out = new MultiChannelGrid2D(roi.getBounds().width, roi.getBounds().height, numChannels);

			if (roi.getMask() == null){
				for (int j=0; j < roi.getBounds().height; j++){
					for (int i=0; i < roi.getBounds().width; i++){
						for (int k=0;k<numChannels;k++){
							out.getChannel(k).putPixelValue(i, j, vectors[k][(j*roi.getBounds().width)+i]);
						}
					}
				}
			} else {
				int index = 0;
				for (int j=0; j < roi.getBounds().height; j++){
					for (int i=0; i < roi.getBounds().width; i++){
						if (mask.getPixel(i, j) == 255){
							for (int k=0;k<numChannels;k++){
								out.getChannel(k).putPixelValue(i, j, vectors[k][index]);
							}
							index++;
						}
					}
				}
			}
			out.show("Components using" + operation);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public String toString() {
		return "Compute Components";
	}

}


