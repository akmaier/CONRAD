package edu.stanford.rsl.apps.gui.roi;

import ij.IJ;
import ij.process.ByteProcessor;

import org.fastica.FastICA;
import org.fastica.FastICAException;

import weka.core.matrix.Matrix;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
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
			
			double [][] vectors = null;
			if (operation.equals(ICA)){
				FastICA ica = new FastICA(signals, numChannels);
				vectors = ica.getICVectors();
			}
			if (operation.equals(PCA)){
				org.fastica.PCA pca = new org.fastica.PCA(signals);
				vectors = org.fastica.math.Matrix.mult(pca.getEigenVectors(), pca.getVectorsZeroMean());
				for (int k=0;k<numChannels;k++){
					System.out.println("Eigen Value "+k+" "+pca.getEigenValues()[k]);
				}
			}
			if(operation.equals(SVD)){
				weka.core.matrix.SingularValueDecomposition svd = new weka.core.matrix.SingularValueDecomposition(new Matrix(signals).transpose());
				vectors=svd.getU().transpose().getArray();
				for (int k=0;k<numChannels;k++){
					System.out.println("Singular Value "+k+" "+svd.getSingularValues()[k]);
				}
			}
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
		} catch (FastICAException e) {
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

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
