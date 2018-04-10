/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.util.ArrayList;

import ij.ImagePlus;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import weka.core.Attribute;
import weka.core.Instances;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.segmentation.ImageJFeatureExtractor;
import edu.stanford.rsl.conrad.utils.ImageUtil;

public class VesselnessFeatureExtractor extends ImageJFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3651080447445190302L;

	FeatureStackArray featureStackArray;

	/** flags of filters to be used */
	private boolean[] enabledFeatures = new boolean[] { false, /* Gaussian_blur */
	false, /* Sobel_filter */
	true, /* Hessian */
	false, /* Difference_of_gaussians */
	false, /* Membrane_projections */
	false, /* Variance */
	false, /* Mean */
	false, /* Minimum */
	false, /* Maximum */
	false, /* Median */
	false, /* Anisotropic_diffusion */
	false, /* Bilateral */
	false, /* Lipschitz */
	false, /* Kuwahara */
	false, /* Gabor */
	false, /* Derivatives */
	false, /* Laplacian */
	false, /* Structure */
	false, /* Entropy */
	false /* Neighbors */
	};

	/** use neighborhood flag */
	private boolean useNeighbors = true;
	/** expected membrane thickness */
	private int membraneThickness = 1;
	/** size of the patch to use to enhance the membranes */
	private int membranePatchSize = 19;

	/** minimum sigma to use on the filters */
	private float minimumSigma = 1f;
	/** maximum sigma to use on the filters */
	private float maximumSigma = 16f;

	protected double minimumVesselness = Double.MAX_VALUE;
	protected double maximumVesselness = Double.MIN_VALUE;

	@Override
	public void setDataGrid(Grid2D grid){
		dataGrid = grid;
		featureStackArray = null;
	}
	
	@Override
	public void setLabelGrid(Grid2D grid){
		labelGrid = grid;
		featureStackArray = null;
	}
	
	public synchronized void initFeatureStack(){
		ImagePlus trainingImage = ImageUtil.wrapGrid(dataGrid, "Data Grid");

		featureStackArray = new FeatureStackArray(
				trainingImage.getImageStackSize(), minimumSigma, maximumSigma,
				useNeighbors, membraneThickness, membranePatchSize,
				enabledFeatures);
		for (int i = 0; i < trainingImage.getImageStackSize(); i++) {
			FeatureStack stack = new FeatureStack(trainingImage.getImageStack()
					.getProcessor(i + 1));
			stack.setEnabledFeatures(featureStackArray.getEnabledFeatures());
			stack.setMembranePatchSize(membranePatchSize);
			stack.setMembraneSize(membraneThickness);
			stack.setMaximumSigma(maximumSigma);
			stack.setMinimumSigma(minimumSigma);
			stack.updateFeaturesMT();
			featureStackArray.set(stack, i);
		}
	}
	
	public void prepareForSerialization(){
		super.prepareForSerialization();
		featureStackArray = null;
	}
	
	@Override
	public void configure() throws Exception {

		ImagePlus trainingImage = ImageUtil.wrapGrid(new Grid2D(2,2), "Data Grid");
		if (dataGrid instanceof MultiChannelGrid2D){
			int channels = ((MultiChannelGrid2D) dataGrid).getNumberOfChannels();
			trainingImage = ImageUtil.wrapGrid(new MultiChannelGrid2D(2,2, channels), "Data Grid");
		}


		FeatureStackArray featureStackArray = new FeatureStackArray(
				trainingImage.getImageStackSize(), minimumSigma, maximumSigma,
				useNeighbors, membraneThickness, membranePatchSize,
				enabledFeatures);
		for (int i = 0; i < trainingImage.getImageStackSize(); i++) {
			FeatureStack stack = new FeatureStack(trainingImage.getImageStack()
					.getProcessor(i + 1));
			stack.setEnabledFeatures(featureStackArray.getEnabledFeatures());
			stack.setMembranePatchSize(membranePatchSize);
			stack.setMembraneSize(membraneThickness);
			stack.setMaximumSigma(maximumSigma);
			stack.setMinimumSigma(minimumSigma);
			stack.updateFeaturesMT();
			featureStackArray.set(stack, i);
		}

		ImageJnumFeatures = featureStackArray.getNumOfFeatures();

		String[] GaussianKernel = {"0,0", "1,0", "2,0", "4,0", "8,0", "16,0"};
		attribs = new ArrayList<Attribute>(19);

		for (int i = 0; i < trainingImage.getImageStackSize(); i++) {
			for (int j = 0; j < GaussianKernel.length; j++) {
				String name = "Slice " + i + ": Veselness for Gaussian Kernel " + GaussianKernel[j];
				attribs.add(new weka.core.Attribute(name));
			}

		}		 

		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);
		// leeres set von feature vectoren
		instances = new Instances(className, attribs, 0);
		instances.setClass(classAttribute);
		configured = true;

	}
		
	@Override
	public double[] extractFeatureAtIndex(int x, int y) {
		if (featureStackArray == null) initFeatureStack();
		double[] values = new double[(12 * featureStackArray.getSize())];
		for (int i = 0; i < featureStackArray.getSize(); i++) {
			double[] singleVector = featureStackArray.get(i)
					.createInstance(x, y, 0).toDoubleArray();
			int j = 4;
			int k = 0;
			while (j < ImageJnumFeatures) {
				values[(i * 12) + k] = singleVector[j];
				values[(i * 12) + (k + 1)] = singleVector[j + 1];
				k = k + 2;
				j = j + 8;
			}
		}

		double[] result = new double[(int) (Math.ceil(values.length/2) + 1)];
		int j = 0;
		//control parameter beta always 0.5
		double beta = 0.5;
		double bd = 2 * beta * beta;
		//The value of the threshold c depends on the grey-scale range of the
		//image and half the value of the maximum Hessian norm has proven to work in most cases.
		double c = 4;
		double cd = 2 * c * c;
		
		int i = 0;
		while(i <= values.length - 2){
			double eigenvalue_1 = values[i];
			double eigenvalue_2 = values[i + 1];

			if (Math.abs(eigenvalue_1) > Math.abs(eigenvalue_2)) {
				double R_b = eigenvalue_2 / eigenvalue_1;
				double S = Math.sqrt(eigenvalue_1 * eigenvalue_1 + eigenvalue_2
						* eigenvalue_2);
				double R_b_square = -(R_b * R_b);
				double S_square = -(S * S);
				//check for eigenvalue > 0
				if (eigenvalue_1 > 0) {
					result[j] = Math.exp(R_b_square / bd)
							* (1 - Math.exp(S_square) / cd);
					
				} else {
					result[j] = 0;
				}
				
			}
		
			
			//parameters for scaling of the output display range
			if (!Double.isNaN(result[j])) {
				maximumVesselness = Math.max(result[j], maximumVesselness);
				minimumVesselness = Math.min(result[j], minimumVesselness);
			}
			j++;
			i = i+2;
		}
		
		result[result.length-1] = labelGrid.getAtIndex(x, y);
		return result; //result;
	}

	@Override
	public String getName() {
		return "Veselness as defined by Frangi";
	}

}
