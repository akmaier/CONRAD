/*
 * Copyright (C) 2017 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.segmentation;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.util.ArrayList;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JDialog;
import javax.swing.JFrame;

import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.MultiChannelGrid2D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import trainableSegmentation.FeatureStack;
import trainableSegmentation.FeatureStackArray;
import weka.core.Attribute;

public class ImageJFeatureExtractor extends GridFeatureExtractor {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3703678979250549741L;
	FeatureStackArray featureStackArray;

	/** flags of filters to be used */
	private boolean[] enabledFeatures = new boolean[] { false, /* Gaussian_blur */
			false, /* Sobel_filter */
			false, /* Hessian */
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

	int ImageJnumFeatures = 0;

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
		ImageJInputDialog dialog;
		JFrame frame=null;

		dialog = new ImageJInputDialog(frame, true);
		dialog.setVisible(true);

		CheckEnabledFeatures(dialog);

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

		attribs = new ArrayList<Attribute>(ImageJnumFeatures
				* trainingImage.getStackSize() + 1);

		// trainingImage.getImageStackSize() = 3
		for (int i = 0; i < trainingImage.getImageStackSize(); i++) {
			FeatureStack stack = featureStackArray.get(i);
			for (int j = 0; j < ImageJnumFeatures; j++) {
				String name = "Slice " + i + " Feature: "
						+ stack.getSliceLabel(j + 1);
				attribs.add(new weka.core.Attribute(name));
			}
		}

		Attribute classAttribute = generateClassAttribute();
		attribs.add(classAttribute);
		clearInstances(classAttribute);
		configured = true;
	}


	public void CheckEnabledFeatures(ImageJInputDialog dialog) {

		if (dialog.Gaussian_blur_Button.isSelected()) {
			enabledFeatures[0] = true;
		}
		if (dialog.Sobel_filter_Button.isSelected()) {
			enabledFeatures[1] = true;
		}
		if (dialog.Hessian_Button.isSelected()) {
			enabledFeatures[2] = true;
		}
		if (dialog.Difference_of_gaussians_Button.isSelected()) {
			enabledFeatures[3] = true;
		}
		if (dialog.Membrane_projections_Button.isSelected()) {
			enabledFeatures[4] = true;
		}
		if (dialog.Variance_Button.isSelected()) {
			enabledFeatures[5] = true;
		}
		if (dialog.Mean_Button.isSelected()) {
			enabledFeatures[6] = true;
		}
		if (dialog.Minimum_Button.isSelected()) {
			enabledFeatures[7] = true;
		}
		if (dialog.Maximum_Button.isSelected()) {
			enabledFeatures[8] = true;
		}
		if (dialog.Median_Button.isSelected()) {
			enabledFeatures[9] = true;
		}
		if (dialog.Anisotropic_diffusion_Button.isSelected()) {
			enabledFeatures[10] = true;
		}
		if (dialog.Bilateral_Button.isSelected()) {
			enabledFeatures[11] = true;
		}
		if (dialog.Lipschitz_Button.isSelected()) {
			enabledFeatures[12] = true;
		}
		if (dialog.Kuwahara_Button.isSelected()) {
			enabledFeatures[13] = true;
		}
		if (dialog.Gabor_Button.isSelected()) {
			enabledFeatures[14] = true;
		}
		if (dialog.Derivatives_Button.isSelected()) {
			enabledFeatures[15] = true;
		}
		if (dialog.Laplacian_Button.isSelected()) {
			enabledFeatures[16] = true;
		}
		if (dialog.Structure_Button.isSelected()) {
			enabledFeatures[17] = true;
		}
		if (dialog.Entropy_Button.isSelected()) {
			enabledFeatures[18] = true;
		}
		if (dialog.Neighbors_Button.isSelected()) {
			enabledFeatures[19] = true;
		}
	}

	@Override
	public double[] extractFeatureAtIndex(int x, int y) {
		if (featureStackArray == null) initFeatureStack();
		double[] result = new double[(ImageJnumFeatures * featureStackArray
				.getSize()) + 1];
		for (int i = 0; i < featureStackArray.getSize(); i++) {
			double[] singleVector = featureStackArray.get(i)
					.createInstance(x, y, 0).toDoubleArray();
			for (int j = 0; j < ImageJnumFeatures; j++) {
				result[(i * ImageJnumFeatures) + j] = singleVector[j];
			}
		}
		result[result.length - 1] = labelGrid.getAtIndex(x, y);
		return result;
	}

	public class ImageJInputDialog extends JDialog {
		private static final long serialVersionUID = 2854553949275357634L;
		JFrame frame = new JFrame();
		JCheckBox Gaussian_blur_Button;
		JCheckBox Sobel_filter_Button;
		JCheckBox Hessian_Button;
		JCheckBox Difference_of_gaussians_Button;
		JCheckBox Membrane_projections_Button;
		JCheckBox Variance_Button;
		JCheckBox Mean_Button;
		JCheckBox Minimum_Button;
		JCheckBox Maximum_Button;
		JCheckBox Median_Button;
		JCheckBox Anisotropic_diffusion_Button;
		JCheckBox Bilateral_Button;
		JCheckBox Lipschitz_Button;
		JCheckBox Kuwahara_Button;
		JCheckBox Gabor_Button;
		JCheckBox Derivatives_Button;
		JCheckBox Laplacian_Button;
		JCheckBox Structure_Button;
		JCheckBox Entropy_Button;
		JCheckBox Neighbors_Button;

		public ImageJInputDialog(JFrame frame, boolean modal) {
			super(frame, modal);
			setSize(400, 400);
			setLocationRelativeTo(null);
			setTitle("Choose the ImageJ Features");

			Gaussian_blur_Button = new JCheckBox("Gaussian blur");
			Sobel_filter_Button = new JCheckBox("Sobel filter");
			Hessian_Button = new JCheckBox("Hessian");
			Difference_of_gaussians_Button = new JCheckBox(
					"Difference of gaussians");
			Membrane_projections_Button = new JCheckBox("Membrane projections");
			Variance_Button = new JCheckBox("Variance");
			Mean_Button = new JCheckBox("Mean");
			Minimum_Button = new JCheckBox("Minimum");
			Maximum_Button = new JCheckBox("Maximum");
			Median_Button = new JCheckBox("Median");
			Anisotropic_diffusion_Button = new JCheckBox(
					"Anisotropic diffusion");
			Bilateral_Button = new JCheckBox("Bilateral filter");
			Lipschitz_Button = new JCheckBox("Lipschitz filter");
			Kuwahara_Button = new JCheckBox("Kuwahara filter");
			Gabor_Button = new JCheckBox("Gabor filter");
			Derivatives_Button = new JCheckBox("Derivatives");
			Laplacian_Button = new JCheckBox("Laplacian");
			Structure_Button = new JCheckBox("Structure filter");
			Entropy_Button = new JCheckBox("Entropy");
			Neighbors_Button = new JCheckBox("Neighbors");

			setLayout(new GridLayout(11, 1));
			add(Gaussian_blur_Button);
			add(Sobel_filter_Button);
			add(Hessian_Button);
			add(Difference_of_gaussians_Button);
			add(Membrane_projections_Button);
			add(Variance_Button);
			add(Mean_Button);
			add(Minimum_Button);
			add(Maximum_Button);
			add(Median_Button);
			add(Anisotropic_diffusion_Button);
			add(Bilateral_Button);
			add(Lipschitz_Button);
			add(Kuwahara_Button);
			add(Gabor_Button);
			add(Derivatives_Button);
			add(Laplacian_Button);
			add(Structure_Button);
			add(Entropy_Button);
			add(Neighbors_Button);

			add(new JButton(new ApproveAction()), BorderLayout.LINE_START);
			add(new JButton(new DeclineAction()), BorderLayout.LINE_END);

			pack();
			setDefaultCloseOperation(DISPOSE_ON_CLOSE);
			setResizable(false);
			setLocationRelativeTo(frame);
		}

		private class DeclineAction extends AbstractAction {
			private static final long serialVersionUID = -8989787206917692082L;

			private DeclineAction() {
				super("Cancel");
			}

			// TODO richtige aktion bei cancel einbauen
			public void actionPerformed(ActionEvent ae) {
				dispose();
				return;
			}
		}

		private class ApproveAction extends AbstractAction {
			private static final long serialVersionUID = -3410220235309510114L;

			private ApproveAction() {
				super("Ok");
			}

			public void actionPerformed(ActionEvent ae) {

				dispose();
			}
		}
	}

	@Override
	public String getName() {
		return "ImageJ Features (as in Weka Trainable Segmentation)";
	}
}
