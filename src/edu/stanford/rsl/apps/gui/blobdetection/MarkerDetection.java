package edu.stanford.rsl.apps.gui.blobdetection;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

import javax.swing.JFrame;
import javax.swing.JSpinner;
import javax.swing.JTextPane;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.XmlUtils;

/**
 * This code was edited or generated using CloudGarden's Jigloo
 * SWT/Swing GUI Builder, which is free for non-commercial
 * use. If Jigloo is being used commercially (ie, by a corporation,
 * company or business for any purpose whatever) then you
 * should purchase a license for each developer using Jigloo.
 * Please visit www.cloudgarden.com for details.
 * Use of Jigloo implies acceptance of these licensing terms.
 * A COMMERCIAL LICENSE HAS NOT BEEN PURCHASED FOR
 * THIS MACHINE, SO JIGLOO OR THIS CODE CANNOT BE USED
 * LEGALLY FOR ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
public class MarkerDetection extends JFrame implements ActionListener, ChangeListener, DocumentListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5274899502815414214L;
	private JButton jButtonRun;

	private MarkerDetectionWorker pWorker;
	private JSpinner jSpinnerGradientThreshold;
	private JTextPane jTextPane4;
	private JTextPane jTextPane3;
	private JTextPane jTextPane2;
	private JTextPane jTextPane1;
	private JTextPane jTextPaneBlobRadii;
	private JSpinner jSpinnerDistance;
	private JCheckBox jCheckBoxRefineAutoDetection;
	private JSpinner jSpinnerNumberOfBeadsToDetect;
	private JButton jButtonSaveResults;
	private JTextPane jTextPane6;
	private JSpinner jSpinnerNrOfIterations;
	private JCheckBox jCheckBoxAutomaticDetection;
	private JTextPane jTextPane5;
	private JSpinner jSpinnerCircularity;
	private JSpinner jSpinnerBinarizationThreshold;
	private boolean AutoDetectioncheckBoxChecked = false;
	private boolean RefinementcheckBoxChecked = false;

	public MarkerDetection () {
		initGUI();
		pWorker = new MarkerDetectionWorker();
		updateParameters();
	}


	private void initGUI(){
		{
			GridBagLayout thisLayout = new GridBagLayout();
			thisLayout.rowWeights = new double[] {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
			thisLayout.rowHeights = new int[] {20, 7, 20, 20, 7, 7, 7};
			thisLayout.columnWeights = new double[] {0.0, 0.1, 0.0, 0.0, 0.1, 0.1};
			thisLayout.columnWidths = new int[] {154, 20, 185, 29, 20, 7};
			getContentPane().setLayout(thisLayout);
			{
				jButtonRun = new JButton();
				getContentPane().add(jButtonRun, new GridBagConstraints(4, 6, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonRun.setText("Run Detection");
				jButtonRun.addActionListener(this);
			}
			{
				SpinnerModel jSpinnerBinarizationThresholdModel = 
						new SpinnerNumberModel(0.5, 0, 15, 0.1);
				jSpinnerBinarizationThreshold = new JSpinner();
				getContentPane().add(jSpinnerBinarizationThreshold, new GridBagConstraints(1, 5, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerBinarizationThreshold.setModel(jSpinnerBinarizationThresholdModel);
				jSpinnerBinarizationThreshold.setPreferredSize(new java.awt.Dimension(61, 28));
				jSpinnerBinarizationThreshold.addChangeListener(this);
			}
			{
				SpinnerModel jSpinnerGradientThresholdModel = 
						new SpinnerNumberModel(0,0,1,0.01);
				jSpinnerGradientThreshold = new JSpinner();
				getContentPane().add(jSpinnerGradientThreshold, new GridBagConstraints(1, 2, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerGradientThreshold.setModel(jSpinnerGradientThresholdModel);
				jSpinnerGradientThreshold.setPreferredSize(new java.awt.Dimension(61, 28));
				jSpinnerGradientThreshold.addChangeListener(this);
			}
			{
				SpinnerModel jSpinnerCircularityModel = 
						new SpinnerNumberModel(3,1,5,1);
				jSpinnerCircularity = new JSpinner();
				getContentPane().add(jSpinnerCircularity, new GridBagConstraints(1, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerCircularity.setModel(jSpinnerCircularityModel);
				jSpinnerCircularity.setPreferredSize(new java.awt.Dimension(61,28));
				jSpinnerCircularity.addChangeListener(this);
			}
			{
				jTextPane1 = new JTextPane();
				getContentPane().add(jTextPane1, new GridBagConstraints(0, 5, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane1.setText("Binarization Threshold");
			}
			{
				jTextPane2 = new JTextPane();
				getContentPane().add(jTextPane2, new GridBagConstraints(0, 2, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane2.setText("Low Gradient Threshold");
			}
			{
				jTextPane3 = new JTextPane();
				getContentPane().add(jTextPane3, new GridBagConstraints(0, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane3.setText("Blob radii");
			}
			{
				jTextPane4 = new JTextPane();
				getContentPane().add(jTextPane4, new GridBagConstraints(0, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane4.setText("Circularity Measure");
			}
			{
				jTextPaneBlobRadii = new JTextPane();
				getContentPane().add(jTextPaneBlobRadii, new GridBagConstraints(1, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jTextPaneBlobRadii.setText("[3.0]");
				jTextPaneBlobRadii.setPreferredSize(new java.awt.Dimension(61,28));
				jTextPaneBlobRadii.getDocument().addDocumentListener(this);
			}
			{
				jTextPane5 = new JTextPane();
				getContentPane().add(jTextPane5, new GridBagConstraints(0, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane5.setText("Distance to Reference");
			}
			{
				SpinnerModel jSpinnerDistanceModel = 
						new SpinnerNumberModel(15,1,200,10);
				jSpinnerDistance = new JSpinner();
				getContentPane().add(jSpinnerDistance, new GridBagConstraints(1, 4, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerDistance.setModel(jSpinnerDistanceModel);
				jSpinnerDistance.setPreferredSize(new java.awt.Dimension(61, 28));
				jSpinnerDistance.addChangeListener(this);
			}
			{
				jCheckBoxAutomaticDetection = new JCheckBox();
				getContentPane().add(jCheckBoxAutomaticDetection, new GridBagConstraints(4, 0, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jCheckBoxAutomaticDetection.setText("Use Automatic Marker Detection");
				jCheckBoxAutomaticDetection.addChangeListener(this);
			}
			{
				jCheckBoxRefineAutoDetection = new JCheckBox();
				getContentPane().add(jCheckBoxRefineAutoDetection, new GridBagConstraints(4, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.HORIZONTAL, new Insets(0, 0, 0, 0), 0, 0));
				jCheckBoxRefineAutoDetection.setText("Refine results / Set no. of beads");
				jCheckBoxRefineAutoDetection.setVisible(false);
				jCheckBoxRefineAutoDetection.addChangeListener(this);
			}
			{
				SpinnerModel jSpinnerNumberOfBeadsModel = 
						new SpinnerNumberModel(10,1,1000,1);
				jSpinnerNumberOfBeadsToDetect = new JSpinner();
				getContentPane().add(jSpinnerNumberOfBeadsToDetect, new GridBagConstraints(5, 1, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerNumberOfBeadsToDetect.setModel(jSpinnerNumberOfBeadsModel);
				jSpinnerNumberOfBeadsToDetect.setPreferredSize(new java.awt.Dimension(61,28));
				jSpinnerNumberOfBeadsToDetect.setEnabled(false);
				jSpinnerNumberOfBeadsToDetect.setVisible(false);
				jSpinnerNumberOfBeadsToDetect.addChangeListener(this);
			}
			{
				SpinnerModel jSpinnerNumberOfIterationsModel = 
						new SpinnerNumberModel(1,1,250,1);
				jSpinnerNrOfIterations = new JSpinner();
				getContentPane().add(jSpinnerNrOfIterations, new GridBagConstraints(1, 6, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jSpinnerNrOfIterations.setModel(jSpinnerNumberOfIterationsModel);
				jSpinnerNrOfIterations.setPreferredSize(new java.awt.Dimension(61,28));
				jSpinnerNrOfIterations.addChangeListener(this);
			}
			{
				jTextPane6 = new JTextPane();
				getContentPane().add(jTextPane6, new GridBagConstraints(0, 6, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jTextPane6.setText("No of Iterations");
			}
			{
				jButtonSaveResults = new JButton();
				getContentPane().add(jButtonSaveResults, new GridBagConstraints(4, 5, 1, 1, 0.0, 0.0, GridBagConstraints.CENTER, GridBagConstraints.NONE, new Insets(0, 0, 0, 0), 0, 0));
				jButtonSaveResults.setText("Save Results");
				jButtonSaveResults.addActionListener(this);
				jButtonSaveResults.setVisible(false);
			}
		}
		{
			this.setSize(757, 386);
		}
	}


	@Override
	public void actionPerformed(ActionEvent arg0) {
		if (arg0.getSource().equals(jButtonRun)){

			Thread test = new Thread(){
				@Override
				public void run() {
					boolean[] wasEnabled = new boolean[getContentPane().getComponents().length];
					for (int i = 0; i < getContentPane().getComponents().length; i++) {
						wasEnabled[i]=getContentPane().getComponents()[i].isEnabled();
						getContentPane().getComponents()[i].setEnabled(false);
					}
					try {
						updateParameters();
						// save states
							for (int j = 0; j < ((SpinnerNumberModel)jSpinnerNrOfIterations.getModel()).getNumber().intValue(); j++) {
								runDetection();
							}
							jButtonSaveResults.setVisible(true);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					for (int i = 0; i < getContentPane().getComponents().length; i++) {
						if (wasEnabled[i])
							getContentPane().getComponents()[i].setEnabled(true);
					}
				}
			};
			test.start();


		}
		
		if (arg0.getSource().equals(jButtonSaveResults)){
			
			Thread test = new Thread(){
				@Override
				public void run() {
					if (pWorker != null){
						try {
							XmlUtils.exportToXML(pWorker.getMeasuredTwoDPoints());
							XmlUtils.exportToXML(pWorker.getMergedTwoDPositions());
							XmlUtils.exportToXML(pWorker.getReferenceThreeDPoints());
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
				}
			};
			test.start();
			
		}
	}


	@Override
	public void stateChanged(ChangeEvent arg0) {
		
		if (arg0.getSource().equals(jSpinnerBinarizationThreshold) || arg0.getSource().equals(jSpinnerGradientThreshold)
				|| arg0.getSource().equals(jTextPaneBlobRadii) || arg0.getSource().equals(jSpinnerCircularity) ||
				arg0.getSource().equals(jSpinnerNumberOfBeadsToDetect) || arg0.getSource().equals(jSpinnerNrOfIterations)){

			Thread test = new Thread(){
				@Override
				public void run() {
					updateParameters();
				}
			};
			test.start();
		}

		
		if (arg0.getSource().equals(jCheckBoxAutomaticDetection)){
			if (AutoDetectioncheckBoxChecked != jCheckBoxAutomaticDetection.isSelected()){
				AutoDetectioncheckBoxChecked = jCheckBoxAutomaticDetection.isSelected();
				if (AutoDetectioncheckBoxChecked){
					pWorker = new AutomaticMarkerDetectionWorker();
					jCheckBoxRefineAutoDetection.setVisible(true);
					jSpinnerNumberOfBeadsToDetect.setVisible(true);
				}
				else{
					pWorker = new MarkerDetectionWorker();
					jCheckBoxRefineAutoDetection.setVisible(false);
					jSpinnerNumberOfBeadsToDetect.setVisible(false);
				}
				jButtonSaveResults.setVisible(false);
			}
		}
		
		
		if (arg0.getSource().equals(jCheckBoxRefineAutoDetection)){
			if (RefinementcheckBoxChecked != jCheckBoxRefineAutoDetection.isSelected()){
				RefinementcheckBoxChecked = jCheckBoxRefineAutoDetection.isSelected();
				if (RefinementcheckBoxChecked){
					pWorker = new AutomaticMarkerDetectionWorker();
					jSpinnerNumberOfBeadsToDetect.setEnabled(true);
				}
				else{
					jSpinnerNumberOfBeadsToDetect.setEnabled(false);
				}
				updateParameters();
			}
		}

	}


	private void updateParameters(){
		if (pWorker instanceof AutomaticMarkerDetectionWorker){
			((AutomaticMarkerDetectionWorker)pWorker).setParameters(((SpinnerNumberModel)jSpinnerBinarizationThreshold.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerCircularity.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerGradientThreshold.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerDistance.getModel()).getNumber().doubleValue(),
					jTextPaneBlobRadii.getText(),
					RefinementcheckBoxChecked,
					((SpinnerNumberModel)jSpinnerNumberOfBeadsToDetect.getModel()).getNumber().intValue());
		}
		else{
			pWorker.setParameters(((SpinnerNumberModel)jSpinnerBinarizationThreshold.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerCircularity.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerGradientThreshold.getModel()).getNumber().doubleValue(),
					((SpinnerNumberModel)jSpinnerDistance.getModel()).getNumber().doubleValue(),
					jTextPaneBlobRadii.getText());
		}
			
	}

	private void runDetection(){
		pWorker.run();
	}


	@Override
	public void changedUpdate(DocumentEvent arg0) {
		Thread test = new Thread(){
			@Override
			public void run() {
				updateParameters();
			}
		};
		test.start();
	}


	@Override
	public void insertUpdate(DocumentEvent arg0) {
		Thread test = new Thread(){
			@Override
			public void run() {
				updateParameters();
			}
		};
		test.start();
	}


	@Override
	public void removeUpdate(DocumentEvent arg0) {
		Thread test = new Thread(){
			@Override
			public void run() {
				updateParameters();
			}
		};
		test.start();
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		CONRAD.setup();
		MarkerDetection md = new MarkerDetection();
		md.setVisible(true);
		//Opener op = new Opener();
		//op.openImage("C:\\Users\\berger\\Desktop\\Result of WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy-2.tif").show("Result of WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy-2.tif");
		//op.openImage("D:\\Data\\WeightBearing\\stanford_knee_data_jang_2013_07_08\\Weight-bearing_NO6_IR1\\Projection\\AfterPreCorrection\\WEIGHT-BEARING.XA._.9.Standing_Straight_2Legs244LbsTotal_248views_70kV_0.54uGy.tif").show();
	}

}

/*
 * Copyright (C) 2010-2014 - Martin Berger 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/