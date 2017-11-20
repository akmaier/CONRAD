package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.WeightedInnerProductObjectiveFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class GUIOptimizationMode extends JFrame {

	private static final long serialVersionUID = 1L;

	// Panels
	private JPanel panel;

	private JLabel question;
	// Show original Points
	private JCheckBox showOriginalPoints;

	// Simple Optimization
	private JCheckBox simpleOptimization;
	private JLabel qsoK;
	private JTextField soK;

	// Time-Error Plot
	private JCheckBox plotTimeError;
	private JLabel qteFilename;
	private JTextField teFilename;
	private JCheckBox cteShowBestResult;
	private JLabel qteK;
	private JTextField teK;

	// K-Error Plot
	private JCheckBox plotKError;
	private JLabel qkeFilename;
	private JTextField keFilename;
	private JLabel qkeEndIntervalK;
	private JTextField keEndIntervalK;
	private JLabel qkeSteplengthK;
	private JTextField keSteplengthK;
	private JCheckBox keShowBestResult;

	// Parameter-K-Error Plot
	private JCheckBox plotParameterKError;
	private JLabel qpkeFilename;
	private JTextField pkeFilename;
	private JLabel qpkeEndIntervalK;
	private JTextField pkeEndIntervalK;
	private JLabel qpkeSteplengthK;
	private JTextField pkeSteplengthK;
	private JLabel qpkeParameter;
	private JRadioButton pkeParaOpt1;
	private JRadioButton pkeParaOpt2;
	private JRadioButton pkeParaOpt3;
	private JRadioButton pkeParaOpt4;
	private JLabel qpkeStartParameter;
	private JTextField pkeStartParameter;
	private JLabel qpkeEndParameter;
	private JTextField pkeEndParameter;
	private JLabel qpkeSteplengthParameter;
	private JTextField pkeSteplengthParameter;

	private JButton buttonNext;

	private DimensionalityReduction dimRed;
	private String code; 

	public GUIOptimizationMode(DimensionalityReduction dimRed, String code) {
		super("Sammon");
		this.code = code; 
		this.dimRed = dimRed;
	 
		setSize(800, 600);
		setLocation(100, 100);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);

		panel = new JPanel(new GridLayout(0, 2));

		getContentPane().add(panel);

		question = new JLabel("Chose your settinggs!");
		question.setHorizontalAlignment(JLabel.CENTER);
		getContentPane().add(BorderLayout.NORTH, question);

		showOriginalPoints = new JCheckBox("show original points");

		// Original Points
		showOriginalPoints = new JCheckBox("Show Original Points");
		showOriginalPoints.setFont(new java.awt.Font("Arial", Font.BOLD, 18));
		panel.add(showOriginalPoints);
		panel.add(new JLabel(""));

		// Simple Optimization
		simpleOptimization = new JCheckBox("Simple Optimization");
		simpleOptimization.setFont(new java.awt.Font("Arial", Font.BOLD, 18));
		qsoK = new JLabel("k:");
		soK = new JTextField();
		soK.setPreferredSize(new Dimension(200, 7));

		panel.add(simpleOptimization);
		panel.add(new JLabel(""));
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			panel.add(qsoK);
			panel.add(soK);
		}
		// Time-Error Plot
		plotTimeError = new JCheckBox("Time-Error Plot");
		plotTimeError.setFont(new java.awt.Font("Arial", Font.BOLD, 18));
		qteFilename = new JLabel("If you want to save this result, type in a filename, without .txt!");
		teFilename = new JTextField();
		cteShowBestResult = new JCheckBox("Show best result over time");
		qteK = new JLabel("K:");
		teK = new JTextField();

		panel.add(plotTimeError);
		panel.add(new JLabel(""));
		panel.add(qteFilename);
		panel.add(teFilename);
		panel.add(cteShowBestResult);
		panel.add(new JLabel(""));
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			panel.add(qteK);
			panel.add(teK);
		}
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			// k-Error Plot
			plotKError = new JCheckBox("k-Error Plot");
			plotKError.setFont(new java.awt.Font("Arial", Font.BOLD, 18));
			qkeFilename = new JLabel("If you want to save this result, type in a filename, without .txt!");
			keFilename = new JTextField();
			qkeEndIntervalK = new JLabel("intervall of k: 0 to ");
			keEndIntervalK = new JTextField();
			qkeSteplengthK = new JLabel("with a steplength of: ");
			keSteplengthK = new JTextField();
			keShowBestResult = new JCheckBox("Show the best result of all k");

			panel.add(plotKError);
			panel.add(new JLabel(""));
			panel.add(qkeFilename);
			panel.add(keFilename);
			panel.add(qkeEndIntervalK);
			panel.add(keEndIntervalK);
			panel.add(qkeSteplengthK);
			panel.add(keSteplengthK);
			panel.add(keShowBestResult);
			panel.add(new JLabel(""));

			// Parameter-k-Error plot
			if (dimRed.getProject() == "SwissRoll" || dimRed.getProject() == "Cube") {
				plotParameterKError = new JCheckBox("Parameter-k-Error Plot");
				plotParameterKError.setFont(new java.awt.Font("Arial", Font.BOLD, 18));
				qpkeFilename = new JLabel("If you want to save this result, type in a filename, without .txt!");
				pkeFilename = new JTextField();
				qpkeEndIntervalK = new JLabel("intervall of k: 0 to ");
				pkeEndIntervalK = new JTextField();
				qpkeSteplengthK = new JLabel("with a steplength of: ");
				pkeSteplengthK = new JTextField();
				qpkeParameter = new JLabel("Varying Parameter:");
				if (dimRed.getProject() == "SwissRoll") {
					ButtonGroup group = new ButtonGroup();
					pkeParaOpt1 = new JRadioButton("gap");
					pkeParaOpt1.setSelected(true);
					pkeParaOpt2 = new JRadioButton("width");
					pkeParaOpt3 = new JRadioButton("number of points");
					group.add(pkeParaOpt1);
					group.add(pkeParaOpt2);
					group.add(pkeParaOpt3);
				} else if (dimRed.getProject() == "Cube") {
					ButtonGroup group = new ButtonGroup();
					pkeParaOpt1 = new JRadioButton("dimension");
					pkeParaOpt1.setSelected(true);
					pkeParaOpt2 = new JRadioButton("width");
					pkeParaOpt3 = new JRadioButton("number of points per cloud");
					pkeParaOpt4 = new JRadioButton("standard deviation of the clouds");
					group.add(pkeParaOpt1);
					group.add(pkeParaOpt2);
					group.add(pkeParaOpt3);
					group.add(pkeParaOpt4);
				}

				qpkeStartParameter = new JLabel("lower bound of parameter interval:");
				pkeStartParameter = new JTextField();
				qpkeEndParameter = new JLabel("upper bound of parameter interval:");
				pkeEndParameter = new JTextField();
				qpkeSteplengthParameter = new JLabel("steplength: ");
				pkeSteplengthParameter = new JTextField();

				panel.add(plotParameterKError);
				panel.add(new JLabel(""));
				panel.add(qpkeFilename);
				panel.add(pkeFilename);
				panel.add(qpkeEndIntervalK);
				panel.add(pkeEndIntervalK);
				panel.add(qpkeSteplengthK);
				panel.add(pkeSteplengthK);
				panel.add(qpkeParameter);
				panel.add(new JLabel(""));
				panel.add(pkeParaOpt1);
				panel.add(pkeParaOpt2);
				panel.add(pkeParaOpt3);
				if (dimRed.getProject() == "Cube") {
					panel.add(pkeParaOpt4);
				} else {
					panel.add(new JLabel(""));
				}
				panel.add(qpkeStartParameter);
				panel.add(pkeStartParameter);
				panel.add(qpkeEndParameter);
				panel.add(pkeEndParameter);
				panel.add(qpkeSteplengthParameter);
				panel.add(pkeSteplengthParameter);
			}
		}

		qteFilename.setVisible(false);
		teFilename.setVisible(false);
		cteShowBestResult.setVisible(false);
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			qsoK.setVisible(false);
			soK.setVisible(false);
			qteK.setVisible(false);
			teK.setVisible(false);
			qkeFilename.setVisible(false);
			keFilename.setVisible(false);
			qkeEndIntervalK.setVisible(false);
			keEndIntervalK.setVisible(false);
			qkeSteplengthK.setVisible(false);
			keSteplengthK.setVisible(false);
			keShowBestResult.setVisible(false);
			if (dimRed.getProject() == "SwissRoll" || dimRed.getProject() == "Cube") {
				qpkeFilename.setVisible(false);
				pkeFilename.setVisible(false);
				qpkeEndIntervalK.setVisible(false);
				pkeEndIntervalK.setVisible(false);
				qpkeSteplengthK.setVisible(false);
				pkeSteplengthK.setVisible(false);
				qpkeParameter.setVisible(false);
				pkeParaOpt1.setVisible(false);
				pkeParaOpt2.setVisible(false);
				pkeParaOpt3.setVisible(false);
				if (dimRed.getProject() == "Cube")
					pkeParaOpt4.setVisible(false);
				qpkeStartParameter.setVisible(false);
				pkeStartParameter.setVisible(false);
				qpkeEndParameter.setVisible(false);
				pkeEndParameter.setVisible(false);
				qpkeSteplengthParameter.setVisible(false);
				pkeSteplengthParameter.setVisible(false);
			}
		}

		buttonNext = new JButton("NEXT");
		getContentPane().add(BorderLayout.SOUTH, buttonNext);

		simpleOptimization.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
					qsoK.setVisible(simpleOptimization.isSelected());
					soK.setVisible(simpleOptimization.isSelected());
					qteK.setVisible(plotTimeError.isSelected() && !simpleOptimization.isSelected());
					teK.setVisible(plotTimeError.isSelected() && !simpleOptimization.isSelected());
					
			}
		});
		plotTimeError.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				qteK.setVisible(plotTimeError.isSelected() && !simpleOptimization.isSelected());
				teK.setVisible(plotTimeError.isSelected() && !simpleOptimization.isSelected());
				qteFilename.setVisible(plotTimeError.isSelected());
				teFilename.setVisible(plotTimeError.isSelected());
				cteShowBestResult.setVisible(plotTimeError.isSelected());

			}
		});
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			plotKError.addActionListener(new ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {

					qkeFilename.setVisible(plotKError.isSelected());
					keFilename.setVisible(plotKError.isSelected());
					qkeEndIntervalK.setVisible(plotKError.isSelected());
					keEndIntervalK.setVisible(plotKError.isSelected());
					qkeSteplengthK.setVisible(plotKError.isSelected());
					keSteplengthK.setVisible(plotKError.isSelected());
					keShowBestResult.setVisible(plotKError.isSelected());
				}
			});
			if (dimRed.getProject() == "SwissRoll" || dimRed.getProject() == "Cube") {
				plotParameterKError.addActionListener(new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						qpkeFilename.setVisible(plotParameterKError.isSelected());
						pkeFilename.setVisible(plotParameterKError.isSelected());
						qpkeEndIntervalK.setVisible(plotParameterKError.isSelected());
						pkeEndIntervalK.setVisible(plotParameterKError.isSelected());
						qpkeSteplengthK.setVisible(plotParameterKError.isSelected());
						pkeSteplengthK.setVisible(plotParameterKError.isSelected());
						qpkeParameter.setVisible(plotParameterKError.isSelected());
						pkeParaOpt1.setVisible(plotParameterKError.isSelected());
						pkeParaOpt2.setVisible(plotParameterKError.isSelected());
						pkeParaOpt3.setVisible(plotParameterKError.isSelected());
						if (dimRed.getProject() == "Cube"){
							pkeParaOpt4.setVisible(plotParameterKError.isSelected());
						}
						qpkeStartParameter.setVisible(plotParameterKError.isSelected());
						pkeStartParameter.setVisible(plotParameterKError.isSelected());
						qpkeEndParameter.setVisible(plotParameterKError.isSelected());
						pkeEndParameter.setVisible(plotParameterKError.isSelected());
						qpkeSteplengthParameter.setVisible(plotParameterKError.isSelected());
						pkeSteplengthParameter.setVisible(plotParameterKError.isSelected());
					}
				});
			}
		}

		buttonNext.addActionListener(new ActionListener() { 
			@Override
			public void actionPerformed(ActionEvent e) {
				optimizeFunction(); 
				
				
			}
		});
	}
	
	
	public void optimizeFunction() {
		String codeCopy = code; 
		dimRed.setshowOrigPoints(showOriginalPoints.isSelected());
		code += "dimRed.setshowOrigPoints(" + showOriginalPoints.isSelected() + "); <br>";
		dimRed.setNormalOptimizationMode(simpleOptimization.isSelected());
		code += "dimRed.setNormalOptimizationMode(" + simpleOptimization.isSelected()+"); <br>";
		if (plotTimeError.isSelected()) {
			PlotIterationsError pie = new PlotIterationsError(teFilename.getText()); 
			dimRed.setPlotIterError(pie);
			dimRed.setBestTimeValue(cteShowBestResult.isSelected());
			
			code += "dimRed.setBestTimeValue(" + cteShowBestResult.isSelected() + "); <br>";
			code += "PlotIterationsError pie = new PlotIterationsError(\"" + teFilename.getText()+"\"); <br>";
			code += "dimRed.setPlotIterError(pie);<br>" ;
		}
		if (dimRed.getTargetFunction() instanceof WeightedInnerProductObjectiveFunction) {
			if (simpleOptimization.isSelected()) {
				if(soK.getText().length() == 0){
					System.out.println("You have to set a value for k for the simple Optimization!");
					code = codeCopy; 
					return; 
				}
				((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction())
						.setK(Double.parseDouble(soK.getText()));
				code += "((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).setK(" + Double.parseDouble(soK.getText()) + "); <br>";
				
			}
			if (plotTimeError.isSelected()) {
				if(teK.getText().length() == 0 && soK.getText().length() == 0){
					System.out.println("You have to set a value for k for the Time-Error plot!");
					code = codeCopy;
					return; 
				}
				if(teK.getText().length() != 0){
					((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction())
							.setK(Double.parseDouble(teK.getText()));
					code += "((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).setK(" + Double.parseDouble(teK.getText()) + "); <br>";
				}
			}
			if (plotKError.isSelected()) {
				if(keEndIntervalK.getText().length() == 0 || keSteplengthK.getText().length() == 0){
					System.out.println("You have to set the K-value and the steplength for the k-error plot!");
					code = codeCopy;
					return; 
				}
				PlotKError ke = new PlotKError(Double.parseDouble(keEndIntervalK.getText()),
						Double.parseDouble(keSteplengthK.getText()), keShowBestResult.isSelected(),
						((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).getVersionWIPOF(),
						keFilename.getText());
				dimRed.setPlotKError(ke);
				
				code += "PlotKError ke = new PlotKError(" +Double.parseDouble(keEndIntervalK.getText())+ ", " +	Double.parseDouble(keSteplengthK.getText()) + ", " + keShowBestResult.isSelected() + ", " + ((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).getVersionWIPOF() + ", \"" + keFilename.getText() +"\"); <br>" ;
				code += "dimRed.setPlotKError(ke); <br>";
				 
			}
			if (dimRed.getProject() == "SwissRoll" || dimRed.getProject() == "Cube") {
				if (plotParameterKError.isSelected()) {
					String varyingParameter = null;
					if (dimRed.getProject() == "Cube") {
						if (pkeParaOpt1.isSelected()) {
							varyingParameter = "dimension";
						} else if (pkeParaOpt2.isSelected()) {
							varyingParameter = "edgeLength";
						} else if (pkeParaOpt3.isSelected()) {
							varyingParameter = "numberOfPoints";
						} else if (pkeParaOpt4.isSelected()) {
							varyingParameter = "sd";
						}
					} else {
						if (pkeParaOpt1.isSelected()) {
							varyingParameter = "gap";
						} else if (pkeParaOpt2.isSelected()) {
							varyingParameter = "thickness";
						} else if (pkeParaOpt3.isSelected()) {
							varyingParameter = "numPointsSwiss";
						}
					}
					if(pkeEndIntervalK.getText().length() == 0 || pkeSteplengthK.getText().length() == 0 || pkeStartParameter.getText().length() == 0 ||pkeEndParameter.getText().length() == 0 || pkeSteplengthParameter.getText().length() == 0){
						System.out.println("You have to set the end and the steplength of the k interva and the start, end and steplength of the interval for the parameter, for the Parameter-k-error-plot");
						code = codeCopy;
						return; 
					}
					PlotParameterKError plotpke = new PlotParameterKError(dimRed.getProject(),
							((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).getVersionWIPOF(),
							Double.parseDouble(pkeEndIntervalK.getText()),
							Double.parseDouble(pkeSteplengthK.getText()), varyingParameter,
							Double.parseDouble(pkeStartParameter.getText()),
							Double.parseDouble(pkeEndParameter.getText()),
							Double.parseDouble(pkeSteplengthParameter.getText()), pkeFilename.getText());
					dimRed.setPlotParameterKError(plotpke);
					
					code += "PlotParameterKError plotpke = new PlotParameterKError(\"" + dimRed.getProject() + "\", " + ((WeightedInnerProductObjectiveFunction) dimRed.getTargetFunction()).getVersionWIPOF() + ", " + Double.parseDouble(pkeEndIntervalK.getText()) + ", " + Double.parseDouble(pkeSteplengthK.getText()) + " , \"" + varyingParameter + "\" , " +	Double.parseDouble(pkeStartParameter.getText()) + " , " + Double.parseDouble(pkeEndParameter.getText()) + ", " + Double.parseDouble(pkeSteplengthParameter.getText()) + ", \"" + pkeFilename.getText() + "\"); <br>";
					code += "dimRed.setPlotParameterKError(plotpke); <br>";
					
				}
			}
		}
	
		setVisible(false);
		new GUIAdditionalPlots(dimRed, code);
		
	}


}