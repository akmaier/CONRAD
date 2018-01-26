package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.DistanceSquareObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.InnerProductObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.LagrangianDistanceObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.LagrangianDistanceSquareObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.LagrangianInnerProductObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.SammonObjectiveFunction;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.WeightedInnerProductObjectiveFunction;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class GUIOptFunc extends JFrame {

	private static final long serialVersionUID = 1L;
	private JLabel choseLagrangian;
	private JComboBox<String> lagrangians;

	private JLabel versionWIPOFQ;
	private JRadioButton versionWIPOF2;
	private JRadioButton versionWIPOF3;

	private JCheckBox pca;
	private JCheckBox lle;
	private JLabel numNeighborsQ;
	private JSpinner numNeighbors;

	private JButton ok;

	private JLabel question;

	private JPanel panel;
	private DimensionalityReduction dimRed;
	
	private String code;
	
	public GUIOptFunc(DimensionalityReduction dimRed, String code) {
		
		super("Sammon");
		this.dimRed = dimRed;
		this.code = code; 
		setSize(800, 600);
		setLocation(100, 100);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);

		question = new JLabel("Which optimization function do you want to use?");
		question.setHorizontalAlignment(JLabel.CENTER);
		getContentPane().add(BorderLayout.NORTH, question);

		panel = new JPanel(new GridLayout(9, 2));
		choseLagrangian = new JLabel("Lagrangians:");
		panel.add(choseLagrangian);
		String[] lagrangiansList = { "", "Sammon Objective Function",
				"Distance Square Objective Function",
				"Inner Product Objective Function",
				"Weighted InnerProduct Objective Function",
				"Lagrangian Inner Product Objective Function",
				"Lagrangian Distance Objective Function",
				"Lagrangian Distance Square Objective Function" };
		lagrangians = new JComboBox<String>(lagrangiansList);
		panel.add(lagrangians);

		lagrangians.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (lagrangians.getSelectedIndex() == 4) {
					versionWIPOFQ.setVisible(true);
					versionWIPOF2.setVisible(true);
					versionWIPOF3.setVisible(true);
				} else {
					versionWIPOFQ.setVisible(false);
					versionWIPOF2.setVisible(false);
					versionWIPOF3.setVisible(false);
				}
			}
		});
		versionWIPOFQ = new JLabel("Which version of this function?");
		versionWIPOFQ.setVisible(false);
		panel.add(versionWIPOFQ);
		versionWIPOF2 = new JRadioButton("Version 2");
		versionWIPOF2.setSelected(true);
		versionWIPOF2.setVisible(false);

		versionWIPOF3 = new JRadioButton("Version 3");
		versionWIPOF3.setSelected(true);
		versionWIPOF3.setVisible(false);

		ButtonGroup radios = new ButtonGroup();
		radios.add(versionWIPOF2);
		radios.add(versionWIPOF3);
		panel.add(versionWIPOF2);
		panel.add(versionWIPOF3);

		pca = new JCheckBox("PCA");
		lle = new JCheckBox("LLE");
		SpinnerNumberModel modelNumNeighbors = new SpinnerNumberModel(12, 1,
				100, 1);
		numNeighbors = new JSpinner(modelNumNeighbors);
		numNeighbors.setVisible(false);
		numNeighborsQ = new JLabel("how many neighbor points?");
		numNeighborsQ.setVisible(false);
		if (this.dimRed.getProject() != "RealTest") {
			panel.add(pca);
			panel.add(lle);
			panel.add(numNeighborsQ);
			panel.add(numNeighbors);
		}
		lle.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				numNeighbors.setVisible(lle.isSelected());
				numNeighborsQ.setVisible(lle.isSelected());
			}
		});

		getContentPane().add(panel);

		ok = new JButton("OK");

		ok.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				nextPage();

			}
		});
		getContentPane().add(BorderLayout.SOUTH, ok);


	}

	private void nextPage() {
	
		DimensionalityReduction.setLLE(lle.isSelected()); //TODO
//		if (lle.isSelected()) {
//			DimensionalityReduction.setNumNeighbors((int) (numNeighbors
//					.getValue()));
//		}
		DimensionalityReduction.setPCA(pca.isSelected()); //TODO
		PointCloudViewableOptimizableFunction gradientFunc = null;
		switch (lagrangians.getSelectedIndex() - 1) {
			case 0:
				gradientFunc = new SammonObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new SammonObjectiveFunction(); <br>";
				break;
			case 1:
				gradientFunc = new DistanceSquareObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new DistanceSquareObjectiveFunction(); <br>";
				break;
			case 2:
				gradientFunc = new InnerProductObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new InnerProductObjectiveFunction(); <br>";
				break;
			case 3:
				int version; 
				if (versionWIPOF2.isSelected()) {
					version = 2;
				} else {
					version = 3; 
				}
				gradientFunc = new WeightedInnerProductObjectiveFunction(version, 0); //k is set later
				code += "PointCloudViewableOptimizableFunction gradFunc = new WeightedInnerProductObjectiveFunction(" + version + ", 0 ); <br>";
				break;
			case 4:
				gradientFunc = new LagrangianInnerProductObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new LagrangianInnerProductObjectiveFunction(); <br>";
				break;
			case 5:
				gradientFunc = new LagrangianDistanceObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new LagrangianDistanceObjectiveFunction(); <br>";
				break;
			case 6:
				gradientFunc = new LagrangianDistanceSquareObjectiveFunction();
				code += "PointCloudViewableOptimizableFunction gradFunc = new LagrangianDistanceSquareObjectiveFunction(); <br>";
				break;
			default:
				System.out.println("You have to chose one Objective Function!");
				return; 	

		}

		dimRed.setTargetFunction(gradientFunc);
		code +="dimRed.setTargetFunction(gradFunc); <br>";
		setVisible(false);
		new GUIOptimizationMode(dimRed, this.code);

	}
}