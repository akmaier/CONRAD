package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.JTextField;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class GUIAdditionalPlots extends JFrame {

	private static final long serialVersionUID = 1L;
	private JCheckBox twoDTest;
	private JCheckBox threeDTest;

	private JLabel dimension2D;
	private JSpinner dim2D;
	private JLabel save2DQ;
	private JTextField filename2D;

	private JLabel dimension3D;
	private JSpinner dim3D;
	private JLabel save3DQ;
	private JTextField filename3D;
	
	private JCheckBox confidence;
	
	

	private JButton ok;

	private boolean twoDchecked = false;
	private boolean threeDchecked = false;

	private JLabel question;

	private JPanel panel;
	private DimensionalityReduction dimRed; 
	
	private String code; 

	public GUIAdditionalPlots(DimensionalityReduction dimRed, String code) {

		super("Sammon");
		this.dimRed= dimRed; 
		this.code = code; 
		setSize(800, 600);
		setLocation(100, 100);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setVisible(true);

		question = new JLabel(
				"Chose additional features!");
		question.setHorizontalAlignment(JLabel.CENTER);
		getContentPane().add(BorderLayout.NORTH, question);

		panel = new JPanel(new GridLayout(6, 2));
		twoDTest = new JCheckBox("2D convexity test");
		threeDTest = new JCheckBox("3D convexity test");
		confidence = new JCheckBox("Confidence Measure (result will be saved in Confidence.txt)");
		
		twoDTest.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				twoDchecked = !twoDchecked;
				twoDCheckBox();

			}
		});

		threeDTest.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				threeDchecked = !threeDchecked;
				threeDCheckBox();

			}
		});

		panel.add(twoDTest);
		panel.add(threeDTest);

		dimension2D = new JLabel("Which point do yo want to look at?");
		dimension2D.setVisible(false);
		dim2D = new JSpinner();
		dim2D.setVisible(false);
		save2DQ = new JLabel("If you want to save the result, type in a filename! (without .txt)");
		save2DQ.setVisible(false);
		filename2D = new JTextField();
		filename2D.setVisible(false);
		
		dimension3D = new JLabel("Which point do yo want to look at?");
		dimension3D.setVisible(false);
		dim3D = new JSpinner();
		dim3D.setVisible(false);
		save3DQ = new JLabel("If you want to save the result, type in a filename! (without .txt)");
		save3DQ.setVisible(false);
		filename3D = new JTextField();
		filename3D.setVisible(false);
		
		panel.add(dimension2D);
		panel.add(dimension3D);
		panel.add(dim2D);
		panel.add(dim3D);
		panel.add(save2DQ);
		panel.add(save3DQ);
		panel.add(filename2D);
		panel.add(filename3D);
		panel.add(confidence);
		

		
		getContentPane().add(panel);

		ok = new JButton("OK");

		ok.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				setVisible(false);
				try {
					nextPage();
				} catch (Exception e1) {
					e1.printStackTrace();
				}

			}
		});
		getContentPane().add(BorderLayout.SOUTH, ok);
	}

	private void twoDCheckBox() {
		if (twoDchecked) {
			dim2D.setVisible(true);
			dimension2D.setVisible(true);
			save2DQ.setVisible(true);
			filename2D.setVisible(true);
		} else {
			dim2D.setVisible(false);
			dimension2D.setVisible(false);
			save2DQ.setVisible(false);
			filename2D.setVisible(false);
		}
	}

	private void threeDCheckBox() {
		if (threeDchecked) {
			dim3D.setVisible(true);
			dimension3D.setVisible(true);
			save3DQ.setVisible(true);
			filename3D.setVisible(true);
		} else {
			dim3D.setVisible(false);
			dimension3D.setVisible(false);
			save3DQ.setVisible(false);
			filename3D.setVisible(false);
		}
	}

	private void nextPage() throws Exception {
		code += "dimRed.setConvexityTest2D(" + twoDTest.isSelected() + "); <br>";
		if (twoDTest.isSelected()) {
			dimRed.setConvexityTest2D(true);
			dimRed.set2Ddim((Integer)dim2D.getValue());
			dimRed.setFilenameConvexity2D(filename2D.getText());
			code += "dimRed.set2Ddim(" + (Integer)dim2D.getValue() + "); <br>";
			code += "dimRed.setFilenameConvexity2D(" + filename2D.getText() + " ); <br>";
		} else {
			dimRed.setConvexityTest2D(false);
		}
		code += "dimRed.setConvexityTest2D(" + threeDTest.isSelected() + "); <br>";
		if(threeDTest.isSelected()){
			dimRed.setConvexityTest3D(true);
			dimRed.set3Ddim((Integer) dim3D.getValue());
			dimRed.setFilenameConvexity3D(filename3D.getText());
			code += "dimRed.set3Ddim(" + (Integer)dim3D.getValue() + "); <br>";
			code += "dimRed.setFilenameConvexity3D(" + filename3D.getText() + " ); <br>";
		} else {
			dimRed.setConvexityTest3D(false);
		}
		
		dimRed.setConfidenceMeasure(confidence.isSelected());
		code += "dimRed.setConfidenceMeasure(" + confidence.isSelected()+ "); <br>";
		
		code += "dimRed.optimize(); </html>";
		
		
		Thread thread = new Thread() {
			public void run() {
				System.out.println("Thread Running");
				dimRed.optimize();
			}
		};
		thread.start();
		setVisible(false);
		System.out.println(code);
		new GUIshowCode(this.code);
		
		
		
		
	}	
}