package edu.stanford.rsl.conrad.dimreduction.utils;


import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JTextField;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.geometry.shapes.simple.SwissRoll;
/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class GUI extends JFrame {

	private static final long serialVersionUID = 1L;
	private JRadioButton swissB;
	private JRadioButton cubeB;
	private JRadioButton dataB;
	
	private JLabel step1;
	private JPanel panel;

	private JPanel panelLeft;
	private JPanel panelMiddle;
	private JPanel panelRight;
	
	private JLabel swissGap; 
	private JTextField gap;
	private JLabel swissNumberOfPoints; 
	private JTextField numberOfPoints;
	private JLabel swissWidth; 
	private JTextField width; 
	
	private JLabel cubeWidth;
	private JTextField cWidth; 
	private JLabel cubeDimension;
	private JTextField cDimension; 
	private JLabel cubeSD;
	private JTextField cSD; 
	private JLabel cubeNumberOfCloudPoints;
	private JTextField cNumberOfCloudPoints; 
	private JLabel cubeComputeWithSavedPoints;
	private JCheckBox cComputeWithSavedPoints; 
	private JLabel cubeSavePoints;
	private JCheckBox cSavePoints;
	private JLabel cubeNoiseDimension;
	private JTextField cNoiseDimension;
	
	private JLabel dataPathQ; 
	private JTextField dataPath;
	
	private JButton next; 
	
	private JCheckBox checkScaleMax; 
	private JCheckBox checkScaleMean; 
	private JLabel scaleMaxDist; 
	private JLabel scaleMeanDist; 
	private JTextField maxDist; 
	private JTextField meanDist; 
	
	private String code;
	
	public GUI() {
		super("Sammon");
		this.code = "<html>"; 
		setSize(800, 600);
		setLocation(100, 100);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		
		step1 = new JLabel("<html> Step 2: Chose the data you want to map!<br> If you want a Parameter-k-Error plot you have to chose either Hypercube or Swiss Roll, set the value of the parameter you want to change arbitrarily, it will be set later");
		step1.setHorizontalAlignment(JLabel.LEFT);
		getContentPane().add(BorderLayout.NORTH, step1);
		next = new JButton("NEXT");
		getContentPane().add(BorderLayout.SOUTH, next);
		
		panel = new JPanel(new GridLayout(1,3));

		panelLeft = new JPanel(new GridLayout(0, 1));
		panelMiddle = new JPanel(new GridLayout(0, 2));
		panelRight = new JPanel(new GridLayout(0, 1));

		ButtonGroup data = new ButtonGroup(); 
		swissB = new JRadioButton("Swiss Roll", true);
		cubeB = new JRadioButton("Hypercube", false);
		dataB = new JRadioButton("Import real data" , false);
		data.add(swissB);
		data.add(cubeB);
		data.add(dataB);
		
		
		panelLeft.add(swissB);
		panelMiddle.add(cubeB);
		panelRight.add(dataB);
	

		panel.add(panelLeft);
		panel.add(panelMiddle);
		panel.add(panelRight);
		
		getContentPane().add(panel);
		
		cubeWidth = new JLabel("<html></br>Width of the cube:<html>");
		cWidth = new JTextField(10); 
		cubeDimension = new JLabel("<html></br> Dimension of the cube: <html>");
		cDimension = new JTextField(10);
		cubeSD = new JLabel("<html></br>Standard deviation of the point clouds:<html>");
		cSD  = new JTextField(10); 
		cubeNumberOfCloudPoints = new JLabel("<html></br>Number of points per cloud:<html>");
		cNumberOfCloudPoints = new JTextField(10); 
		cubeComputeWithSavedPoints = new JLabel("<html></br>Use saved points?<html>");
		cComputeWithSavedPoints = new JCheckBox(); 
		cubeSavePoints = new JLabel("<html></br>Save the points?<html>");
		cSavePoints = new JCheckBox();
		cubeNoiseDimension = new JLabel("<html></br>Dimension of the noise:<html>");
		cNoiseDimension = new JTextField(10);
		
		swissGap = new JLabel("<html></br>Gap size between 'snakes':<html>");
		gap = new JTextField(10);
		swissWidth = new JLabel("<html></br>Number of 'snakes'<html>");
		width = new JTextField(5);
		swissNumberOfPoints = new JLabel("<html></br>Number of points per 'snake':<html>");
		numberOfPoints = new JTextField(1);
		
		dataPathQ = new JLabel("<html></br>Type the name to the data<html>");
		dataPath = new JTextField(10);
		
		panelLeft.add(swissGap);
		swissGap.setVisible(true);
		panelLeft.add(gap);
		gap.setVisible(true);
		panelLeft.add(swissNumberOfPoints);
		swissNumberOfPoints.setVisible(true);
		panelLeft.add(numberOfPoints);
		numberOfPoints.setVisible(true);
		panelLeft.add(swissWidth);
		swissWidth.setVisible(true);
		panelLeft.add(width);
		width.setVisible(true);
		panelMiddle.add(new JLabel(""));
		panelMiddle.add(cubeWidth);
		cubeWidth.setVisible(false);
		panelMiddle.add(cWidth);
		cWidth.setVisible(false);
		panelMiddle.add(cubeDimension);
		cubeDimension.setVisible(false);
		panelMiddle.add(cDimension);
		cDimension.setVisible(false);
		panelMiddle.add(cubeSD);
		cubeSD.setVisible(false);
		panelMiddle.add(cSD);
		cSD.setVisible(false);
		panelMiddle.add(cubeNumberOfCloudPoints);
		cubeNumberOfCloudPoints.setVisible(false);
		panelMiddle.add(cNumberOfCloudPoints);
		cNumberOfCloudPoints.setVisible(false);
		panelMiddle.add(cubeComputeWithSavedPoints);
		cubeComputeWithSavedPoints.setVisible(false);
		panelMiddle.add(cComputeWithSavedPoints);
		cComputeWithSavedPoints.setVisible(false);
		panelMiddle.add(cubeSavePoints);
		cubeSavePoints.setVisible(false);
		panelMiddle.add(cSavePoints);
		cSavePoints.setVisible(false);
		panelMiddle.add(cubeNoiseDimension);
		cubeNoiseDimension.setVisible(false);
		panelMiddle.add(cNoiseDimension);
		cNoiseDimension.setVisible(false);
		panelRight.add(dataPathQ);
		dataPathQ.setVisible(false);
		panelRight.add(dataPath);
		dataPath.setVisible(false);
		
		
		panelLeft.add(new JLabel(""));
		panelMiddle.add(new JLabel(""));
		panelMiddle.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		
		
		checkScaleMax = new JCheckBox("Scale inner-point distance maximum");  
		checkScaleMean = new JCheckBox("Scale inner-point distance mean"); 
		scaleMaxDist = new JLabel("to"); 
		scaleMaxDist.setVisible(false);
		scaleMeanDist = new JLabel("to");
		scaleMeanDist.setVisible(false);
		maxDist = new JTextField(); 
		maxDist.setVisible(false);
		meanDist = new JTextField();
		meanDist.setVisible(false);
		panelLeft.add(new JLabel(""));
		panelLeft.add(checkScaleMax);
		panelMiddle.add(scaleMaxDist);
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(new JLabel(""));
		panelRight.add(maxDist);

		panelLeft.add(checkScaleMean);
		panelMiddle.add(new JLabel(""));
		panelMiddle.add(scaleMeanDist);
		panelRight.add(meanDist);
		
		setVisible(true);
		
		swissB.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				swissGap.setVisible(true);	
				gap.setVisible(true);	
				swissNumberOfPoints.setVisible(true);
				numberOfPoints.setVisible(true);
				swissWidth.setVisible(true);
				width.setVisible(true);
				cubeWidth.setVisible(false);
				cWidth.setVisible(false);
				cubeDimension.setVisible(false);
				cDimension.setVisible(false);
				cubeSD.setVisible(false);
				cSD.setVisible(false);
				cubeNumberOfCloudPoints.setVisible(false);
				cNumberOfCloudPoints.setVisible(false);;
				cubeComputeWithSavedPoints.setVisible(false);
				cComputeWithSavedPoints.setVisible(false);
				cubeSavePoints.setVisible(false);
				cSavePoints.setVisible(false);
				cubeNoiseDimension.setVisible(false);
				cNoiseDimension.setVisible(false);
				dataPathQ.setVisible(false);
				dataPath.setVisible(false);
			}
		});
		

		cubeB.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {

				swissGap.setVisible(false);	
				gap.setVisible(false);	
				swissNumberOfPoints.setVisible(false);
				numberOfPoints.setVisible(false);
				swissWidth.setVisible(false);
				width.setVisible(false);
				cubeWidth.setVisible(true);
				cWidth.setVisible(true);
				cubeDimension.setVisible(true);
				cDimension.setVisible(true);
				cubeSD.setVisible(true);
				cSD.setVisible(true);
				cubeNumberOfCloudPoints.setVisible(true);
				cNumberOfCloudPoints.setVisible(true);
				cubeComputeWithSavedPoints.setVisible(true);
				cComputeWithSavedPoints.setVisible(true);
				cubeSavePoints.setVisible(true);
				cSavePoints.setVisible(true);
				cubeNoiseDimension.setVisible(true);
				cNoiseDimension.setVisible(true);
				dataPathQ.setVisible(false);
				dataPath.setVisible(false);
			}
		});
		
		dataB.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				swissGap.setVisible(false);	
				gap.setVisible(false);	
				swissNumberOfPoints.setVisible(false);
				numberOfPoints.setVisible(false);
				swissWidth.setVisible(false);
				width.setVisible(false);
				cubeWidth.setVisible(false);
				cWidth.setVisible(false);
				cubeDimension.setVisible(false);
				cDimension.setVisible(false);
				cubeSD.setVisible(false);
				cSD.setVisible(false);
				cubeNumberOfCloudPoints.setVisible(false);
				cNumberOfCloudPoints.setVisible(false);;
				cubeComputeWithSavedPoints.setVisible(false);
				cComputeWithSavedPoints.setVisible(false);
				cubeSavePoints.setVisible(false);
				cSavePoints.setVisible(false);
				cubeNoiseDimension.setVisible(false);
				cNoiseDimension.setVisible(false);
				dataPathQ.setVisible(true);
				dataPath.setVisible(true);
			}
		});
		checkScaleMax.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				scaleMaxDist.setVisible(checkScaleMax.isSelected());			
				maxDist.setVisible(checkScaleMax.isSelected());
				
				checkScaleMean.setEnabled(!checkScaleMax.isSelected());
			}
		});
		checkScaleMean.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				scaleMeanDist.setVisible(checkScaleMean.isSelected());			
				meanDist.setVisible(checkScaleMean.isSelected());
			
				checkScaleMax.setEnabled(!checkScaleMean.isSelected());
			}
		});
		
		next.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				PointCloud pc;
				DimensionalityReduction dimRed; 
				if(swissB.isSelected() && width.getText().length() > 0 && gap.getText().length() > 0 && numberOfPoints.getText().length() > 0){
					SwissRoll sw = new SwissRoll(Double.parseDouble(gap.getText()), Integer.parseInt(numberOfPoints.getText()), Integer.parseInt(width.getText()));
					pc = new PointCloud(sw.getPointList());
					
					code += "SwissRoll sw = new SwissRoll(" + Double.parseDouble(gap.getText()) + ", " + Integer.parseInt(numberOfPoints.getText()) + ", " + Integer.parseInt(width.getText()) + "); <br>";
					code += "PointCloud pc = new PointCloud(sw.getPointList()); <br>";
						
					if(pc.points.size() > 0){
						 if(checkScaleMax.isSelected()){
								if(maxDist.getText().length() == 0){
									System.out.println("You have to set a max inner-point distance!");
									return;
								}
								pc.normalizeInnerPointDistancesMax(Double.parseDouble(maxDist.getText()));
								code +="pc.normalizeInnerPointDistancesMax(" + Double.parseDouble(maxDist.getText())+ "); <br>";
						} else if(checkScaleMean.isSelected()){
							if(meanDist.getText().length() == 0){
								System.out.println("You have to set a mean inner-point distance!");
								return;
							}
							pc.normalizeInnerPointDistancesMean(Double.parseDouble(meanDist.getText()));
							code +="pc.normalizeInnerPointDistancesMean(" + Double.parseDouble(meanDist.getText())+ "); <br>";
						}
						
						dimRed = new DimensionalityReduction(pc); 
						dimRed.setProject("SwissRoll"); //TODO loeschen?
						
						code += "DimensionalityReduction dimRed = new DimensionalityReduction(pc); <br>" ;
					} else {
						System.out.println("It must be at least one point!");
						return; 
					}
					
					
					//TODO ???
				} else if(cubeB.isSelected() && cWidth.getText().length() > 0 && cDimension.getText().length() > 0 && cSD.getText().length() > 0  && cNumberOfCloudPoints.getText().length() > 0 && cNoiseDimension.getText().length() > 0){
					Cube c = new Cube(Double.parseDouble(cWidth.getText()),Integer.parseInt(cDimension.getText()), Double.parseDouble(cSD.getText()), Integer.parseInt(cNumberOfCloudPoints.getText()), cComputeWithSavedPoints.isSelected(),  cSavePoints.isSelected(), Integer.parseInt(cNoiseDimension.getText()));
					pc = new PointCloud(c.getPointList());
					
					code += "Cube c = new Cube(" + Double.parseDouble(cWidth.getText())+ ", " +Integer.parseInt(cDimension.getText()) + ", " + Double.parseDouble(cSD.getText()) + ", " + Integer.parseInt(cNumberOfCloudPoints.getText()) + ", " + cComputeWithSavedPoints.isSelected() + ", " + cSavePoints.isSelected() + ", " + Integer.parseInt(cNoiseDimension.getText()) + "); <br>";
					code += "PointCloud pc = new PointCloud(c.getPointList()); <br>";
					
					
					if(pc.points.size() > 0){
						if(checkScaleMean.isSelected()){
							if(meanDist.getText().length() == 0){
								System.out.println("You have to set a mean inner-point distance!");
								return;
							}
							pc.normalizeInnerPointDistancesMean(Double.parseDouble(meanDist.getText()));	
							code +="pc.normalizeInnerPointDistancesMean(" + Double.parseDouble(meanDist.getText())+ "); <br>";
						} else if(checkScaleMax.isSelected()){
							if(maxDist.getText().length() == 0){
								System.out.println("You have to set a max inner-point distance!");
								return;
							}
							pc.normalizeInnerPointDistancesMax(Double.parseDouble(maxDist.getText()));		
							code +="pc.normalizeInnerPointDistancesMax(" + Double.parseDouble(maxDist.getText())+ "); <br>";
						}
						
						dimRed = new DimensionalityReduction(pc); 
						dimRed.setProject("Cube"); //TODO
						
						code += "DimensionalityReduction dimRed = new DimensionalityReduction(pc); <br>";
					} else {
						System.out.println("It must be at least one point!");
						return; 
					}
					//TODO ??
				} else if (dataB.isSelected() && dataPath.getText().length() != 0){
					try {
						double[][] distances = FileHandler.loadData(dataPath.getText());
						
						code += "double[][] distances = SaveOrLoadFile.loadData(" + dataPath.getText() + "); <br>";
						
						if(checkScaleMean.isSelected()){
							if(meanDist.getText().length() == 0){
								System.out.println("You have to set a mean inner-point distance!");
								return;
							}
							distances = HelperClass.normalizeInnerPointDistanceMean(distances, Double.parseDouble(meanDist.getText()));
							code +="distances = HelperClass.normalizeInnerPointDistanceMean(distances, " + Double.parseDouble(meanDist.getText()) + "); <br>";
						} else if(checkScaleMax.isSelected()){

							if(maxDist.getText().length() == 0){
								System.out.println("You have to set a max inner-point distance!");
								return;
							}
							distances = HelperClass.normalizeInnerPointDistanceMax(distances, Double.parseDouble(maxDist.getText()));
							code +="distances = HelperClass.normalizeInnerPointDistanceMean(distances, " + Double.parseDouble(maxDist.getText()) + "); <br>";
						}
						dimRed = new DimensionalityReduction(distances); 
						code += "dimRed = new DimensionalityReduction(distances); <br>";
						dimRed.setProject("RealData");
					} catch (NumberFormatException e1) {
						return; 
					} catch (IOException e1) {
						return;
					} catch (MyException e1) {
						System.out.println("Your file does not exist, type in another file name!");
						return;
					}
				} else {
					System.out.println("You have to answer all questions!");
					return;
				}
				setVisible(false); 
				new GUIOptFunc(dimRed, code);
				
			}
		});
	}
}