package edu.stanford.rsl.conrad.filtering;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.WeightBearingBeadPositionBuilder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.hough.FixedCircleHoughSpace;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class FiducialMarkerDetectionTool extends IndividualImageFilteringTool {

	/**
	 * Fiducial marker detection (e.g. bead marker for motion correction in the weight-bearing project)
	 * 
	 * @author Jang-Hwan Choi
	 */
	private static final long serialVersionUID = -6110577475173113158L;

//	For 1mm bead	
	private int radiusOfBeads = 3;	
	private double offset = 0.5; // 0.75		
	private double start = 0.1;	
	private double stop = 2;
	private double distance = 30;//20
	
//	For 2mm bead
//	private int radiusOfBeads = 5;	
//	private double offset = 0.5; // 0.75		
//	private double start = 0.1;	
//	private double stop = 3.0;
//	private double distance = 20;

	static WeightBearingBeadPositionBuilder beadBuilder;
	Configuration config = Configuration.getGlobalConfiguration();
	
	private static boolean initBead = false;
	
	public FiducialMarkerDetectionTool (){
		configured = true;
	}
	

	protected synchronized static void initializeBead(){
		if (!initBead){
			System.out.println("Read in initial bead positions.");
			
			beadBuilder = new WeightBearingBeadPositionBuilder();
			beadBuilder.readInitialBeadPositionFromFile();
			beadBuilder.estimateBeadMeanPositionIn3D();			
			initBead = true;
		}
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {
		
		if(!initBead) initializeBead();
		ImageProcessor imp = new FloatProcessor(imageProcessor.getWidth(), imageProcessor.getHeight());
		imp.setPixels(imageProcessor.getBuffer());
		
		double [][] beadMean3D = config.getBeadMeanPosition3D(); // [beadNo][x,y,z]		
		double [] uv = new double[1];
		
		SimpleMatrix pMatrix = config.getGeometry().getProjectionMatrix(imageIndex).computeP();
		// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough seraching]]
		double [][][] beadPosition2D = config.getBeadPosition2D();
						
		//for (int i=WeightBearingBeadPositionBuilder.currentBeadNo; i>= WeightBearingBeadPositionBuilder.currentBeadNo; i--){			
		for (int i=WeightBearingBeadPositionBuilder.currentBeadNo; i>= 0; i--){
						
			if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
				
				uv = compute2DCoordinates(beadMean3D[i], pMatrix);
				
				imp.setValue(2);
				
				// find bead location if registered by txt: state 1
				if (beadPosition2D[imageIndex][i][2] == 1){
					imp.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10));
					imp.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10));
					
					//imp.drawString(i + " (state:"+ (int) beadPosition2D[imageIndex][i][2] + ")", (int) beadPosition2D[imageIndex][i][0], (int) beadPosition2D[imageIndex][i][1] - 10);
					imp.drawString(i + "", (int) beadPosition2D[imageIndex][i][0], (int) beadPosition2D[imageIndex][i][1] - 10);
				} else {
					
					// offset loop
					for (double tmpOffset = 1.1; tmpOffset >= 0.1; tmpOffset -= 0.2) {
								
						offset = tmpOffset;
						// START hough if no registered bead
						FixedCircleHoughSpace houghBead = new FixedCircleHoughSpace(1.0, 1.0, imp.getWidth(), imp.getHeight(), radiusOfBeads);
						double scale = 1.0 / (stop - start); 
						
						for (int py = (int) (uv[1]-distance);  py < (int) (uv[1]+distance); py++){
							for (int px = (int) (uv[0]-distance); px < (int) (uv[0]+distance); px++){
								
								if (px < 0 || py < 0 || Math.sqrt(Math.pow(px-uv[0], 2)+Math.pow(py-uv[1], 2))>distance)
									continue;
								
								double value = imp.getPixelValue(px, py);

								if (value > stop) value = stop;	// if value is bigger than max, value=max
								value -= start;
								if (value > 0) {
									value *= scale;
									//System.out.println("projecting " + value + " at " + imp.getPixelValue(i, j));
									houghBead.fill(px, py, value);					
								}
							}
						}
						// get sampled hough spaces
						ImageProcessor houghSpace = houghBead.getImagePlus().getChannelProcessor();
						
						// get large candidates
						ArrayList<PointND> beadCandidate = General.extractCandidatePoints(houghSpace, offset);
						// filter circles with less distance in between than min distance
						ArrayList<PointND> houghBeads = General.extractClusterCenter(beadCandidate, distance/2);
						
						// from here 2D/3D notation, coordinates in projection image u:=x, v:=y
						double minDistance = 1000;
						for (int n = 0; n < houghBeads.size(); n ++){
							double u = houghBeads.get(n).get(0) + 3.5;
							double v = houghBeads.get(n).get(1);
							
							if (Math.sqrt(Math.pow(u-uv[0], 2)+Math.pow(v-uv[1], 2))<minDistance)
							{
								minDistance = Math.sqrt(Math.pow(u-uv[0], 2)+Math.pow(v-uv[1], 2));
								
								beadPosition2D[imageIndex][i][0] = u;
								beadPosition2D[imageIndex][i][1] = v;
								beadPosition2D[imageIndex][i][2] = 2;  // state: detected by hough							
							}
						}
						if (minDistance<1000) {
							// detected bead closest to mean projected bead
							imp.drawOval((int)beadPosition2D[imageIndex][i][0] - radiusOfBeads, (int) beadPosition2D[imageIndex][i][1] - radiusOfBeads, 2 * radiusOfBeads, 2 * radiusOfBeads);
							System.out.println(imageIndex + "\t" + i + "\t" + beadPosition2D[imageIndex][i][0] + "\t" + beadPosition2D[imageIndex][i][1] + "\t" + 2);
							break;
						}
						
					}					
				}
				
				// mean projected bead
				imp.drawLine((int) Math.round(uv[0]-10), (int) Math.round(uv[1]), (int) Math.round(uv[0]+10), (int) Math.round(uv[1]));
				imp.drawLine((int) Math.round(uv[0]), (int) Math.round(uv[1]-10), (int) Math.round(uv[0]), (int) Math.round(uv[1]+10));
			}			
		}
		
		int projNo = config.getGeometry().getProjectionStackSize();
		// this tells if all projections are accessed 
		boolean [] fAccessed;
		boolean fUpdate = true;
		fAccessed = config.getfAccessed();
		fAccessed[imageIndex] = true;
		
		for (int i=0; i< projNo; i++)
			if (!fAccessed[i]) fUpdate = false;
		
		if (fUpdate){
			System.out.println(imageIndex + " execuate copy");
			beadBuilder.writeBeadPositionToFile();
		}
		
		return imageProcessor;
	}
	
	private double [] compute2DCoordinates(double [] point3D, SimpleMatrix pMatrix){
		
		// Compute coordinates in projection data.
		SimpleVector homogeneousPoint = SimpleOperators.multiply(pMatrix, new SimpleVector(point3D[0], point3D[1], point3D[2], 1));
		// Transform to 2D coordinates
		double coordU = homogeneousPoint.getElement(0) / homogeneousPoint.getElement(2);
		double coordV = homogeneousPoint.getElement(1) / homogeneousPoint.getElement(2);
		
		//double pxlSize = config.getGeometry().getPixelDimensionX();

		return new double [] {coordU, coordV};
	}
		
	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new FiducialMarkerDetectionTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Fiducial Marker Detection Tool";
	}

	@Override
	public void configure() throws Exception {
		setConfigured(true);
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return null;
	}

}
/*
 * Copyright (C) 2010-2014 - Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/