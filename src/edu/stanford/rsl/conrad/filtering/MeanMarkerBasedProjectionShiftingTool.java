package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.motion.WeightBearingBeadPositionBuilder;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class MeanMarkerBasedProjectionShiftingTool extends IndividualImageFilteringTool {

	/**
	 * The VOIBasedReconstructionFilter is an implementation of the backprojection which employs a volume-of-interest (VOI) to
	 * speed up reconstruction. Only voxels within the VOI will be regarded in the backprojection step. Often this can save up to 30 to 40 % in computation time
	 * as volumes are usually described as boxes but the VOI is just a cylinder.
	 * 
	 * This is initially used for the weight-bearing project. 
	 * First, it reads in marker positions and mean of the bead position in 3D. 
	 * Second, it shift the projections based on markers' mean position in 2D.
	 * It is based on akmaier's motioncompenstedvoibasedreconstructionfilter.java.
	 * 
	 * @author Jang-Hwan Choi
	 *
	 */
	private static final long serialVersionUID = 1449423447534168982L;
	
	WeightBearingBeadPositionBuilder beadBuilder;
	Configuration config = Configuration.getGlobalConfiguration();
	
	private boolean initBead = false;
	// display bead indication, horizontal & vertical lines
	private boolean isDisplay = false;
	
	public MeanMarkerBasedProjectionShiftingTool (){
		configured = true;
	}
	
	protected synchronized void initializeBead() throws Exception{
		if (!initBead){
			System.out.println("Read in initial bead positions.");
			
			beadBuilder = new WeightBearingBeadPositionBuilder();
			beadBuilder.readInitialBeadPositionFromFile();
			beadBuilder.estimateBeadMeanPositionIn3D();			
			initBead = true;
		}
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		FloatProcessor imp = new FloatProcessor(imageProcessor.getWidth(),imageProcessor.getHeight());
		imp.setPixels(imageProcessor.getBuffer());
		if(!initBead) initializeBead();
		ImageProcessor imp1 = imp.duplicate();	// original
		
		//ImageProcessor currentProjection = projection;
		SimpleMatrix mat = config.getGeometry().getProjectionMatrix(imageIndex).computeP();
				
//		 Project Matrix Printing for Matlab Optimization 
//		if (imageIndex == 0){
//			for (int j=0; j<396; j++){
//				for (int i=0; i<3; i++){			
//					if (i==0) {
//						System.out.println("projMat(:,:,"+(j+1)+")=["+mat.getElement(i, 0)+"\t"+mat.getElement(i, 1)+"\t"+mat.getElement(i, 2)+"\t"+mat.getElement(i, 3)+";");
//					}else if (i==1){
//						System.out.println("\t"+mat.getElement(i, 0)+"\t"+mat.getElement(i, 1)+"\t"+mat.getElement(i, 2)+"\t"+mat.getElement(i, 3)+";");
//					}else if (i==2){
//						System.out.println("\t"+mat.getElement(i, 0)+"\t"+mat.getElement(i, 1)+"\t"+mat.getElement(i, 2)+"\t"+mat.getElement(i, 3)+"];\n");
//					}
//				}	
//			}
//		}	
//		if (imageIndex != 0)
//			return imageProcessor.duplicate();
		
		double coordUSumInitial = 0;
		double coordVSumInitial = 0;		
		double coordUInitial = 0;
		double coordVInitial = 0;			
		
		double coordUSum = 0;
		double coordVSum = 0;		
		double coordU= 0;
		double coordV= 0;		
				
		double [][] beadMean3D = Configuration.getGlobalConfiguration().getBeadMeanPosition3D(); // [beadNo][x,y,z]		
		double [] uv = new double[1];
		
		// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough searching]]
		double [][][] beadPosition2D = Configuration.getGlobalConfiguration().getBeadPosition2D();		
		int noBeadRegistered = 0;
					
		// Error: Euclidean distance
//		double distanceReferenceToCurrentBead = 0;
		
		//for (int i=WeightBearingBeadPositionBuilder.currentBeadNo; i>= 0; i--){
		for (int i=0; i<= WeightBearingBeadPositionBuilder.currentBeadNo; i++){
			
			if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
				
				// find bead location if registered by txt: state 1
				if (beadPosition2D[imageIndex][i][2] == 1){
				
					uv = compute2DCoordinates(beadMean3D[i], mat);
					noBeadRegistered++;
					
					if (isDisplay) {
						imp1.setValue(2);
						imp1.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10));
						imp1.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10));					
						imp1.drawString("Bead " + i + " (state:"+ (int) beadPosition2D[imageIndex][i][2] + ")", (int) beadPosition2D[imageIndex][i][0], (int) beadPosition2D[imageIndex][i][1] - 10);
					}
					
					// mean bead position, time invariant projected from mean position in 3d
					coordUSumInitial = coordUSumInitial + uv[0];
					coordVSumInitial = coordVSumInitial + uv[1];
					
					// bead detected position in 2d					
					// Transform to 2D coordinates, time variant position
					coordUSum = coordUSum + beadPosition2D[imageIndex][i][0];
					coordVSum = coordVSum + beadPosition2D[imageIndex][i][1];
					
					//distanceReferenceToCurrentBead += Math.sqrt(Math.pow(uv[0]-beadPosition2D[imageIndex][i][0], 2)+Math.pow(uv[1]-beadPosition2D[imageIndex][i][1], 2));					
					
//					if (imageIndex == 0){
					if (true){
						// For matlab conversion
						// reference & current bead
//						System.out.println("xyz_r("+(i+1)+",:)=["+beadMean3D[i][0]+ "\t" + beadMean3D[i][1]+ "\t" + beadMean3D[i][2]+"];");
						System.out.println("uv_c("+(i+1)+",1:2,"+(imageIndex+1)+")=["+beadPosition2D[imageIndex][i][0]+ "\t" + beadPosition2D[imageIndex][i][1] + "];");						
					}
				}
			}			
		}
		//System.out.println("Euclidean distance\t" + imageIndex + "\t" + distanceReferenceToCurrentBead/noBeadRegistered);

		if (isDisplay) {
			for (int x=0; x< config.getGeometry().getDetectorWidth(); x+=100)
				imp1.drawLine(x, 0, x, config.getGeometry().getDetectorHeight());
			for (int y=0; y< config.getGeometry().getDetectorHeight(); y+=100)
				imp1.drawLine(0, y, config.getGeometry().getDetectorWidth(), y);
		}
				
		coordUInitial = coordUSumInitial/noBeadRegistered;
		coordVInitial = coordVSumInitial/noBeadRegistered;		
		coordU = coordUSum/noBeadRegistered;
		coordV = coordVSum/noBeadRegistered;		
				
		ImageProcessor imp2 = imp1.duplicate();	// warped
		
		double devU = coordU-coordUInitial;
		double devV = coordV-coordVInitial;				
		//Do warping 		

		for (int y=0; y<config.getGeometry().getDetectorHeight(); y++) {
		//for (int y=252; y<253; y++) {
			for (int x=0; x<config.getGeometry().getDetectorWidth(); x++) {
			//for (int x=606; x<607; x++) {
//					devU = 0;
//					devV = 0;				
				imp2.setf(x, y, (float)imp1.getInterpolatedValue(x+devU, y+devV));
				
				//System.out.println("x, y=" + x + ", " + y + "\t" + devU + ", " + devV);
				//maxDevU = Math.max(maxDevU, devU);
				//maxDevV = Math.max(maxDevV, devV);				
			}
		}
			
//			// Error estimate after transformation
//			for (int i=0; i<= WeightBearingBeadPositionBuilder.currentBeadNo; i++){
//				
//				if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
//					
//					// find bead location if registered by txt: state 1
//					if (beadPosition2D[imageIndex][i][2] == 1){
//					
//						// Projected Reference
//						uv = compute2DCoordinates(beadMean3D[i], mat);						
//						
//						// bead detected position in 2d					
//						// Transform to 2D coordinates, time variant position
//						//beadPosition2D[imageIndex][i][0];
//						//beadPosition2D[imageIndex][i][1];
//						
//						distanceReferenceToCurrentBead += Math.sqrt(Math.pow(uv[0]-(beadPosition2D[imageIndex][i][0]-devU), 2)+Math.pow(uv[1]-(beadPosition2D[imageIndex][i][1]-devV), 2));				
//						
//					}				
//				}			
//			}
//			System.out.println("Euclidean distance\t" + imageIndex + "\t" + distanceReferenceToCurrentBead/noBeadRegistered);	
			
		if (isDisplay) {
			for (int i=WeightBearingBeadPositionBuilder.currentBeadNo; i>= 0; i--){
				
				if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
					
					uv = compute2DCoordinates(beadMean3D[i], mat);				
					
					imp2.setValue(2);								
					// mean projected bead
					imp2.drawLine((int) Math.round(uv[0]-10), (int) Math.round(uv[1]), (int) Math.round(uv[0]+10), (int) Math.round(uv[1]));
					imp2.drawLine((int) Math.round(uv[0]), (int) Math.round(uv[1]-10), (int) Math.round(uv[0]), (int) Math.round(uv[1]+10));				
				}			
			}
		}
				
		Grid2D result = new Grid2D((float[]) imp2.getPixels(), imp2.getWidth(), imp2.getHeight());
		return result;
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
		IndividualImageFilteringTool clone = new MeanMarkerBasedProjectionShiftingTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Projection Shifting Using Mean Bead Position";
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
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

}
/*
 * Copyright (C) 2010-2014 - Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
