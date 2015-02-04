package edu.stanford.rsl.conrad.filtering;

import java.util.ArrayList;
import java.util.Random;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionUtil;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.xcat.DynamicSquatScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class MeanMarkerBasedProjectionShiftingToolForXCAT extends IndividualImageFilteringTool {

	/**
	 * This version of the reconstruction algorithm
	 *  applies the motion field stored in 4D_SPLINE_LOCATION before the backprojection.
	 * 
	 * @author Jang-Hwan Choi
	 *
	 */
	private static final long serialVersionUID = -4287558757650497769L;
		
	Configuration config = Configuration.getGlobalConfiguration();
		
	public MeanMarkerBasedProjectionShiftingToolForXCAT (){
		configured = true;
	}
	
	private MotionField motionField;
	private ArrayList<TimeVariantSurfaceBSpline> variants = new ArrayList<TimeVariantSurfaceBSpline>();
	
	private int[][] markers; // [marker #][variant#, control point #]
	private boolean fAddMarkerErr = false;
	private boolean initMotionField = false;
	private boolean isDisplay = false;
		
	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		motionField = null;
	}

	protected synchronized void initialize() throws Exception{
		if (!initMotionField){			
			if (motionField == null) {
				System.out.println("loading Motion Field at image index:" + imageIndex);
				motionField = MotionUtil.get4DSpline();
				initMotionField = true;				
			}			
		}
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {

//		 Project Matrix Printing for Matlab Optimization 
//		if (imageIndex == 0){
//			for (int j=0; j<248; j++){
//				SimpleMatrix PM = config.getGeometry().getProjectionMatrix(j).computeP();
//				for (int i=0; i<3; i++){			
//					if (i==0) {
//						System.out.println("projMat(:,:,"+(j+1)+")=["+PM.getElement(i, 0)+"\t"+PM.getElement(i, 1)+"\t"+PM.getElement(i, 2)+"\t"+PM.getElement(i, 3)+";");
//					}else if (i==1){
//						System.out.println("\t"+PM.getElement(i, 0)+"\t"+PM.getElement(i, 1)+"\t"+PM.getElement(i, 2)+"\t"+PM.getElement(i, 3)+";");
//					}else if (i==2){
//						System.out.println("\t"+PM.getElement(i, 0)+"\t"+PM.getElement(i, 1)+"\t"+PM.getElement(i, 2)+"\t"+PM.getElement(i, 3)+"];\n");
//					}
//				}	
//			}
//		}	
//		if (imageIndex != 0)
//			return imageProcessor.duplicate();
		
		
		FloatProcessor imp = new FloatProcessor(imageProcessor.getWidth(),imageProcessor.getHeight());
		imp.setPixels(imageProcessor.getBuffer());
		
		if(!initMotionField) initialize();
		ImageProcessor imp1 =imp.duplicate();	// original
				
		int p = imageIndex;		
		SimpleMatrix mat = config.getGeometry().getProjectionMatrix(p).computeP();
		SimpleVector centerTranlation = null;
		SimpleVector xcatCenter = null;
				
		String [] xcatParts = {	
				"leftleg",
				"rightleg"				
		};
		
		int referenceFrameNo = 0;
		
		if (motionField instanceof DynamicSquatScene){
			DynamicSquatScene phantom = (DynamicSquatScene) motionField;			
			variants = phantom.getVariants();
					
			if (markers == null){
				int noMarkers = 8; //14 limit of no of markers for 2 leg, used to be 8(default)
				markers = new int[noMarkers][2];
				
				// define markers from ctrlpoints
//				double zRotationCenter = 440.7783;
//				double zRangeFromCenter = 90;
//				int markerIdx = 0;				

				/* Variant # & Control Point # for XCAT paper*/
//				for (int i = 0; i < noMarkers; i++){					
//					if (i%2==0)
//						markers[i][0]=68; // right leg, variant #, used for xcat paper
//					else 
//						markers[i][0]=69; // left leg
//				}
//
				/* Mark distribution evenly all over, used for paper*/
//				markers[0][1]=183; //ctrl pnt #
//				markers[1][1]=180; 
//				markers[2][1]=246; 
//				markers[3][1]=252; 
//				markers[4][1]=195; 
//				markers[5][1]=195; 
//				markers[6][1]=207; 
//				markers[7][1]=222;
//				markers[8][1]=237;
//				markers[9][1]=207;
//				markers[10][1]=192; 
//				markers[11][1]=204; 
//				markers[12][1]=210; 
//				markers[13][1]=231; 
				

				/* Mark distribution only top and bottom, back and forth*/
//				markers[0][0]=68;
//				markers[1][0]=68;
//				markers[2][0]=68;
//				markers[3][0]=68;
//				markers[4][0]=69;
//				markers[5][0]=69;								
//				markers[6][0]=69;
//				markers[7][0]=69;
//				
//				markers[0][1]=181; //ctrl pnt #
//				markers[1][1]=184; 
//				markers[2][1]=268; 
//				markers[3][1]=265;
//				markers[4][1]=175; 
//				markers[5][1]=178;				
//				markers[6][1]=263;
//				markers[7][1]=251;
				/* END of Mark distribution only top and bottom*/				
				
//				/* Mark distribution only top and bottom, only frontal part*/
//				markers[0][0]=68;
//				markers[1][0]=68;
//				markers[2][0]=68;
//				markers[3][0]=68;
//				markers[4][0]=69;
//				markers[5][0]=69;								
//				markers[6][0]=69;
//				markers[7][0]=69;				
//				
//				markers[0][1]=182; //ctrl pnt #
//				markers[1][1]=179; 
//				markers[2][1]=255; 
//				markers[3][1]=265;
//				markers[4][1]=182; 
//				markers[5][1]=178;				
//				markers[6][1]=262;
//				markers[7][1]=261;				
//				/* END of Mark distribution only top and bottom,  only frontal part*/
				
				/* Mark distribution only top and bottom, 1 patella*/
				markers[0][0]=68;
				markers[1][0]=68;
				markers[2][0]=68;
				markers[3][0]=68;
				markers[4][0]=69;
				markers[5][0]=69;
				markers[6][0]=69;
				markers[7][0]=69;
//				markers[1][0]=68;
//				markers[6][0]=69;
				
				markers[0][1]=181; //ctrl pnt #
				markers[1][1]=207;
				markers[2][1]=268; 
				markers[3][1]=265;
				markers[4][1]=175; 
				markers[5][1]=178;
				markers[6][1]=251; 
				markers[7][1]=204; 
//				markers[1][1]=184;
//				markers[6][1]=263;
				/* END of Mark distribution only top and bottom*/
				
//				int j=0;
//				for (int i=162; i <= 280; i=i+2){					
//					markers[j][0]=68;
//					markers[j][1]=i;
//					markers[j+1][0]=69;
//					markers[j+1][1]=i;
//					j=j+2;
//				}
				
//				for (int i = variants.size()-1; i >= 0; i--){
//					
//					for (String s: xcatParts){					
//						if (variants.get(i).getTitle().toLowerCase().contains(s)){
//							ArrayList<PointND> ctrlPoints = variants.get(i).getControlPoints(referenceFrameNo); // control point at time 0;
//							
//							int firstj = 0;
//							int noPointsInROI = 0;
//							for (int j = ctrlPoints.size()-1; j >= 0; j=j-1){
//								// if control points are in the bead implanting range
//								if (ctrlPoints.get(j).get(2) > zRotationCenter-zRangeFromCenter && ctrlPoints.get(j).get(2) < zRotationCenter+zRangeFromCenter){
//									noPointsInROI++;
//									if (firstj ==0) firstj=j;
//								}
//								
//								if (imageIndex == 0){
//									if (ctrlPoints.get(j).get(2) > zRotationCenter-zRangeFromCenter && ctrlPoints.get(j).get(2) < zRotationCenter+zRangeFromCenter)
//										System.out.println("[" + i + "," +  j + "]\t" + ctrlPoints.get(j).get(0) + "\t" + ctrlPoints.get(j).get(1) + "\t" + ctrlPoints.get(j).get(2));
//								}
//							}
//							
//							int stepSize = (int)Math.floor(noPointsInROI/(noMarkers));
//							if (markerIdx>0) noMarkers=noMarkers*2;//if the second spline, then double the marker index scope
//							double residue = 0;							
//							for (int j = firstj; j >= 0 && markerIdx<noMarkers; j=j-stepSize){	
//								if (ctrlPoints.get(j).get(2) < zRotationCenter+zRangeFromCenter){
//									residue = Math.ceil(markerIdx/2); 
//									if (residue==1) j=+(int) (stepSize*0.5);									
//									
//									if (imageIndex == 0){
//										System.out.println("[" + i + "," +  j + "]\t" + ctrlPoints.get(j).get(0) + "\t" + ctrlPoints.get(j).get(1) + "\t" + ctrlPoints.get(j).get(2));
//									}
//									markers[markerIdx][0]=i; // variant #
//									markers[markerIdx][1]=j; // ctrl pnt #
//									markerIdx++;
//								}
//							}
//						}
//					}														
//				}
			}
			//System.out.println("Marker #:\t" + markers.length);

			
			String translationString = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING);
			if (translationString != null){
				// Center b/w RKJC & LKJC: -292.6426  211.7856  440.7783
				// XCAT Center by min & min: -177.73999504606988, 179.8512744259873, 312.19713254613583					
				String [] values = translationString.split(", ");
				//centerTranlation = new SimpleVector(Double.parseDouble(values[0]), Double.parseDouble(values[1]), Double.parseDouble(values[2]));
				
				//centerTranlation = new SimpleVector(-292.6426, 211.7856, 440.7783); for subj 5, static 60
				// for subj2 (-(GLOBAL_TRANSLATION_4DPHANTOM_PROJECTION_RENDERING-XCAT))
				xcatCenter = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);				
				centerTranlation = new SimpleVector(xcatCenter.getElement(0)-Double.parseDouble(values[0]), 
						xcatCenter.getElement(1)-Double.parseDouble(values[1]), 
						xcatCenter.getElement(2)-Double.parseDouble(values[2]));								
			} else {
				centerTranlation = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);
			}	
//			centerTranlation = null;
		}
		
		double [] times = new double [config.getGeometry().getNumProjectionMatrices()];
		for (int i=0; i< times.length; i++){
			times [i]= ((double)i) / config.getGeometry().getNumProjectionMatrices();
		}
		
		double coordUSumInitial = 0;
		double coordVSumInitial = 0;				
		double coordUInitial = 0;
		double coordVInitial = 0;
				
		double coordUSum = 0;
		double coordVSum = 0;
		
		double coordU= 0;
		double coordV= 0;		
		
		double uInitial=0;
		double vInitial=0;
		double uCurrent=0;
		double vCurrent=0;
		
		PointND pointReferenceInitial;
		PointND pointReference;
		
		double [][] beadMean3D = new double[markers.length][3]; //config.getBeadMeanPosition3D(); // [beadNo][x,y,z]
		// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough searching]]
		double [][][] beadPosition2D = new double[config.getGeometry().getNumProjectionMatrices()][markers.length][3]; // config.getBeadPosition2D();
		
		for (int lp = markers.length-1; lp >= 0; lp--){
			if (markers[lp][0]==0 && markers[lp][1]==0) continue;
			ArrayList<PointND> ctrlPointsInitial = variants.get(markers[lp][0]).getControlPoints(referenceFrameNo); // control point at time 0;
			pointReferenceInitial = new PointND(ctrlPointsInitial.get(markers[lp][1]).get(0), ctrlPointsInitial.get(markers[lp][1]).get(1), ctrlPointsInitial.get(markers[lp][1]).get(2));
			
			ArrayList<PointND> ctrlPoints = variants.get(markers[lp][0]).getControlPoints(p); // control point at time p;
			pointReference = new PointND(ctrlPoints.get(markers[lp][1]).get(0), ctrlPoints.get(markers[lp][1]).get(1), ctrlPoints.get(markers[lp][1]).get(2));
			
			// locate XCAT into voxel location 
			if (centerTranlation != null){ 
				pointReferenceInitial.getAbstractVector().subtract(centerTranlation); 
				pointReference.getAbstractVector().subtract(centerTranlation);
			}
			
			// Compute coordinates in projection data.
			SimpleVector homogeneousPointInitial = SimpleOperators.multiply(mat, new SimpleVector(pointReferenceInitial.get(0), pointReferenceInitial.get(1), pointReferenceInitial.get(2), 1));
			// Transform to 2D coordinates
			uInitial = homogeneousPointInitial.getElement(0) / homogeneousPointInitial.getElement(2);
			vInitial = homogeneousPointInitial.getElement(1) / homogeneousPointInitial.getElement(2);
			coordUSumInitial = coordUSumInitial + uInitial;
			coordVSumInitial = coordVSumInitial + vInitial;
			//coordWSumInitial = coordWSumInitial + homogeneousPointInitial.getElement(2);
			
			// Compute coordinates in projection data. & add measurement error
			SimpleVector homogeneousPoint= SimpleOperators.multiply(mat, 
					new SimpleVector(
							pointReference.get(0)+getRandomDouble(-1, 1), 
							pointReference.get(1)+getRandomDouble(-1, 1), 
							pointReference.get(2)+getRandomDouble(-1, 1), 1));
			// Transform to 2D coordinates
			uCurrent = homogeneousPoint.getElement(0) / homogeneousPoint.getElement(2);
			vCurrent = homogeneousPoint.getElement(1) / homogeneousPoint.getElement(2);
			coordUSum = coordUSum + uCurrent;
			coordVSum = coordVSum + vCurrent;
			//coordWSum = coordWSum + homogeneousPoint.getElement(2);
			
			beadMean3D[lp][0] = pointReferenceInitial.get(0);
			beadMean3D[lp][1] = pointReferenceInitial.get(1);
			beadMean3D[lp][2] = pointReferenceInitial.get(2);			
			
			// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough searching]]
			beadPosition2D[imageIndex][lp][0] = uCurrent;
			beadPosition2D[imageIndex][lp][1] = vCurrent;
			beadPosition2D[imageIndex][lp][2] = 1;

			// For matlab conversion
			// reference & current bead
			if (imageIndex == 0)
				System.out.println("xyz_r("+(lp+1)+",:)=["+beadMean3D[lp][0]+ "\t" + beadMean3D[lp][1]+ "\t" + beadMean3D[lp][2]+"];");
			if (true)
				System.out.println("uv_c("+(lp+1)+",1:2,"+(imageIndex+1)+")=["+beadPosition2D[imageIndex][lp][0]+ "\t" + beadPosition2D[imageIndex][lp][1] + "];");					
		}
		
		coordUInitial = coordUSumInitial/markers.length;
		coordVInitial = coordVSumInitial/markers.length;		
		coordU = coordUSum/markers.length;
		coordV = coordVSum/markers.length;		
						
		ImageProcessor imp2 = imp1.duplicate();	// warped
		
		double devU = coordU-coordUInitial;
		double devV = coordV-coordVInitial;		
		
//		System.out.println("devU, devV\t" + imageIndex + "\t" + devU + "\t" + devV);
//		devU = 0;
//		devV = 0;	
		 		
		for (int y=0; y<config.getGeometry().getDetectorHeight(); y++) {		
			for (int x=0; x<config.getGeometry().getDetectorWidth(); x++) {
				imp2.setf(x, y, (float)imp1.getInterpolatedValue(x+devU, y+devV));									
			}
		}
					
		if (isDisplay) {	// display reference (initial) markers
			imp2.setValue(2);				
			imp2.setFont(new java.awt.Font ("Serif", java.awt.Font.BOLD, 15));
			
			double [] uvInitial = new double[2];   
			// display individual reference markers 
			for (int lp = markers.length-1; lp >= 0; lp--){
				if (markers[lp][0]==0 && markers[lp][1]==0) continue;
//				if (markers[lp][0]!=69) continue; 

				// individual bead of current (x)				
				uCurrent = beadPosition2D[imageIndex][lp][0] - devU;
				vCurrent = beadPosition2D[imageIndex][lp][1] - devV;				
				imp2.drawLine((int) Math.round(uCurrent-10), (int) Math.round(vCurrent+10), (int) Math.round(uCurrent+10), (int) Math.round(vCurrent-10));
				imp2.drawLine((int) Math.round(uCurrent-10), (int) Math.round(vCurrent-10), (int) Math.round(uCurrent+10), (int) Math.round(vCurrent+10));
				//imp2.drawString("Bead " + lp + " ("+ (int) markers[lp][0] + "," + markers[lp][1] + ")", (int) uCurrent, (int) vCurrent - 10);
				if (imageIndex == 0)
					System.out.println("Bead " + (lp + 1)+ " ("+ (int) markers[lp][0] + "," + markers[lp][1] + ")");
				
				imp2.drawString("[" + (lp+1) + "]", (int) uCurrent + 10, (int) vCurrent, java.awt.Color.WHITE); // for paper
				
				// individual bead of initial (+)
//				uvInitial = compute2DCoordinates(beadMean3D[lp], mat);
//				imp2.drawLine((int) Math.round(uvInitial[0]-10), (int) Math.round(uvInitial[1]), (int) Math.round(uvInitial[0]+10), (int) Math.round(uvInitial[1]));
//				imp2.drawLine((int) Math.round(uvInitial[0]), (int) Math.round(uvInitial[1]-10), (int) Math.round(uvInitial[0]), (int) Math.round(uvInitial[1]+10));				
			}
			
			coordU -= devU; 
			coordV -= devV;			
//			// mean projected bead of current (x)
//			imp2.drawLine((int) Math.round(coordU-10), (int) Math.round(coordV+10), (int) Math.round(coordU+10), (int) Math.round(coordV-10));
//			imp2.drawLine((int) Math.round(coordU-10), (int) Math.round(coordV-10), (int) Math.round(coordU+10), (int) Math.round(coordV+10));
//			imp2.drawOval((int) Math.round(coordU-5), (int)Math.round(coordV-5), 10, 10);			
//			
//			// mean projected bead of initial (+)
//			imp2.drawLine((int) Math.round(coordUInitial-10), (int) Math.round(coordVInitial), (int) Math.round(coordUInitial+10), (int) Math.round(coordVInitial));
//			imp2.drawLine((int) Math.round(coordUInitial), (int) Math.round(coordVInitial-10), (int) Math.round(coordUInitial), (int) Math.round(coordVInitial+10));
//			imp2.drawOval((int) Math.round(coordUInitial-5), (int)Math.round(coordVInitial-5), 10, 10);
			
			// grid
			for (int x=0; x< config.getGeometry().getDetectorWidth(); x+=100)
				imp2.drawLine(x, 0, x, config.getGeometry().getDetectorHeight());
			for (int y=0; y< config.getGeometry().getDetectorHeight(); y+=100)
				imp2.drawLine(0, y, config.getGeometry().getDetectorWidth(), y);
		}
			
		// Error estimate after transformation
		double [] uv = new double[1];
		double distanceReferenceToCurrentBead = 0;
		for (int i = markers.length-1; i >= 0; i--){

			// find bead location if registered by txt: state 1
			if (beadPosition2D[imageIndex][i][2] == 1){
			
				// Projected Reference
				uv = compute2DCoordinates(beadMean3D[i], mat);						
				
				// bead detected position in 2d					
				// Transform to 2D coordinates, time variant position
				//beadPosition2D[imageIndex][i][0];
				//beadPosition2D[imageIndex][i][1];
				
				distanceReferenceToCurrentBead += Math.sqrt(Math.pow(uv[0]-(beadPosition2D[imageIndex][i][0]-devU), 2)+Math.pow(uv[1]-(beadPosition2D[imageIndex][i][1]-devV), 2));				
				
			}	
		}
		// Error metric
		// System.out.println("Euclidean distance\t" + imageIndex + "\t" + distanceReferenceToCurrentBead/markers.length);
							
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
		IndividualImageFilteringTool clone = new MeanMarkerBasedProjectionShiftingToolForXCAT();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Projection Shifting Using Mean Bead Position For XCAT";
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
	
	/**
	 * @param start of random number
	 * @param end of random number
	 * @return double random number in a range assigned
	 */
	public double getRandomDouble(double start, double end){				
		if ( start > end ) {
			throw new IllegalArgumentException("range setting err");
		}		
		Random random = new Random();
		double range = end - start;
		// nextDouble generate double number between 0 to 1
		if (fAddMarkerErr) return (range * random.nextDouble() + start);
		else return 0;
	}

}

/*
 * Copyright (C) 2010-2014 - Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
