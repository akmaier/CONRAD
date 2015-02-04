package edu.stanford.rsl.conrad.filtering;

import java.util.ArrayList;
import java.util.Random;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionUtil;
import edu.stanford.rsl.conrad.geometry.motion.WeightBearingBeadPositionBuilder;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.phantom.xcat.DynamicSquatScene;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class ThinPlateSplinesBasedProjectionWarpingToolForXCAT extends IndividualImageFilteringTool {

	/**
	 * Warping projections using thin plate splines for marker based motion correction 
	 * 
	 * http://elonen.iki.fi/code/tpsdemo/index.html,
	 * Based mostly on "Approximation Methods for Thin Plate Spline Mappings and Principal Warps" by Gianluca Donato and Serge Belongie, 2002.
	 * (http://cseweb.ucsd.edu/~sjb/pami_tps.pdf)
	 * 
	 * @author Jang-Hwan Choi
	 */
	
	private static final long serialVersionUID = -3377571138470875648L;	
	
	Configuration config = Configuration.getGlobalConfiguration();
	
	// display bead indication, horizontal & vertical lines	
	private boolean isCornerIncluded = true;

	private MotionField motionField;
	private ArrayList<TimeVariantSurfaceBSpline> variants = new ArrayList<TimeVariantSurfaceBSpline>();
	
	private int[][] markers; // [marker #][variant#, control point #]
	private boolean fAddMarkerErr = false;
	private boolean initMotionField = false;
	private boolean isDisplay = false;
	private boolean fScaling = true;
	
	public ThinPlateSplinesBasedProjectionWarpingToolForXCAT (){
		configured = true;
	}
	
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
//		if(imageIndex != 1) return imageProcessor;
		
		FloatProcessor imp = new FloatProcessor(imageProcessor.getWidth(),imageProcessor.getHeight());
		imp.setPixels(imageProcessor.getBuffer());
		if(!initMotionField) initialize();
		ImageProcessor imp1 = imp.duplicate();	// original

		SimpleMatrix pMatrix = config.getGeometry().getProjectionMatrix(imageIndex).computeP();		
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
				int noMarkers = 8; // limit of no of markers for 2 leg, used to be 8(default)
				markers = new int[noMarkers][2];
				
				// define markers from ctrlpoints
//				double zRotationCenter = 440.7783;
//				double zRangeFromCenter = 90;
//				int markerIdx = 0;				
				
//				for (int i = 0; i < noMarkers; i++){					
	//				if (i%2==0)
	//					markers[i][0]=68; // right leg
	//				else 
	//					markers[i][0]=69; // left leg
	//			}
	//
				/* Mark distribution evenly all over, used for paper*/
	//			markers[0][1]=183; //ctrl pnt #
	//			markers[1][1]=180; 
	//			markers[2][1]=246; 
	//			markers[3][1]=252; 
	//			markers[4][1]=195; 
	//			markers[5][1]=195; 
	//			markers[6][1]=207; 
	//			markers[7][1]=222;
	//			markers[8][1]=237;
	//			markers[9][1]=207;
	//			markers[10][1]=192; 
	//			markers[11][1]=204; 
	//			markers[12][1]=210; 
	//			markers[13][1]=231; 
				
				markers[0][0]=68; // variant #
				markers[1][0]=68; 
				markers[2][0]=68; 
				markers[3][0]=68; 
				markers[4][0]=69; 
				markers[5][0]=69; 
				markers[6][0]=69; 
	
				/* Mark distribution only top and bottom, back and forth*/
	//			markers[0][0]=68;
	//			markers[1][0]=68;
	//			markers[2][0]=68;
	//			markers[3][0]=68;
	//			markers[4][0]=69;
	//			markers[5][0]=69;								
	//			markers[6][0]=69;
	//			markers[7][0]=69;
	//			
	//			markers[0][1]=181; //ctrl pnt #
	//			markers[1][1]=184; 
	//			markers[2][1]=268; 
	//			markers[3][1]=265;
	//			markers[4][1]=175; 
	//			markers[5][1]=178;				
	//			markers[6][1]=263;
	//			markers[7][1]=251;
				/* END of Mark distribution only top and bottom*/				
				
				/* Mark distribution only top and bottom, only frontal part*/
	//			markers[0][0]=68;
	//			markers[1][0]=68;
	//			markers[2][0]=68;
	//			markers[3][0]=68;
	//			markers[4][0]=69;
	//			markers[5][0]=69;								
	//			markers[6][0]=69;
	//			markers[7][0]=69;				
	//			
	//			markers[0][1]=182; //ctrl pnt #
	//			markers[1][1]=179; 
	//			markers[2][1]=255; 
	//			markers[3][1]=265;
	//			markers[4][1]=182; 
	//			markers[5][1]=178;				
	//			markers[6][1]=262;
	//			markers[7][1]=261;				
				/* END of Mark distribution only top and bottom,  only frontal part*/
				
				/* Mark distribution only top and bottom, 1 patella*/
				markers[0][0]=68;
				markers[1][0]=68;
				markers[2][0]=68;
				markers[3][0]=68;
				markers[4][0]=69;
				markers[5][0]=69;
				markers[6][0]=69;
				markers[7][0]=69;
	//			markers[1][0]=68;
	//			markers[6][0]=69;
				
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
			//centerTranlation = null;
		}
		
		double [] times = new double [config.getGeometry().getNumProjectionMatrices()];
		for (int i=0; i< times.length; i++){
			times [i]= ((double)i) / config.getGeometry().getNumProjectionMatrices();
		}
		
		double coordUInitial = 0;
		double coordVInitial = 0;
				
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
			
			ArrayList<PointND> ctrlPoints = variants.get(markers[lp][0]).getControlPoints(imageIndex); // control point at time p;
			pointReference = new PointND(ctrlPoints.get(markers[lp][1]).get(0), ctrlPoints.get(markers[lp][1]).get(1), ctrlPoints.get(markers[lp][1]).get(2));
			
			// locate XCAT into voxel location 
			if (centerTranlation != null){ 
				pointReferenceInitial.getAbstractVector().subtract(centerTranlation); 
				pointReference.getAbstractVector().subtract(centerTranlation);
			}
			
			// Compute coordinates in projection data.
			SimpleVector homogeneousPointInitial = SimpleOperators.multiply(pMatrix, new SimpleVector(pointReferenceInitial.get(0), pointReferenceInitial.get(1), pointReferenceInitial.get(2), 1));
			// Transform to 2D coordinates
			uInitial = homogeneousPointInitial.getElement(0) / homogeneousPointInitial.getElement(2);
			vInitial = homogeneousPointInitial.getElement(1) / homogeneousPointInitial.getElement(2);
						
			// Compute coordinates in projection data. & add measurement error
			SimpleVector homogeneousPoint= SimpleOperators.multiply(pMatrix, 
					new SimpleVector(
							pointReference.get(0)+getRandomDouble(-1, 1), 
							pointReference.get(1)+getRandomDouble(-1, 1), 
							pointReference.get(2)+getRandomDouble(-1, 1), 1));
			// Transform to 2D coordinates
			uCurrent = homogeneousPoint.getElement(0) / homogeneousPoint.getElement(2);
			vCurrent = homogeneousPoint.getElement(1) / homogeneousPoint.getElement(2);			
			
			beadMean3D[lp][0] = pointReferenceInitial.get(0);
			beadMean3D[lp][1] = pointReferenceInitial.get(1);
			beadMean3D[lp][2] = pointReferenceInitial.get(2);			
			
			// [projection #][bead #][u, v, state[0: initial, 1: registered, 2: updated by hough searching]]
			beadPosition2D[imageIndex][lp][0] = uCurrent;
			beadPosition2D[imageIndex][lp][1] = vCurrent;
			beadPosition2D[imageIndex][lp][2] = 1;

		}
						
		double [] uv = new double[1];			
		int noBeadRegistered = 0;
		
		double [][] xy1 = new double [18][2]; // original
		double [][] xy2 = new double [18][2]; // warped	(mapped to the mean), control points, reference
		
		double [][] xy1_hat = new double [18][2]; // original
		double [][] xy2_hat = new double [18][2]; // original
		
		double distanceReferenceToCurrentBead = 0;
		
		for (int i = markers.length-1; i >= 0; i--){
			
			if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
				
				uv = compute2DCoordinates(beadMean3D[i], pMatrix);				
				
				// find bead location if registered by txt: state 1
				if (beadPosition2D[imageIndex][i][2] == 1){
					
					noBeadRegistered++;
					
					if (isDisplay) {
						imp1.setValue(2);
						imp1.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10));
						imp1.drawLine((int) Math.round(beadPosition2D[imageIndex][i][0]-10), (int) Math.round(beadPosition2D[imageIndex][i][1]+10), (int) Math.round(beadPosition2D[imageIndex][i][0]+10), (int) Math.round(beadPosition2D[imageIndex][i][1]-10));					
						//imp1.drawString("Bead " + i + " (state:"+ (int) beadPosition2D[imageIndex][i][2] + ")", (int) beadPosition2D[imageIndex][i][0], (int) beadPosition2D[imageIndex][i][1] - 10);
					}
					
					xy1[noBeadRegistered-1][0] = beadPosition2D[imageIndex][i][0]; 
					xy1[noBeadRegistered-1][1] = beadPosition2D[imageIndex][i][1];
					
					xy2[noBeadRegistered-1][0] = uv[0]; 
					xy2[noBeadRegistered-1][1] = uv[1];
					
				}else if (imageIndex != 0 && imageIndex != config.getGeometry().getNumProjectionMatrices()-1) {
					
					if (beadPosition2D[imageIndex-1][i][2] == 1 && beadPosition2D[imageIndex+1][i][2] == 1){
						
						noBeadRegistered++;
						
						double xMean = (beadPosition2D[imageIndex-1][i][0] + beadPosition2D[imageIndex-1][i][0])/2;
						double yMean = (beadPosition2D[imageIndex+1][i][1] + beadPosition2D[imageIndex+1][i][1])/2;
						
						if (isDisplay) {
							imp1.setValue(2);
							imp1.drawLine((int) Math.round(xMean-10), (int) Math.round(yMean-10), (int) Math.round(xMean+10), (int) Math.round(yMean+10));
							imp1.drawLine((int) Math.round(xMean-10), (int) Math.round(yMean+10), (int) Math.round(xMean+10), (int) Math.round(yMean-10));					
							//imp1.drawString("Bead " + i + " (state:"+ "M)", (int) xMean, (int) yMean - 10);
						}
						
						xy1[noBeadRegistered-1][0] = xMean; 
						xy1[noBeadRegistered-1][1] = yMean;
						
						xy2[noBeadRegistered-1][0] = uv[0]; 
						xy2[noBeadRegistered-1][1] = uv[1];
						
					}					
				}
								
				// mean projected bead
//				imp1.drawLine((int) Math.round(uv[0]-10), (int) Math.round(uv[1]), (int) Math.round(uv[0]+10), (int) Math.round(uv[1]));
//				imp1.drawLine((int) Math.round(uv[0]), (int) Math.round(uv[1]-10), (int) Math.round(uv[0]), (int) Math.round(uv[1]+10));
			}			
		}
		
		if (isDisplay) {
			for (int x=0; x< config.getGeometry().getDetectorWidth(); x+=100)
				imp1.drawLine(x, 0, x, config.getGeometry().getDetectorHeight());
			for (int y=0; y< config.getGeometry().getDetectorHeight(); y+=100)
				imp1.drawLine(0, y, config.getGeometry().getDetectorWidth(), y);
		}
		
		if (isCornerIncluded) {
			xy1[noBeadRegistered+0][0] = 0; 
			xy1[noBeadRegistered+0][1] = 0;		
			xy2[noBeadRegistered+0][0] = 0; 
			xy2[noBeadRegistered+0][1] = 0;
			
			xy1[noBeadRegistered+1][0] = 0; 
			xy1[noBeadRegistered+1][1] = config.getGeometry().getDetectorHeight();		
			xy2[noBeadRegistered+1][0] = 0; 
			xy2[noBeadRegistered+1][1] = config.getGeometry().getDetectorHeight();
			
			xy1[noBeadRegistered+2][0] = config.getGeometry().getDetectorWidth(); 
			xy1[noBeadRegistered+2][1] = 0;		
			xy2[noBeadRegistered+2][0] = config.getGeometry().getDetectorWidth(); 
			xy2[noBeadRegistered+2][1] = 0;
			
			xy1[noBeadRegistered+3][0] = config.getGeometry().getDetectorWidth(); 
			xy1[noBeadRegistered+3][1] = config.getGeometry().getDetectorHeight();		
			xy2[noBeadRegistered+3][0] = config.getGeometry().getDetectorWidth(); 
			xy2[noBeadRegistered+3][1] = config.getGeometry().getDetectorHeight();
			
			noBeadRegistered = noBeadRegistered + 4;
		}
		
		double minX = Double.MAX_VALUE;		
		double maxX = 0;
		double minY = Double.MAX_VALUE;
		double maxY = 0;
		double c = 0;
		if (fScaling) {			//----- scaling to reduce condition # of A matrix 
			for (int i=0; i< noBeadRegistered; i++){
				minX = Math.min(minX, xy1[i][0]);
				maxX = Math.max(maxX, xy1[i][0]);
				minY = Math.min(minY, xy1[i][1]);
				maxY = Math.max(maxY, xy1[i][1]);
			}
			c = Math.max(maxX - minX, maxY - minY);
	
			for (int i=0; i< noBeadRegistered; i++){
				xy1_hat[i][0] = (xy1[i][0] - minX)/c; 
				xy1_hat[i][1] = (xy1[i][1] - minY)/c;
				
				xy2_hat[i][0] = (xy2[i][0] - minX)/c; 
				xy2_hat[i][1] = (xy2[i][1] - minY)/c;
			}
		}else {
			xy1_hat = xy1;
			xy2_hat = xy2;
		}				

		ImageProcessor imp2 = imp1.duplicate();	// warped	

		/*
		 * A*x = b  
		 * Matrix A = (n + 3) * (n + 3);
		 * n (noBeadRegistered + 4): # of control points + 4 corner points (assume corner points are static)
		 */
		
		int n = noBeadRegistered + 3;
						
		SimpleMatrix A = new SimpleMatrix (n, n);
		SimpleVector x_x = new SimpleVector (n);
		SimpleVector x_y = new SimpleVector (n);
		SimpleVector b_x = new SimpleVector (n);
		SimpleVector b_y = new SimpleVector (n);
				
		double rij = 0;
		double valA = 0;
		double valb_x = 0;
		double valb_y = 0;
		
		//Matrix L formation
		//alpha: mean of distances between control points' xy-projections) is a constant only present on the diagonal of K
		//lambda: TPS smoothing regularization coefficient
		
		double alpha = 0.0;
		double lambda = 0.3; //1.6, [XCAT 0.3 best]
		for (int i=0; i<noBeadRegistered; i++){		// i= # of row
			for (int j=i; j<noBeadRegistered; j++){	// j= # of column
				alpha += Math.sqrt(Math.pow(xy2_hat[i][0]-xy2_hat[j][0], 2)+Math.pow(xy2_hat[i][1]-xy2_hat[j][1], 2));				
			}
		}
		alpha = alpha/Math.pow(noBeadRegistered, 2);
		
		for (int i=0; i<n; i++){		// i= # of row
			for (int j=i; j<n; j++){	// j= # of column
				if (i<3 && j<3) valA = 0;				
				else if (i>=3 && j>=3 && i==j) {
					valA = Math.pow(alpha, 2)*lambda;
					//valA = lambda;
					//if(imageIndex < 10) System.out.println("Regularization = " + valA + ", lambda= " + lambda);
				}
				else if (i==0 && j>=0) valA = 1;
				else if (i==1 && j>=3) valA = xy1_hat[j-3][0];
				else if (i==2 && j>=3) valA = xy1_hat[j-3][1];				
				else {
					rij = Math.pow(xy1_hat[j-3][0]-xy1_hat[i-3][0], 2) + Math.pow(xy1_hat[j-3][1]-xy1_hat[i-3][1], 2);
					if (rij == 0) valA = 0; 
					else valA = rij*Math.log(rij);										
				}
				
				A.setElementValue(i, j, valA);
				A.setElementValue(j, i, valA);			
			}	
			
			if (i<3) {
				valb_x = 0; valb_y = 0;
			}else {
//				valb_x = xy2_hat[i-3][0]-xy1_hat[i-3][0];
//				valb_y = xy2_hat[i-3][1]-xy1_hat[i-3][1];
				valb_x = xy2[i-3][0]-xy1[i-3][0];
				valb_y = xy2[i-3][1]-xy1[i-3][1];
//				if (imageIndex > 150 && imageIndex < 170)
//					System.out.println("Idx" + imageIndex + ",Elevation" + (i-3) + ": " + valb_x + "---" + valb_y);
			}
			
			b_x.setElementValue(i, valb_x);			
			b_y.setElementValue(i, valb_y);
		}
				
		//System.out.println(imageIndex + "\tA condition number=" + A.conditionNumber(MatrixNormType.MAT_NORM_L1));
		//System.out.println("A condition number=" + A.conditionNumber(MatrixNormType.MAT_NORM_LINF));
		
		x_x = Solvers.solveLinearSysytemOfEquations(A, b_x);
		x_y = Solvers.solveLinearSysytemOfEquations(A, b_y);		
		
		if (fScaling) {
			//----- pixel space coefficients a, b scaling back
			double tmpA0 = x_x.getElement(0)-x_x.getElement(1)*(minX/c)-x_x.getElement(2)*(minY/c);
			for (int j=0; j<noBeadRegistered; j++){
				tmpA0 -= Math.log(c)*2*x_x.getElement(j+3)*(Math.pow(xy1_hat[j][0], 2) + Math.pow(xy1_hat[j][1], 2)); 
			}
			x_x.setElementValue(0, tmpA0);		
			tmpA0 = x_y.getElement(0)-x_y.getElement(1)*(minX/c)-x_y.getElement(2)*(minY/c);
			for (int j=0; j<noBeadRegistered; j++){
				tmpA0 -= Math.log(c)*2*x_y.getElement(j+3)*(Math.pow(xy1_hat[j][0], 2) + Math.pow(xy1_hat[j][1], 2)); 
			}
			x_y.setElementValue(0, tmpA0);
			
			x_x.setElementValue(1, x_x.getElement(1)/c);
			x_y.setElementValue(1, x_y.getElement(1)/c);
			x_x.setElementValue(2, x_x.getElement(2)/c);				
			x_y.setElementValue(2, x_y.getElement(2)/c);
			
			for (int i=3; i<n; i++){		
				x_x.setElementValue(i, x_x.getElement(i)/Math.pow(c, 2));
				x_y.setElementValue(i, x_y.getElement(i)/Math.pow(c, 2));
			}
			//----- pixel space coefficients a, b scaling back end
		}		
		
		double devU = 0;
		double devV = 0;				
		//Do warping 		
		//if (imageIndex == 0) {
			for (int y=0; y<config.getGeometry().getDetectorHeight(); y++) {
			//for (int y=252; y<253; y++) {
				for (int x=0; x<config.getGeometry().getDetectorWidth(); x++) {
				//for (int x=606; x<607; x++) {
					devU = x_x.getElement(0) + x_x.getElement(1)*x + x_x.getElement(2)*y;
					devV = x_y.getElement(0) + x_y.getElement(1)*x + x_y.getElement(2)*y;
					for (int i=0; i<noBeadRegistered; i++){
						rij = Math.pow(xy1[i][0]-x, 2) + Math.pow(xy1[i][1]-y, 2);
						if (rij > 0) {
							devU += x_x.getElement(i+3)*rij*Math.log(rij);
							devV += x_y.getElement(i+3)*rij*Math.log(rij);
						}
					}
					
//					devU = 0;
//					devV = 0;
					
					imp2.setf(x, y, (float)imp1.getInterpolatedValue(x-devU, y-devV));
					
//					System.out.println("x, y=" + x + ", " + y + "\t" + devU + ", " + devV);
					//maxDevU = Math.max(maxDevU, devU);
					//maxDevV = Math.max(maxDevV, devV);				
				}
			}
			
			// Error estimate after transformation
			for (int i = markers.length-1; i >= 0; i--){
				
				if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
					
					// find bead location if registered by txt: state 1
					if (beadPosition2D[imageIndex][i][2] == 1){
					
						// Projected Reference
						uv = compute2DCoordinates(beadMean3D[i], pMatrix);						
						double x = uv[0];
						double y = uv[1];
						// bead detected position in 2d					
						// Transform to 2D coordinates, time variant position
						//beadPosition2D[imageIndex][i][0];
						//beadPosition2D[imageIndex][i][1];
						
						devU = x_x.getElement(0) + x_x.getElement(1)*x + x_x.getElement(2)*y;
						devV = x_y.getElement(0) + x_y.getElement(1)*x + x_y.getElement(2)*y;
						for (int j=0; j<noBeadRegistered; j++){
							rij = Math.pow(xy1[j][0]-x, 2) + Math.pow(xy1[j][1]-y, 2);
							if (rij > 0) {
								devU += x_x.getElement(j+3)*rij*Math.log(rij);
								devV += x_y.getElement(j+3)*rij*Math.log(rij);
							}
						}
						
						distanceReferenceToCurrentBead += Math.sqrt(Math.pow(uv[0]-(beadPosition2D[imageIndex][i][0]+devU), 2)+Math.pow(uv[1]-(beadPosition2D[imageIndex][i][1]+devV), 2));		
						
					}				
				}			
			}
			//System.out.println("Euclidean distance\t" + imageIndex + "\t" + distanceReferenceToCurrentBead/noBeadRegistered);			
			
		//}
			
		if (isDisplay) {
			for (int i = markers.length-1; i >= 0; i--){
			
				if (beadMean3D[i][0] != 0 || beadMean3D[i][1] != 0 || beadMean3D[i][2] != 0){ // assume bead 3d is registered.
					
					uv = compute2DCoordinates(beadMean3D[i], pMatrix);				
					
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
		
	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new ThinPlateSplinesBasedProjectionWarpingToolForXCAT();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Projection Warping Using Thin Plate Splines For XCAT";
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
		return "@article{Bookstein89-PWT,\n" +
		"  author={Bookstein FL},\n" +
		"  title={Principal warps: Thin-plate splines and the decomposition of deformations},\n" +
		"  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},\n" +
		"  volume={11},\n" +
		"  number={6},\n" +
		"  pages={568-585},\n" +
		"  year={1989}\n" +
		"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Bookstein FL. Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on Pattern Analysis and Machine Intelligence 11(6):568-585, 1989.";
	}

}
/*
 * Copyright (C) 2010-2014 - Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
