/**
 * 
 */
package edu.stanford.rsl.conrad.geometry.motion;

import java.util.ArrayList;
import java.util.TreeMap;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.xcat.ViconAffineTransform;
import edu.stanford.rsl.conrad.phantom.xcat.ViconMarkerBuilder;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;


/**
 * @author Jang CHOI
 *
 */
public class VICONMarkerMotionField extends SimpleMotionField {
	/**
	 * 
	 */
	private static final long serialVersionUID = -719945083192209758L;	
	protected ArrayList<ArrayList<Double>> markers;
	protected int fieldNo;

	private PointND[] beforePts = new PointND[2]; 			//===== XCAT	
	private PointND[] afterPts= new PointND[2];				//===== VICON 
	private ViconAffineTransform tmpTransformation1 = null;
	private ViconAffineTransform tmpTransformation2 = null; // for dual transformation
	private String part;
	private String splineTitle;	
	private ViconMarkerBuilder vBuilder = null;
	private double weight;
	
	private boolean staticScene = false;
	private int refProjection = 0;
	
	private static TreeMap<String, TreeMap<Integer, Double>> boneScalingMemorizer;
	
	private static TreeMap<String, TreeMap<Integer, SimpleMatrix>> boneGlobalTransformMemorizer;
	
	public VICONMarkerMotionField(String transformPart, String spTitle, ViconMarkerBuilder viconBuilder, boolean staticScene, int projReference){
		vBuilder = viconBuilder;
		warp = new IdentityTimeWarper();
		markers = vBuilder.getVICONMarkers();
		part = transformPart;
		splineTitle = spTitle;
		weight = 0;
		if (boneScalingMemorizer==null)
			boneScalingMemorizer = new TreeMap<String, TreeMap<Integer,Double>>();
		if (boneGlobalTransformMemorizer==null)
			boneGlobalTransformMemorizer = new TreeMap<String, TreeMap<Integer,SimpleMatrix>>();
		this.staticScene = staticScene;
		refProjection = projReference;
		//ctrlPnts = points;
	}
	
	private void addElementToScalingMemorizer(String part, int projectionNr, double scaling){
		if (!boneScalingMemorizer.containsKey(part))
			boneScalingMemorizer.put(part, new TreeMap<Integer,Double>());
		boneScalingMemorizer.get(part).put(projectionNr, scaling);
	}
	
	private void addElementToTransformMemorizer(String part, int projectionNr, SimpleMatrix tform){
		if (!boneGlobalTransformMemorizer.containsKey(part))
			boneGlobalTransformMemorizer.put(part, new TreeMap<Integer,SimpleMatrix>());
		boneGlobalTransformMemorizer.get(part).put(projectionNr, tform);
	}
	
	

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {		
		
		boolean fKneeGapElimination = true; // want to eliminate the gap between the knee??
		double kneeGapEliWeightKnee 	= 0.71;		// for Subj5, Static60 = 0.67
		double kneeGapEliWeightAnkle 	= 0.74;		// for Subj5, Static60 = 0.70
		
		double projections = Configuration.getGlobalConfiguration().getGeometry().getNumProjectionMatrices();
		double viconSamplingRate = Double.parseDouble(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.VICON_SAMPLING_RATE));	// 60Hz
		double viconSkip = Double.parseDouble(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.VICON_SKIP_SAMPLES)); // subj5, static60=200
		// get 248 views for 8 seconds -> 248/8 = 31 Hz (For 400 views, 47.75??)
		double projectionSamplingRate = Double.parseDouble(Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.PROJECTION_SAMPLING_RATE));
		
		if (staticScene){
			double currTime = ((double)refProjection+1.0)/(double)projections;
			fieldNo = (int) (viconSkip + projections*(warp.warpTime(currTime) - warp.warpTime(initialTime))*(viconSamplingRate/projectionSamplingRate)); // vicon field #
		}
		else{
			fieldNo = (int) (viconSkip + projections*(warp.warpTime(time) - warp.warpTime(initialTime))*(viconSamplingRate/projectionSamplingRate)); // vicon field #
		}
		
		int projectionNumber = (int) Math.round(warp.warpTime(time)*projections-1);
		
//		System.out.println("Field #:\t" + fieldNo);
//		fieldNo = 201; // subj5, static60=201, Static reference
//		fieldNo = 651; // subj2, static60=651, Static reference
		
		PointND p1 = null;
		PointND p2 = null;
		
		// for subj 5, squat 60
//		double leftAxialRotDeg = -80.0;
//		double rightAxialRotDeg = -95.0;
		// for subj 2, squat 60
		double leftAxialRotDeg = -85.0;
		double rightAxialRotDeg = -90.0;
		
		// temporary points
		PointND leftPts = new PointND();
		PointND rightPts = new PointND();		
		
	
//		// XCAT FemurTop
//		PointND[] XcatFemurTop = new PointND[2];		
//		XcatFemurTop[0] = new PointND(233.3, 259.99, -166.63); 	// LeftFemurTop
//		XcatFemurTop[1] = new PointND(57.72, 259.99, -166.63); 	// RightFemurTop		
//		// VICON FemurTop
//		PointND[] ViconHJC= new PointND[2];		
//		ViconHJC[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLHJCX()), markers.get(fieldNo).get(vBuilder.colNoLHJCY()), markers.get(fieldNo).get(vBuilder.colNoLHJCZ()));
//		ViconHJC[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRHJCX()), markers.get(fieldNo).get(vBuilder.colNoRHJCY()), markers.get(fieldNo).get(vBuilder.colNoRHJCZ()));
//		
		// additional transformation for patella
		if (splineTitle.toLowerCase().contains("l_patella") || splineTitle.toLowerCase().contains("leftpatella")){
			
			// conversion factor from "V Eijden 1986_Math model of patellofemoral joint Fig.5"
			double viconKneeFlexionAngle = markers.get(fieldNo).get(vBuilder.colNoLKneeAngleX());
			//double anglePatellaVsFemor = 0.34304*viconKneeFlexionAngle + 0.0063*Math.pow(viconKneeFlexionAngle, 2) - 3.07859*Math.pow(10,-5)*Math.pow(viconKneeFlexionAngle, 3) + 2.06145;
			//double anglePatellaVsFemor = General.toRadians(0.6912*viconKneeFlexionAngle - 1.84588);
			double anglePatellaVsFemor = General.toRadians(0.9*viconKneeFlexionAngle - 1.84588);
			
			//System.out.println("[Flexion angle]\t(" + fieldNo + ")\t" + viconKneeFlexionAngle);
			
			PointND leftPatellaRotCenter = new PointND(264.8, 264.6,-590.92);
			//PointND leftPatellaRotCenter = new PointND(264.8, 264.6,-580.92);			
			RotationMotionField leftFemurMotion = new RotationMotionField(leftPatellaRotCenter, new SimpleVector(1,0,0), anglePatellaVsFemor);
			initialPosition = leftFemurMotion.getPosition(initialPosition, 0, 1);
			
		} else if (splineTitle.toLowerCase().contains("r_patella") || splineTitle.toLowerCase().contains("rightpatella")){

			// conversion factor from "V Eijden 1986_Math model of patellofemoral joint Fig.5"
			double viconKneeFlexionAngle = markers.get(fieldNo).get(vBuilder.colNoRKneeAngleX());
			//double anglePatellaVsFemor = 0.34304*viconKneeFlexionAngle + 0.0063*Math.pow(viconKneeFlexionAngle, 2) - 3.07859*Math.pow(10,-5)*Math.pow(viconKneeFlexionAngle, 3) + 2.06145;
			//double anglePatellaVsFemor = General.toRadians(0.6912*viconKneeFlexionAngle - 1.84588);
			double anglePatellaVsFemor = General.toRadians(0.9*viconKneeFlexionAngle - 1.84588);

			PointND rightPatellaRotCenter = new PointND(21.1, 269.1,-590.92);		
			//PointND rightPatellaRotCenter = new PointND(25.1, 269.1,-590.92);			
			RotationMotionField rightFemurMotion = new RotationMotionField(rightPatellaRotCenter, new SimpleVector(1,0,0), anglePatellaVsFemor);
			initialPosition = rightFemurMotion.getPosition(initialPosition, 0, 1);
			
		}	
		
		if (part == "LeftLower") {
			beforePts[0] = new PointND(316.71, 293.02, -1011.74); 	// LeftTibiaBottom
			beforePts[1] = new PointND(274.87, 275.59, -649.16+30); 	// LeftTibiaTop
			//beforePts[1] = new PointND(274.87, 275.59, -632.25); 	// (LeftTibiaTop+LeftFemurBottom)/2 in Z
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLAJCX()), markers.get(fieldNo).get(vBuilder.colNoLAJCY()), markers.get(fieldNo).get(vBuilder.colNoLAJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ()));		
			
			if (fKneeGapElimination)
			{
				leftPts = afterPts[1]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));				
				afterPts[1].set(0, leftPts.get(0)*kneeGapEliWeightKnee + rightPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(1, leftPts.get(1)*kneeGapEliWeightKnee + rightPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(2, leftPts.get(2)*kneeGapEliWeightKnee + rightPts.get(2)*(1-kneeGapEliWeightKnee));
				
				//System.out.println("[##LKJC## "+ fieldNo +"] " + afterPts[1].get(0) + ", " + afterPts[1].get(1) + ", " + afterPts[1].get(2));
				leftPts = afterPts[0]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRAJCX()), markers.get(fieldNo).get(vBuilder.colNoRAJCY()), markers.get(fieldNo).get(vBuilder.colNoRAJCZ()));;				
				afterPts[0].set(0, leftPts.get(0)*kneeGapEliWeightAnkle + rightPts.get(0)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(1, leftPts.get(1)*kneeGapEliWeightAnkle + rightPts.get(1)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(2, leftPts.get(2)*kneeGapEliWeightAnkle + rightPts.get(2)*(1-kneeGapEliWeightAnkle));				
			}
//			if (fieldNo==651){
//				System.out.println(afterPts[1].get(0) + "\t" + afterPts[1].get(1) + "\t" + afterPts[1].get(2));
//			}
			
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			addElementToScalingMemorizer("LeftLower", projectionNumber, tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("LeftLower", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if (part == "RightLower") {
			beforePts[0] = new PointND(-28.12, 295.11, -1013.33); 	// RightTibiaBottom
			beforePts[1] = new PointND(10.90, 275.60, -650.46+30); 	// RightTibiaTop	
			//beforePts[1] = new PointND(10.90, 275.60, -632.9); 		// (RightTibiaTop+RightFemurBottom)/2 in Z
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRAJCX()), markers.get(fieldNo).get(vBuilder.colNoRAJCY()), markers.get(fieldNo).get(vBuilder.colNoRAJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));		
			
			if (fKneeGapElimination)
			{
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ())); 
				rightPts = afterPts[1];				
				afterPts[1].set(0, rightPts.get(0)*kneeGapEliWeightKnee + leftPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(1, rightPts.get(1)*kneeGapEliWeightKnee + leftPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(2, rightPts.get(2)*kneeGapEliWeightKnee + leftPts.get(2)*(1-kneeGapEliWeightKnee));
				
				//System.out.println("[##RKJC## "+ fieldNo +"] " + afterPts[1].get(0) + ", " + afterPts[1].get(1) + ", " + afterPts[1].get(2));
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLAJCX()), markers.get(fieldNo).get(vBuilder.colNoLAJCY()), markers.get(fieldNo).get(vBuilder.colNoLAJCZ())); 
				rightPts = afterPts[0];				
				afterPts[0].set(0, rightPts.get(0)*kneeGapEliWeightAnkle + leftPts.get(0)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(1, rightPts.get(1)*kneeGapEliWeightAnkle + leftPts.get(1)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(2, rightPts.get(2)*kneeGapEliWeightAnkle + leftPts.get(2)*(1-kneeGapEliWeightAnkle));
			}			
//			if (fieldNo==651){
//				System.out.println(afterPts[1].get(0) + "\t" + afterPts[1].get(1) + "\t" + afterPts[1].get(2));
//			}
			
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			addElementToScalingMemorizer("RightLower",projectionNumber,tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("RightLower", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());
			
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if (part == "LeftUpper") {
			//beforePts[0] = new PointND(264.52, 271.70, -615.34); 	// LeftFemurBottom
			beforePts[0] = new PointND(264.52, 271.70, -632.25); 	// (LeftTibiaTop+LeftFemurBottom)/2 in Z
			beforePts[1] = new PointND(233.3, 259.99, -166.63); 	// LeftFemurTop			
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLHJCX()), markers.get(fieldNo).get(vBuilder.colNoLHJCY()), markers.get(fieldNo).get(vBuilder.colNoLHJCZ()));		
			
			if (fKneeGapElimination)
			{
				leftPts = afterPts[0]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));
				
				afterPts[0].set(0, leftPts.get(0)*kneeGapEliWeightKnee + rightPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(1, leftPts.get(1)*kneeGapEliWeightKnee + rightPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(2, leftPts.get(2)*kneeGapEliWeightKnee + rightPts.get(2)*(1-kneeGapEliWeightKnee));
			}
			
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			addElementToScalingMemorizer("LeftUpper",projectionNumber,tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("LeftUpper", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());

			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if (part == "RightUpper") {
			//beforePts[0] = new PointND(22.60, 271.70, -615.34); 	// RightFemurBottom
			beforePts[0] = new PointND(22.60, 271.70, -632.9); 		// (RightTibiaTop+RightFemurBottom)/2 in Z
			beforePts[1] = new PointND(57.72, 259.99, -166.63); 	// RightFemurTop			
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRHJCX()), markers.get(fieldNo).get(vBuilder.colNoRHJCY()), markers.get(fieldNo).get(vBuilder.colNoRHJCZ()));
			
			if (fKneeGapElimination)
			{
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ())); 
				rightPts = afterPts[0];
				
				afterPts[0].set(0, rightPts.get(0)*kneeGapEliWeightKnee + leftPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(1, rightPts.get(1)*kneeGapEliWeightKnee + leftPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(2, rightPts.get(2)*kneeGapEliWeightKnee + leftPts.get(2)*(1-kneeGapEliWeightKnee));
			}	
			
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			addElementToScalingMemorizer("RightUpper",projectionNumber,tmpTransformation1.getScalingMatrix().getElement(0, 0));
			addElementToTransformMemorizer("RightUpper", projectionNumber, tmpTransformation1.getAffineTransformationMatrix());

			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
		} else if (part == "LeftDual") {
			double skinningWeightLimit = 70; // mm vertical distance from the joint (-642)		
			double jointPositionInZaxis = -642;
			
			double ZLimitTop = jointPositionInZaxis + skinningWeightLimit;
			double ZLimitBottom = jointPositionInZaxis - skinningWeightLimit;
			double Zcurrent = initialPosition.get(2);
			
			// left upper
			//beforePts[0] = new PointND(264.52, 271.70, -615.34); 	// LeftFemurBottom
			beforePts[0] = new PointND(264.52, 271.70, -632.25); 	// (LeftTibiaTop+LeftFemurBottom)/2 in Z
			beforePts[1] = new PointND(233.3, 259.99, -166.63); 	// LeftFemurTop			
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLHJCX()), markers.get(fieldNo).get(vBuilder.colNoLHJCY()), markers.get(fieldNo).get(vBuilder.colNoLHJCZ()));
			if (fKneeGapElimination)
			{
				leftPts = afterPts[0]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));
				
				afterPts[0].set(0, leftPts.get(0)*kneeGapEliWeightKnee + rightPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(1, leftPts.get(1)*kneeGapEliWeightKnee + rightPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(2, leftPts.get(2)*kneeGapEliWeightKnee + rightPts.get(2)*(1-kneeGapEliWeightKnee));
			}			
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			
			// left lower
			beforePts[0] = new PointND(316.71, 293.02, -1011.74); 	// LeftTibiaBottom
			beforePts[1] = new PointND(274.87, 275.59, -649.16+30); 	// LeftTibiaTop
			//beforePts[1] = new PointND(274.87, 275.59, -632.25); 	// (LeftTibiaTop+LeftFemurBottom)/2 in Z
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLAJCX()), markers.get(fieldNo).get(vBuilder.colNoLAJCY()), markers.get(fieldNo).get(vBuilder.colNoLAJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ()));		
			if (fKneeGapElimination)
			{
				leftPts = afterPts[1]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));				
				afterPts[1].set(0, leftPts.get(0)*kneeGapEliWeightKnee + rightPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(1, leftPts.get(1)*kneeGapEliWeightKnee + rightPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(2, leftPts.get(2)*kneeGapEliWeightKnee + rightPts.get(2)*(1-kneeGapEliWeightKnee));
				
				leftPts = afterPts[0]; 
				rightPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoRAJCX()), markers.get(fieldNo).get(vBuilder.colNoRAJCY()), markers.get(fieldNo).get(vBuilder.colNoRAJCZ()));;				
				afterPts[0].set(0, leftPts.get(0)*kneeGapEliWeightAnkle + rightPts.get(0)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(1, leftPts.get(1)*kneeGapEliWeightAnkle + rightPts.get(1)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(2, leftPts.get(2)*kneeGapEliWeightAnkle + rightPts.get(2)*(1-kneeGapEliWeightAnkle));
			}
			tmpTransformation2 = new ViconAffineTransform(beforePts, afterPts, leftAxialRotDeg);
			
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
			p2 = tmpTransformation2.getTransformedPoints(initialPosition);
			
			if (Zcurrent >= ZLimitTop)	weight = 1;
			else if (Zcurrent < ZLimitBottom)	weight = 0;
			else weight = (Zcurrent - ZLimitBottom) / (ZLimitTop - ZLimitBottom);
			
			p1.set(0, p1.get(0)*weight + p2.get(0)*(1-weight)); 
			p1.set(1, p1.get(1)*weight + p2.get(1)*(1-weight));
			p1.set(2, p1.get(2)*weight + p2.get(2)*(1-weight));
			
		} else if (part == "RightDual") {
			double skinningWeightLimit = 70; // mm vertical distance from the joint (-642)		
			double jointPositionInZaxis = -642;
			
			double ZLimitTop = jointPositionInZaxis + skinningWeightLimit;
			double ZLimitBottom = jointPositionInZaxis - skinningWeightLimit;
			double Zcurrent = initialPosition.get(2);
			
			// right upper				
			//beforePts[0] = new PointND(22.60, 271.70, -615.34); 	// RightFemurBottom
			beforePts[0] = new PointND(22.60, 271.70, -632.9); 		// (RightTibiaTop+RightFemurBottom)/2 in Z
			beforePts[1] = new PointND(57.72, 259.99, -166.63); 	// RightFemurTop
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRHJCX()), markers.get(fieldNo).get(vBuilder.colNoRHJCY()), markers.get(fieldNo).get(vBuilder.colNoRHJCZ()));		
			if (fKneeGapElimination)
			{
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ())); 
				rightPts = afterPts[0];
				
				afterPts[0].set(0, rightPts.get(0)*kneeGapEliWeightKnee + leftPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(1, rightPts.get(1)*kneeGapEliWeightKnee + leftPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[0].set(2, rightPts.get(2)*kneeGapEliWeightKnee + leftPts.get(2)*(1-kneeGapEliWeightKnee));
			}
			tmpTransformation1 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);
			
			// right lower
			beforePts[0] = new PointND(-28.12, 295.11, -1013.33); 	// RightTibiaBottom
			beforePts[1] = new PointND(10.90, 275.60, -650.46+30); 	// RightTibiaTop	
			//beforePts[1] = new PointND(10.90, 275.60, -632.9); 		// (RightTibiaTop+RightFemurBottom)/2 in Z			
			afterPts[0] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRAJCX()), markers.get(fieldNo).get(vBuilder.colNoRAJCY()), markers.get(fieldNo).get(vBuilder.colNoRAJCZ()));
			afterPts[1] = new PointND(markers.get(fieldNo).get(vBuilder.colNoRKJCX()), markers.get(fieldNo).get(vBuilder.colNoRKJCY()), markers.get(fieldNo).get(vBuilder.colNoRKJCZ()));			
			if (fKneeGapElimination)
			{
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLKJCX()), markers.get(fieldNo).get(vBuilder.colNoLKJCY()), markers.get(fieldNo).get(vBuilder.colNoLKJCZ())); 
				rightPts = afterPts[1];				
				afterPts[1].set(0, rightPts.get(0)*kneeGapEliWeightKnee + leftPts.get(0)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(1, rightPts.get(1)*kneeGapEliWeightKnee + leftPts.get(1)*(1-kneeGapEliWeightKnee));
				afterPts[1].set(2, rightPts.get(2)*kneeGapEliWeightKnee + leftPts.get(2)*(1-kneeGapEliWeightKnee));
				
				leftPts = new PointND(markers.get(fieldNo).get(vBuilder.colNoLAJCX()), markers.get(fieldNo).get(vBuilder.colNoLAJCY()), markers.get(fieldNo).get(vBuilder.colNoLAJCZ())); 
				rightPts = afterPts[0];				
				afterPts[0].set(0, rightPts.get(0)*kneeGapEliWeightAnkle + leftPts.get(0)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(1, rightPts.get(1)*kneeGapEliWeightAnkle + leftPts.get(1)*(1-kneeGapEliWeightAnkle));
				afterPts[0].set(2, rightPts.get(2)*kneeGapEliWeightAnkle + leftPts.get(2)*(1-kneeGapEliWeightAnkle));
			}	
			tmpTransformation2 = new ViconAffineTransform(beforePts, afterPts, rightAxialRotDeg);		
			
			p1 = tmpTransformation1.getTransformedPoints(initialPosition);
			p2 = tmpTransformation2.getTransformedPoints(initialPosition);
			
			if (Zcurrent >= ZLimitTop)	weight = 1;
			else if (Zcurrent < ZLimitBottom)	weight = 0;
			else weight = (Zcurrent - ZLimitBottom) / (ZLimitTop - ZLimitBottom);
			
			p1.set(0, p1.get(0)*weight + p2.get(0)*(1-weight)); 
			p1.set(1, p1.get(1)*weight + p2.get(1)*(1-weight));
			p1.set(2, p1.get(2)*weight + p2.get(2)*(1-weight));
		}
		
		return p1;
	}
	
	public TreeMap<String, TreeMap<Integer, Double>> getBoneScalingMemorizer() {
		return boneScalingMemorizer;
	}
	
	public TreeMap<String, TreeMap<Integer, SimpleMatrix>> getBoneGlobalTransformMemorizer() {
		return boneGlobalTransformMemorizer;
	}
}


/*
 * Copyright (C) 2010-2014 Jang-Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
