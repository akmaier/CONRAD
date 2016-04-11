package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.motion.CompressionMotionField;
import edu.stanford.rsl.conrad.geometry.motion.ConstantMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MixedSurfaceBSplineMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.PlanarMotionField;
import edu.stanford.rsl.conrad.geometry.motion.RotationMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.DualPhasePeriodicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceUniformCubicBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class BreathingScene extends WholeBodyScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4935602839572704230L;
	// Chest expansion [mm]
	private double chestExpansion = 10;
	private HashMap<String, RotationMotionField> rotationsMap = new HashMap<String, RotationMotionField>();
	private double diaphragmMovement = 24;
	private int numberOfBSplineTimePoints = 10;
	private MotionField chestMotion;
	private PointND diaphragmMin = new PointND(-50,-55,190);
	private PointND diaphragmMax = new PointND(301,348,250);
	private PointND diaphragmTop = new PointND(75,250,250);
	private DualPhasePeriodicTimeWarper dualPhaseWarper = new DualPhasePeriodicTimeWarper(2,3);
	private boolean ignoreArms = true;

	@Override
	public MotionField getMotionField(){
		return chestMotion;
	}
	
	
	public BreathingScene (){


	}

	@Override
	public void configure(){
		//min = new PointND(-32, 96, -26);
		//max = new PointND(370, 400, 426);
		//large
		//min = new PointND(-162, 50, -26);
		//max = new PointND(420, 500, 626);
		//min = new PointND(-62, 100, 100);
		//max = new PointND(320, 450, 426);

		double breathCycles = 1.0;
		try {
			breathCycles = UserUtil.queryDouble("Number of Breathing Cycles in Scene (acquistion time * breathing rate):", 1.0);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		setName("Breathing Scene");
		init();
		String [] matches = {
				// right ribs
				".*RRIB1",
				".*RRIB2",
				".*RRIB3",
				".*RRIB4",
				".*RRIB5",
				".*RRIB6",
				".*RRIB7",
				".*RRIB8",
				".*RRIB9",
				".*RRIB10",
				".*RRIB11",
				".*RRIB12",
				// left ribs
				".*LRIB1",
				".*LRIB2",
				".*LRIB3",
				".*LRIB4",
				".*LRIB5",
				".*LRIB6",
				".*LRIB7",
				".*LRIB8",
				".*LRIB9",
				".*LRIB10",
				".*LRIB11",
				".*LRIB12"
		};
		String [] constants = {
				"vert",
				"backbone",
				//"r_cartilage",
				"collar",
				"humerus",
				"ulna",
				"radius",
				"scapula",
				"hip"
		};
		String [] arms = {
				"humerus",
				"collar",
				"ulna",
				"radius",
				"rightarm",
				"leftarm"
		};
		PointND [] points = {
				// right ribs
				new PointND(116, 272, 422),
				new PointND(108.6, 285, 417),
				new PointND(103, 305, 400),
				new PointND(107, 317, 371),
				new PointND(108.6, 324.2, 348.5),
				new PointND(114.2, 326.0, 324.4),
				new PointND(116.0, 331.6, 296.6),
				new PointND(119.7, 333.5, 270.6),
				new PointND(123.4, 327.9, 246.5),
				new PointND(123.4, 324.2, 224.5),
				new PointND(121.6, 320.5, 196.5),
				new PointND(125.3, 313.0, 159.4),
				// left ribs
				new PointND(151.2, 268.6, 424.5),
				new PointND(162.4, 290.8, 417.1),
				new PointND(158.7, 303.8, 400.4),
				new PointND(153.1, 314.9, 374.4),
				new PointND(147.5, 316.8, 352.2),
				new PointND(151.2, 327.9, 326.2),
				new PointND(156.8, 333.5, 298.4),
				new PointND(158.7, 333.5, 274.3),
				new PointND(160.5, 331.6, 248.4),
				new PointND(162.4, 327.9, 222.4),
				new PointND(160.5, 311.2, 153.8),
				new PointND(160.5, 311.2, 153.8)
		};
		
		String catheterOption = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ADD_HEART_CATHETER);
		if (catheterOption != null){
			// get diameter
			double diameter = Double.parseDouble(catheterOption);
			// get aorta spline and clone it
			SurfaceBSpline heartCatheter = null;
			SurfaceBSpline aorta = null;
			for (SurfaceBSpline spline : splines){
				if (spline.getTitle().contains("AORTA")){
					aorta = spline;
					heartCatheter = new SurfaceUniformCubicBSpline(spline);
					heartCatheter.setTitle("Heart Catheter");
				}
			}
			String catheterTranslation = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ADD_HEART_CATHETER_TIP_TRANSLATION);
			SimpleVector translationTip = new SimpleVector(0, 0, 0, 1);
			if (catheterTranslation != null){
				translationTip.setVectorSerialization(catheterTranslation);
			}
			int uPoints = heartCatheter.getNumberOfUPoints();
			int vPoints = heartCatheter.getNumberOfVPoints();
			for (int i=0; i < uPoints; i++) {
				SimpleVector currentTranslation = translationTip.getSubVec(0, 3);
				currentTranslation.multiplyBy(Math.max(1.0 - (((double)i)/translationTip.getElement(3)),0));
				PointND center = new PointND(0,0,0);
				int point_index = 0;
				for (int j=0; j < vPoints; j++) {
					point_index = i*vPoints+j;
					for (int d=0; d < 3; d++) center.set(d, heartCatheter.getControlPoints().get(point_index).get(d) / (double) vPoints + center.get(d));
				}
				for (int j=0; j < vPoints; j++) {
					point_index = i*vPoints+j;
					SimpleVector translation = new SimpleVector(heartCatheter.getControlPoints().get(point_index).getAbstractVector());
					translation.add(center.getAbstractVector().negated());
					double length = translation.normL2();
					translation.normalizeL2();
					double scale = diameter/2;
					if (length < diameter/2) {
						scale = length;
					} 
					translation.multiplyBy(scale);
					translation.add(currentTranslation);
					for (int d=0; d < 3; d++) 
						heartCatheter.getControlPoints().get(point_index).getAbstractVector().setElementValue(d, translation.getElement(d) + center.get(d));
				}
			}
			splines.add(heartCatheter);
		}
		
		// build map of rotations
		for (int i = splines.size()-1; i >= 0; i--){
			for (int j =0; j < points.length; j++) {
				if (splines.get(i).getTitle().matches(matches[j])){
					RotationMotionField rot = null;
					if (matches[j].contains("RRIB")){
						rot = createRotationMotionField(points[j], new SimpleVector(-1,1,0), splines.get(i));
					} else {
						rot = createRotationMotionField(points[j], new SimpleVector(-1,-1,0), splines.get(i));
					}
					rotationsMap.put(matches[j]+"(\\D.*)?", rot);
					System.out.println("Created rotation for " + splines.get(i).getTitle());
				}
			}
		}
		// apply rotations to the matching splines
		ArrayList<ArrayList<PointND>> chestMotionField = new ArrayList<ArrayList<PointND>>();
		for (int t=0; t<numberOfBSplineTimePoints; t++){
			chestMotionField.add(new ArrayList<PointND>());
		}
		for (int i = splines.size()-1; i >= 0; i--){		
			if (splines.get(i).getTitle().contains("diaphragm") || splines.get(i).getTitle().contains("LIVER")){
				System.out.println("Creating time-variant spline for " + splines.get(i).getTitle());
				CompressionMotionField compression = new CompressionMotionField(diaphragmMin, diaphragmMax, new SimpleVector(0,0,-diaphragmMovement));
				compression.setTimeWarper(dualPhaseWarper);
				TimeVariantSurfaceBSpline tSpline = new TimeVariantSurfaceBSpline(splines.get(i), compression, numberOfBSplineTimePoints, true);
				for (int j = 0; j<tSpline.getControlPoints(0).size(); j++){
					PointND p = tSpline.getControlPoints(0).get(j);
					PointND p2 = tSpline.getControlPoints(tSpline.getNumberOfTimePoints()/2).get(j);
					// Only regard points that actually moved in the motion field.
					if (p.euclideanDistance(p2) > 0) {
						for (int t=0; t<numberOfBSplineTimePoints; t++){					
							chestMotionField.get(t).add(tSpline.getControlPoints(t).get(j));
						}
					}
				}
			}
			boolean constantTest = false;
			for (String s: constants){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					constantTest = true;
				}
			}
			if (constantTest){
				System.out.println("Creating time-constant spline for " + splines.get(i).getTitle());
				TimeVariantSurfaceBSpline tSpline = new TimeVariantSurfaceBSpline(splines.get(i), new ConstantMotionField(), numberOfBSplineTimePoints, true);
				for (int t=0; t<numberOfBSplineTimePoints; t++){
					chestMotionField.get(t).addAll(tSpline.getControlPoints(t));
				}
			}
			for (String match: rotationsMap.keySet()){
				if (splines.get(i).getTitle().matches(match)){
					System.out.println("Creating time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rotationsMap.get(match), numberOfBSplineTimePoints, true));
					splines.remove(i);
				}
			}
		}
		chestMotion = getSceneMotion(numberOfBSplineTimePoints, chestMotionField, 5);
		chestMotion.setTimeWarper(new IdentityTimeWarper());
		for (int i = splines.size()-1; i >= 0; i--){
			if (splines.get(i).getTitle().contains("STERN")){
				System.out.println("Creating time-variant planar-motion spline for " + splines.get(i).getTitle());
				PlanarMotionField planar = new PlanarMotionField(chestMotion, new SimpleVector(1,0,0));
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), planar, numberOfBSplineTimePoints, true));
				splines.remove(i);
			}
		}
		chestMotion = getSceneMotion(numberOfBSplineTimePoints, chestMotionField, 5);
		chestMotion.setTimeWarper(new IdentityTimeWarper());
		for (int i = splines.size()-1; i >= 0; i--){
			boolean constantTest = false;
			for (String s: constants){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					constantTest = true;
				}
			}
			boolean ignoreArmPart =  false;
			if (this.ignoreArms) {
				for (String s: arms){
					if (splines.get(i).getTitle().toLowerCase().contains(s)){
						ignoreArmPart = true;
					}
				}
			}
			if (!ignoreArmPart) {
				if (!constantTest){
					//System.out.println(splines.get(i).getTitle() + " " + !exclude(splines.get(i).getTitle()));
					if (!exclude(splines.get(i).getTitle())){
						System.out.println("Creating time-variant spline for " + splines.get(i).getTitle());
						variants.add(new TimeVariantSurfaceBSpline(splines.get(i), chestMotion, numberOfBSplineTimePoints, true));
					}
					splines.remove(i);
				}
			} else {
				splines.remove(i);
			}
		}
		setTimeWarper(new HarmonicTimeWarper(breathCycles));
		
		createPhysicalObjects();
		min = new PointND(-62, 150, 50);
		max = new PointND(320, 400, 426);
		chestMotion = new MixedSurfaceBSplineMotionField(variants, splines, 5);
	}

	private RotationMotionField createRotationMotionField(PointND center, SimpleVector axis, SurfaceBSpline spline){
		RotationMotionField rot = new RotationMotionField(center, axis, computeAngle(spline, chestExpansion));
		rot.setTimeWarper(dualPhaseWarper);
		return rot;
	}

	private double computeAngle (AbstractShape spline, double delta){
		// dimension in short axis direction (RSA)
		double rangeY = spline.getMax().get(1) - spline.getMin().get(1);
		// dimension in Z direction
		double rangeZ = spline.getMax().get(2) - spline.getMin().get(2);
		// length of the rib projected onto the y/z plane
		double length = Math.sqrt(Math.pow(rangeY, 2) + Math.pow(rangeZ, 2));
		// desired angle
		double angle = Math.acos((rangeY) / length) - Math.acos((rangeY + delta) / length);
		//System.out.println(rangeY + " " + rangeZ + " " + length + " " +General.toDegrees(angle));
		return angle;
	}

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		return chestMotion.getPosition(initialPosition, initialTime, time);
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		return chestMotion.getPositions(initialPosition, initialTime, times);
	}

	@Override
	public void setTimeWarper(TimeWarper warper){
		super.setTimeWarper(warper);
		chestMotion.setTimeWarper(warper);
	}

	/**
	 * Returns the Motion Vector of the top of the diaphragm from initialTime to time
	 * @param initialTime the initial time
	 * @param time the time
	 * @return the motion vector
	 */
	public SimpleVector getDiaphragmMotionVector(double initialTime, double time){
		SimpleVector diaphragmVectorOne = diaphragmTop.getAbstractVector();
		//diaphragmVectorOne.multiplyBy(dualPhaseWarper.warpTime(warper.warpTime(initialTime)));
		CompressionMotionField compression = new CompressionMotionField(diaphragmMin, diaphragmMax, new SimpleVector(0,0,-diaphragmMovement));
		compression.setTimeWarper(dualPhaseWarper);
		SimpleVector diaphragmVectorTwo = compression.getPosition(diaphragmTop, initialTime, warper.warpTime(time)).getAbstractVector();
		//SimpleVector diaphragmVectorTwo = new SimpleVector(0, 0, -getDiaphragmMovement());
		//diaphragmVectorTwo.multiplyBy(dualPhaseWarper.warpTime(warper.warpTime(time)));
		SimpleVector revan = SimpleOperators.subtract(diaphragmVectorTwo, diaphragmVectorOne);
		//System.out.println("Time t0: " + initialTime + " t1: " + time+ " Heart Vector:" + revan);
		//System.out.println("Time t0: " + initialTime + " t1: " + time+ " Initial:" + diaphragmVectorTwo);
		return revan;
	}
	
	public SimpleVector getDiaphragmMotionVector_old(double initialTime, double time){
		SimpleVector diaphragmVectorOne = new SimpleVector(0, 0, -getDiaphragmMovement());
		diaphragmVectorOne.multiplyBy(dualPhaseWarper.warpTime(warper.warpTime(initialTime)));
		SimpleVector diaphragmVectorTwo = new SimpleVector(0, 0, -getDiaphragmMovement());
		diaphragmVectorTwo.multiplyBy(dualPhaseWarper.warpTime(warper.warpTime(time)));
		return SimpleOperators.subtract(diaphragmVectorTwo, diaphragmVectorOne);
	}

	/**
	 * @return the diaphragmMovement
	 */
	public double getDiaphragmMovement() {
		return diaphragmMovement;
	}

	/**
	 * @param diaphragmMovement the diaphragmMovement to set
	 */
	public void setDiaphragmMovement(double diaphragmMovement) {
		this.diaphragmMovement = diaphragmMovement;
	}

	/**
	 * @return the numberOfBSplineTimePoints
	 */
	public int getNumberOfBSplineTimePoints() {
		return numberOfBSplineTimePoints;
	}

	/**
	 * @param numberOfBSplineTimePoints the numberOfBSplineTimePoints to set
	 */
	public void setNumberOfBSplineTimePoints(int numberOfBSplineTimePoints) {
		this.numberOfBSplineTimePoints = numberOfBSplineTimePoints;
	}

	/**
	 * @return the diaphragmMin
	 */
	public PointND getDiaphragmMin() {
		return diaphragmMin;
	}

	/**
	 * @param diaphragmMin the diaphragmMin to set
	 */
	public void setDiaphragmMin(PointND diaphragmMin) {
		this.diaphragmMin = diaphragmMin;
	}

	/**
	 * @return the diaphragmMax
	 */
	public PointND getDiaphragmMax() {
		return diaphragmMax;
	}

	/**
	 * @param diaphragmMax the diaphragmMax to set
	 */
	public void setDiaphragmMax(PointND diaphragmMax) {
		this.diaphragmMax = diaphragmMax;
	}

	@Override
	public String getName() {
		return "XCat Breathing Torso";
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/