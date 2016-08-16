package edu.stanford.rsl.conrad.phantom.xcat;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.motion.CombinedBreathingHeartMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.ConstantTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.jpop.utils.UserUtil;

public class CombinedBreathingHeartScene extends WholeBodyScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4691335601630948952L;
	private BreathingScene breathing;
	private HeartScene heart;
	private Translation heartTranslation = new Translation(20,55,280);
	MotionField motionField = null;

	@Override
	public void prepareForSerialization(){
		heart.prepareForSerialization();
		breathing.prepareForSerialization();
	}
	
	@Override
	public MotionField getMotionField(){
		return motionField;
	}
	
	public CombinedBreathingHeartScene() {

	}

	@Override
	public void configure() throws Exception{
		boolean dynamic = UserUtil.queryBoolean("Create dynamic scene?");
		SurfaceBSpline catheterSpline = null;
		if ( dynamic )
		{
			double breathCycles = UserUtil.queryDouble("Number of Breathing Cyles in Scene (acquistion time * breathing rate):", 1.0);
			double heartCycles = UserUtil.queryDouble("Number of heart beats in Scene (acquisition time * heart rate):", 4.0);
	
			Configuration config = Configuration.getGlobalConfiguration();
			int dimz = config.getGeometry().getProjectionStackSize() * config.getNumSweeps();
			double [] heartStates = new double [dimz];
			double [] breathingStates = new double [dimz];
			for (int i = 0; i < dimz; i++){
				heartStates[i]= (i*heartCycles) / (dimz);
				breathingStates[i] = (i*breathCycles) / (dimz);
			}
			config.getRegistry().put(RegKeys.BREATHING_PHASES, DoubleArrayUtil.toString(breathingStates));
			config.getRegistry().put(RegKeys.HEART_PHASES, DoubleArrayUtil.toString(heartStates));
			setName("Combined Breathing and Cardiac Motion Scene");

	
			heart = new HeartScene();
			heart.heartCycles = heartCycles;
			heart.configure();
			heart.setTimeWarper(new HarmonicTimeWarper(heartCycles));
	
			breathing = new BreathingScene(false);
			breathing.configure();
			breathing.setTimeWarper(new HarmonicTimeWarper(breathCycles));
			String catheterOption = Configuration.getGlobalConfiguration().getRegistry().get(RegKeys.XCAT_ADD_HEART_CATHETER);
			if (catheterOption != null){
				// remove Heart Catheter from variants in breathing and animate it according to heart and breathing motion.
				for (int i=0; i< breathing.getVariants().size(); i++){
					TimeVariantSurfaceBSpline spline = breathing.getVariants().get(i);
					if (spline.getTitle().equals("Heart Catheter")){
						breathing.getVariants().remove(spline);
						catheterSpline = spline.getSplines().get(0);
						catheterSpline.setTitle("Heart Catheter");
						i--;
					}
				}
			}
			//createArteryTree();
			//createHeartLesions();
		} else {
			setName("Combined Breathing and Cardiac Motion Scene");
			// standard scene

			
			double refHeartPhase = UserUtil.queryDouble("Reference heart phase", 0.75);
			
			heart = new HeartScene();
			heart.configure();
			heart.setTimeWarper(new ConstantTimeWarper(refHeartPhase));
	
			breathing = new BreathingScene(false);
			breathing.configure();
			breathing.setTimeWarper(new HarmonicTimeWarper(0.0));
			
			//createArteryTree();
			//createHeartLesions();
		}
		
		
		variants.addAll(heart.getVariants());
		variants.addAll(breathing.getVariants());
		// heart has only variants...
		//splines.addAll(heart.getSplines());
		splines.addAll(breathing.getSplines());
		if (catheterSpline != null){
			TimeVariantSurfaceBSpline movingCatheter = new TimeVariantSurfaceBSpline(catheterSpline, this, breathing.getNumberOfBSplineTimePoints(), true);
			Translation inverseTranslation = getHeartTranslation().inverse();
			movingCatheter.applyTransform(inverseTranslation);
			heart.getVariants().add(movingCatheter);
			variants.add(movingCatheter);
			// todo tesselate as single scene!!!
		}
		
		createPhysicalObjects();
		// standard scene
		//min = new PointND(-62, 150, -26);
		//max = new PointND(320, 400, 426);
		// Large scene
		//min = new PointND(-162, 50, -26);
		//max = new PointND(420, 500, 626);
		// projection scene
		min = new PointND(-62, 150, 150);
		max = new PointND(320, 400, 426);
		motionField = new CombinedBreathingHeartMotionField(this, 5);
	}
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		return motionField.getPosition(initialPosition, initialTime, time);
	}

	public PointND getPosition_old(PointND initialPosition, double initialTime,
			double time) {
		PointND newPoint;
		if (initialTime != time){
			Translation inverseTranslation = getHeartTranslation().inverse();
			PointND test = initialPosition.clone();
			test.applyTransform(inverseTranslation);
			if (General.isWithinCuboid(test, heart.getMin(), heart.getMax())){
				newPoint = heart.getPosition(test, initialTime, time);
				newPoint.applyTransform(getHeartTranslation());
				newPoint.applyTransform(new Translation(breathing.getDiaphragmMotionVector(initialTime, time)));
			} else {
				newPoint = breathing.getPosition(initialPosition, initialTime, time);
			}
			return newPoint;
		} else {
			return initialPosition;
		}
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		return motionField.getPositions(initialPosition, initialTime, times);
	}
	
	
	public ArrayList<PointND> getPositions_old(PointND initialPosition,
			double initialTime, double... times) {
		Translation inverseTranslation = getHeartTranslation().inverse();
		PointND test = initialPosition.clone();
		test.applyTransform(inverseTranslation);
		if (General.isWithinCuboid(test, heart.getMin(), heart.getMax())){
			ArrayList<PointND> list = heart.getPositions(test, initialTime, times);
			for (int i = 0; i < list.size(); i++){
				PointND newPoint = list.get(i);
				newPoint.applyTransform(getHeartTranslation());
				newPoint.applyTransform(new Translation(breathing.getDiaphragmMotionVector(initialTime, times[i])));
			}
			return list;
		}
		return breathing.getPositions(initialPosition, initialTime, times);
	}

	@Override
	public PrioritizableScene tessellateScene(double time){
		PrioritizableScene scene = new PrioritizableScene();
		PrioritizableScene breathingScene = breathing.tessellateScene(time);
		PrioritizableScene heartScene = heart.tessellateScene(time);
		SimpleVector diaphragmMotion =(breathing.getDiaphragmMotionVector(0, time));
		for (PhysicalObject o: heartScene){
			o.getShape().applyTransform(getHeartTranslation());
			o.getShape().applyTransform(new Translation(diaphragmMotion));		
		}
		//System.out.println(diaphragmMotion);
		//System.out.println(heartTranslation);
		//clear();
		scene.addAll(breathingScene);
		scene.addAll(heartScene);
		return scene;
	}

	/**
	 * Renders cylinder-based regions into the atrium and the left ventricle
	 */
	public void createHeartLesions(){
		heart.createLesions(heart);
	}

	/**
	 * Creates the coronary artery tree
	 */
	public void createArteryTree(){
		heart.createArteryTree(heart);
	}

	@Override
	public String getName() {
		return "XCat Breathing Torso with Cardiac Motion";
	}

	/**
	 * @return the breathing
	 */
	public BreathingScene getBreathing() {
		return breathing;
	}

	/**
	 * @param breathing the breathing to set
	 */
	public void setBreathing(BreathingScene breathing) {
		this.breathing = breathing;
	}

	/**
	 * @return the heart
	 */
	public HeartScene getHeart() {
		return heart;
	}

	/**
	 * @param heart the heart to set
	 */
	public void setHeart(HeartScene heart) {
		this.heart = heart;
	}

	/**
	 * @param heartTranslation the heartTranslation to set
	 */
	public void setHeartTranslation(Translation heartTranslation) {
		this.heartTranslation = heartTranslation;
	}

	/**
	 * @return the heartTranslation
	 */
	public Translation getHeartTranslation() {
		return heartTranslation;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/