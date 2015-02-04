package edu.stanford.rsl.conrad.phantom.xcat;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.PointBasedMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Transformable;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.jpop.utils.UserUtil;


public class CoronaryScene extends XCatScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8901686267381868405L;
	protected ArrayList<SurfaceBSpline> arteries;
	private boolean renderHeartBody = false;
	private MotionField heartMotionField;
	ScaleRotate heartRotation;

	public CoronaryScene(){
		
	}

	@Override
	public void configure() throws Exception{
		double heartCycles = UserUtil.queryDouble("Number of heart beats in Scene (acquisition time * heart rate):", 4.0);
		Configuration config = Configuration.getGlobalConfiguration();
		int dimz = config.getGeometry().getProjectionStackSize() * config.getNumSweeps();
		double [] heartStates = new double [dimz];
		for (int i = 0; i < dimz; i++){
			heartStates[i]= (i*heartCycles) / (dimz);
		}
		config.getRegistry().put(RegKeys.HEART_PHASES, DoubleArrayUtil.toString(heartStates));

		warper = new HarmonicTimeWarper(heartCycles);
		init();
		createArteryTree(this);
		createPhysicalObjects();
	}

	public ArrayList<SurfaceBSpline> readHeartState(int state){
		try {
			return SurfaceBSpline.readSplinesFromFile(XCatDirectory + "/heart_base" + state + ".nrb");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * 
	 * @param referenceMotion
	 */
	public void createArteryTree(MotionField referenceMotion){
		try {
			arteries = SurfaceBSpline.readSplinesFromFile(XCatDirectory + "/artery_tree.nrb");
			for (SurfaceBSpline spline:arteries){
				applyTransform(spline, heartRotation);
				System.out.println("Converting spline " + spline.getTitle() + " to time-variant mode.");
				TimeVariantSurfaceBSpline tspline = new TimeVariantSurfaceBSpline(spline, referenceMotion, 27, true);
				variants.add(tspline);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	

	public void init(){
		setName("Heart Scene");
		SimpleMatrix rotation = SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(
				Rotations.createBasicZRotationMatrix(General.toRadians(165)), SimpleOperators.multiplyMatrixProd(
						Rotations.createBasicXRotationMatrix(General.toRadians(45)), 
						Rotations.createBasicYRotationMatrix(General.toRadians(45)))),
						Rotations.createBasicZRotationMatrix(General.toRadians(50)));
		
		rotation = SimpleOperators.multiplyMatrixProd( 
			SimpleOperators.multiplyMatrixProd(SimpleOperators.multiplyMatrixProd(
				Rotations.createBasicZRotationMatrix(General.toRadians(165)), SimpleOperators.multiplyMatrixProd(
						Rotations.createBasicXRotationMatrix(General.toRadians(45)), 
						Rotations.createBasicYRotationMatrix(General.toRadians(45)))),
						Rotations.createBasicZRotationMatrix(General.toRadians(50))),
						Rotations.createBasicZRotationMatrix(General.toRadians(180)));
		heartRotation = new ScaleRotate(rotation.multipliedBy(0.85));
		String rotationString = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.XCAT_HEART_ROTATION);
		if (rotationString != null){
			String [] values = rotationString.split(", ");
			rotation = SimpleOperators.multiplyMatrixProd(
						Rotations.createBasicXRotationMatrix(General.toRadians(Double.parseDouble(values[0]))), SimpleOperators.multiplyMatrixProd(
								Rotations.createBasicYRotationMatrix(General.toRadians(Double.parseDouble(values[1]))), 
								Rotations.createBasicZRotationMatrix(General.toRadians(Double.parseDouble(values[2])))));
			heartRotation = new ScaleRotate(rotation.multipliedBy(Double.parseDouble(values[3])));
		}
		
		ArrayList<ArrayList<SurfaceBSpline>> completeSplineList = new ArrayList<ArrayList<SurfaceBSpline>>();
		int splineNum = 0;
		for (int k = 0; k < 27; k++){
			ArrayList<SurfaceBSpline> list = readHeartState(k);
			splineNum = list.size();
			completeSplineList.add(list);
		}

		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (int i=0; i<splineNum; i++){
			ArrayList<SurfaceBSpline> surfaces = new ArrayList<SurfaceBSpline>();
			for (int k = 0; k < 27; k++){
				surfaces.add(completeSplineList.get(k).get(i));
			}
			TimeVariantSurfaceBSpline variant = new TimeVariantSurfaceBSpline(surfaces);

			max.updateIfHigher(variant.getMax());
			min.updateIfLower(variant.getMin());
			variants.add(variant);
		}

		heartMotionField = new PointBasedMotionField(variants, 5);
		if (!renderHeartBody) {
			variants.clear();
		}

		//applyTransform(heartRotation);
	}

	private void updateMinAndMax(){
		max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		for (int i=0; i <variants.size(); i++){
			TimeVariantSurfaceBSpline variant = variants.get(i);
			max.updateIfHigher(variant.getMax());
			min.updateIfLower(variant.getMin());
		}
	}

	private void applyTransform(Transform t){
		SimpleVector center = SimpleOperators.add(min.getAbstractVector(), max.getAbstractVector()).dividedBy(2);
		Translation backShift = new Translation(center.clone());
		center.negate();
		Translation centerShift = new Translation(center);
		for (int i=0; i <variants.size(); i++){
			TimeVariantSurfaceBSpline variant = variants.get(i);
			variant.applyTransform(centerShift);
			variant.applyTransform(t);
			variant.applyTransform(backShift);
		}
		heartMotionField = new PointBasedMotionField(variants, 5);
		updateMinAndMax();
	}

	private void applyTransform(Transformable shape, Transform t){
		SimpleVector center = SimpleOperators.add(min.getAbstractVector(), max.getAbstractVector()).dividedBy(2);
		Translation backShift = new Translation(center.clone());
		center.negate();
		Translation centerShift = new Translation(center);
		shape.applyTransform(centerShift);
		shape.applyTransform(t);
		shape.applyTransform(backShift);
	}

	@Override
	public float[] getBinaryRepresentation() {
		int totalsize = 2;
		float [][] timeVariants = new float [variants.size()][];
		for (int i = 0; i<variants.size();i++){
			timeVariants[i] = variants.get(i).getBinaryRepresentation();
			totalsize += timeVariants[i].length;
		}
		float [] binary = new float [totalsize + timeVariants.length + timeVariants.length];
		binary[0] = BSpline.BSPLINECOLLECTION;
		binary[1] = totalsize;
		binary[2] = timeVariants.length;
		int index = 3;
		for (int i = 0; i<timeVariants.length;i++){
			for (int j=0;j<timeVariants[i].length; j++){
				binary[index]= timeVariants[i][j];
				index ++;
			}
		}
		for (int i = 0; i<timeVariants.length;i++){
			binary[index] = XCatScene.getSplinePriorityLUT().get(variants.get(i).getTitle());
		}
		for (int i = 0; i<timeVariants.length;i++){
			binary[index] = (float) XCatMaterialGenerator.generateFromMaterialName(XCatScene.getSplineNameMaterialNameLUT().get(variants.get(i).getTitle())).getDensity();
		}
		return binary;
	}
	
	public PrioritizableScene tessellateScene(double time){
		// remove previous state scene objects.
		PrioritizableScene scene = new PrioritizableScene();
		clearObjectsOnly();

		TessellationThread[] threads = new TessellationThread[CONRAD.getNumberOfThreads()];
		Iterator<TimeVariantSurfaceBSpline> list = Collections.synchronizedList(variants).iterator();

		for (int splineNum = 0; splineNum < CONRAD.getNumberOfThreads(); splineNum++) {
			threads[splineNum] = new TessellationThread(list, warper.warpTime(time));
		}
		ParallelThreadExecutor executor = new ParallelThreadExecutor(threads);
		try {
			executor.execute();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		for (int splineNum = 0; splineNum < CONRAD.getNumberOfThreads(); splineNum++) {
			for (AbstractShape s: threads[splineNum].getObjects()) {
				add(scene, s, s.getName());
			}
		}
		return scene;
	}

	@Override
	public ArrayList<SurfaceBSpline> getSplines() {
		return readHeartState(0);
	}

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		return heartMotionField.getPosition(initialPosition, initialTime, time);
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		return heartMotionField.getPositions(initialPosition, initialTime, times);
	}

	@Override
	public void setTimeWarper(TimeWarper warper){
		super.setTimeWarper(warper);
		heartMotionField.setTimeWarper(warper);
	}

	@Override
	public String getName() {
		return "XCat Coronary";
	}

	@Override
	public String getBibtexCitation() {
		return "@article{Segars08-RCS,\n" +
				"  author={Segars WP, Mahesh M, Beck TJ, Frey EC, Tsui BMW},\n" +
				"  title={Realistic CT simulation using the 4D XCAT phantom},\n" +
				"  journal={Med. Phys.},\n" +
				"  volume={35},\n" +
				"  pages={3800-3808},\n" +
				"  year={2008}\n" +
				"}";
	}

	@Override
	public String getMedlineCitation() {
		return "Segars WP, Mahesh M, Beck TJ, Frey EC, Tsui BMW. Realistic CT simulation using the 4D XCAT phantom. Med. Phys. 35:3800-3808 (2008);";
	}



}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/