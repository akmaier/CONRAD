package edu.stanford.rsl.conrad.phantom.xcat;

import java.io.IOException;
import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.motion.ConstantMotionField;
import edu.stanford.rsl.conrad.geometry.motion.DualMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MixedSurfaceBSplineMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.MovingCenterRotationMotionField;
import edu.stanford.rsl.conrad.geometry.motion.OpenCLParzenWindowMotionField;
import edu.stanford.rsl.conrad.geometry.motion.PointBasedMotionField;
import edu.stanford.rsl.conrad.geometry.motion.RotationMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Plane3D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.splines.TimeVariantSurfaceBSpline;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Class to simluate very simple knee joint motion in XCat.
 * Scene contains only the lower body as only the legs and hip are of interest for a squat.
 * 
 * @author akmaier
 *
 */
public class SquatScene extends WholeBodyScene {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5833963135277640258L;

	MotionField lowerBodyMotionField = null;
	boolean rotationOnly = false;
	double sigmaParzen=10;
	int numberOfNeighborsLinInterp = 5;
	int numberOfBSplineTimePoints = 10;
	double angleInDegrees = 45;
	
	public SquatScene (){
	}
	
	public SquatScene (int numberOfNeighborsLinInterp, double sigmaParzen, int nrOfTimePoints, double maxAngle){
		this.sigmaParzen = sigmaParzen;
		this.numberOfNeighborsLinInterp = numberOfNeighborsLinInterp;
		numberOfBSplineTimePoints = nrOfTimePoints;
		angleInDegrees = maxAngle;
	}

	@Override
	public void configure(){

		String [] constants = {
				"finger",
				"pelvis"
		};

		String [] splinesToKeep = {
				"body"
		};


		TimeWarper warper = new IdentityTimeWarper();

		setName("Squat Scene");		
		//init();

		/*
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		PointND currentOrigin = new PointND(269.3, 287.2, -638.3);
		for (int i = 0; i < geom.getReconDimensions().length; i++) {
			double size = geom.getReconDimensions()[i]*geom.getReconVoxelSizes()[i];
			currentOrigin.set(i, currentOrigin.get(i)-(size/2.0-0.5));
		}
		setMin(currentOrigin);
		SimpleVector volumeSize = new SimpleVector(
				geom.getVoxelSpacingX()*geom.getReconDimensionX(),
				geom.getVoxelSpacingY()*geom.getReconDimensionY(), 
				geom.getVoxelSpacingZ()*geom.getReconDimensionZ()
				);
		setMax(new PointND(SimpleOperators.add(currentOrigin.getAbstractVector(),volumeSize)));
		*/
		splines = readSplines();
		// remove splines outside the volume of interest using z axis
		double maxZ = 0;
		double minZ = 0;
		for (int i=splines.size()-1; i >= 0; i--){
			if (splines.get(i).getTitle().toLowerCase().contains("rightfemur")){
				maxZ = splines.get(i).getMax().get(2)-100;
				break;
			}
		}
		for (int i=splines.size()-1; i >= 0; i--){
			if (splines.get(i).getTitle().toLowerCase().contains("rightfibula")){
				minZ = splines.get(i).getMin().get(2)+100;
				break;
			}
		}
		
		for (int i=splines.size()-1; i >= 0; i--){
			boolean keepSpline = false;
			for (String s : splinesToKeep) {
				if (splines.get(i).getTitle().toLowerCase().contains(s))
					keepSpline = true;
			}

			if (!keepSpline){
				if (splines.get(i).getMax().get(2) < minZ){// below VOI
					splines.remove(i);
				} else {
					if (splines.get(i).getMin().get(2) > maxZ) { // above VOI
						splines.remove(i);
					}
				}
			}
		}
		
		updateSceneLimits();


		// define relevant points in scene

		PointND leftFemurTop = new PointND(16.6, 273.6,-207.42);
		//PointND leftFemurBottom = new PointND(21.1, 269.1,-631.92);
		PointND leftPatellaRotCenter = new PointND(21.1, 269.1,-590.92);

		//ScaleRotate femurTransform = new ScaleRotate(Rotations.createRotationMatrixAboutAxis(new SimpleVector(1,0,0), General.toRadians(-angleInDegrees)));

		//Edge leftFemur = new Edge(leftFemurTop, leftFemurBottom);
		//rotateAround(leftFemur, femurTransform, leftFemurTop);

		//SimpleVector leftLowerLegShift = lib.subtract(leftFemur.getEnd().getAbstractVector(), leftFemurBottom.getAbstractVector());

		PointND rightFemurTop = new PointND(269.3, 273.6,-207.42);
		PointND rightFemurBottom = new PointND(264.8, 264.6,-631.92);
		PointND rightPatellaRotCenter = new PointND(264.8, 264.6,-590.92);

		//Edge rightFemur = new Edge(rightFemurTop, rightFemurBottom);
		//rotateAround(rightFemur, femurTransform, rightFemurTop);

		//SimpleVector rightLowerLegShift = lib.subtract(rightFemur.getEnd().getAbstractVector(), rightFemurBottom.getAbstractVector());

		//AffineTransform leftTibiaFibulaTransform = new AffineTransform();
		//leftTibiaFibulaTransform.setTransformer(Rotations.createRotationMatrixAboutAxis(new SimpleVector(1,0,0), General.toRadians(angleInDegrees)));
		//leftTibiaFibulaTransform.setTranslator(leftLowerLegShift);

		//PointND leftTibiaTop = new PointND(7.6, 269.1,-654.1+lowerZoffset);
		PointND leftTibiaTop = new PointND(7.6, 287.2,-638.3);
		//PointND leftTibiaBottom = new PointND(-33.0, 291.7,-1001.5);

		PointND leftFibulaTop = new PointND(-24.0, 300.7,-667.6);
		//PointND leftFibulaBottom = new PointND(-51.1, 318.8,-1015.0);

		//Edge leftTibia = new Edge(leftTibiaTop, leftTibiaBottom);
		//Edge leftFibula = new Edge(leftFibulaTop, leftFibulaBottom);

		//rotateAround(leftFibula, leftTibiaFibulaTransform, leftFibulaTop);
		//rotateAround(leftTibia, leftTibiaFibulaTransform, leftTibiaTop);

		//AffineTransform rightTibiaFibulaTransform = new AffineTransform();
		//rightTibiaFibulaTransform.setTransformer(Rotations.createRotationMatrixAboutAxis(new SimpleVector(1,0,0), General.toRadians(angleInDegrees)));
		//rightTibiaFibulaTransform.setTranslator(rightLowerLegShift);

		//PointND rightTibiaTop = new PointND(273.8, 269.1,-654.1+lowerZoffset);
		PointND rightTibiaTop = new PointND(269.3, 287.2, -638.3);
		//PointND rightTibiaBottom = new PointND(314.4, 291.7,-1001.5);

		PointND rightFibulaTop = new PointND(309.9, 300.7,-667.6);
		//PointND rightFibulaBottom = new PointND(337.0, 318.8,-1015.0);

		//Edge rightTibia = new Edge(rightTibiaTop, rightTibiaBottom);
		//Edge rightFibula = new Edge(rightFibulaTop, rightFibulaBottom);

		//rotateAround(rightFibula, rightTibiaFibulaTransform, rightFibulaTop);
		//rotateAround(rightTibia, rightTibiaFibulaTransform, rightTibiaTop);

		// define cascading motion fields.

		// define motion around femur bones.

		RotationMotionField leftFemurMotion = new RotationMotionField(leftFemurTop, new SimpleVector(1,0,0), General.toRadians(-angleInDegrees));
		leftFemurMotion.setTimeWarper(warper);

		RotationMotionField rightFemurMotion = new RotationMotionField(rightFemurTop, new SimpleVector(1,0,0), General.toRadians(-angleInDegrees));
		rightFemurMotion.setTimeWarper(warper);

		// create lists for the complete motion field.

		ArrayList<ArrayList<PointND>> lowerBodyMotion = new ArrayList<ArrayList<PointND>>();
		for (int t=0; t<numberOfBSplineTimePoints; t++){
			lowerBodyMotion.add(new ArrayList<PointND>());
		}


		// for each spline in the body
		for (int i = splines.size()-1; i >= 0; i--){
			// test whether the spline is constant.
			boolean constantTest = false;
			for (String s: constants){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					constantTest = true;
				}
			}
			// create a time-constant spline for constants
			if (constantTest){
				System.out.println("Creating time-constant spline for " + splines.get(i).getTitle());
				TimeVariantSurfaceBSpline tSpline = new TimeVariantSurfaceBSpline(splines.get(i), new ConstantMotionField(), numberOfBSplineTimePoints, true);
				for (int t=0; t<numberOfBSplineTimePoints; t++){
					lowerBodyMotion.get(t).addAll(tSpline.getControlPoints(t));
				}
			}
			// rotate right femur bone with the right femur motion field
			if (splines.get(i).getTitle().toLowerCase().contains("rightfemur")){
				System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFemurMotion, numberOfBSplineTimePoints, true));
				splines.remove(i);	
			}
			// rotate right femur bone marrow with the right femur motion field.
			if (splines.get(i).getTitle().toLowerCase().contains("r_femur")){
				System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFemurMotion, numberOfBSplineTimePoints, true));
				splines.remove(i);
			}
			// rotate left femur bone with the left femur motion field.
			if (splines.get(i).getTitle().toLowerCase().contains("leftfemur")){
				System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFemurMotion, numberOfBSplineTimePoints, true));
				splines.remove(i);
			}
			// rotate left femur bone marrow with the left femur motion field.
			if (splines.get(i).getTitle().toLowerCase().contains("l_femur")){
				System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFemurMotion, numberOfBSplineTimePoints, true));
				splines.remove(i);
			}
		}

		// define motion field for lower body 1. iteration
		// femurs move. constants are constant.
		if (!rotationOnly){
			lowerBodyMotionField = getSceneMotion(numberOfBSplineTimePoints, lowerBodyMotion, numberOfNeighborsLinInterp);
			lowerBodyMotionField.setTimeWarper(warper);


			// define cascading motion field
			// i.e. Motion fields that have a moving center of rotation

			// for tibiae

			MovingCenterRotationMotionField leftTibiaMotion = new MovingCenterRotationMotionField(leftTibiaTop, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees));
			leftTibiaMotion.setTimeWarper(warper);
			MovingCenterRotationMotionField rightTibiaMotion = new MovingCenterRotationMotionField(rightTibiaTop, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees));
			rightTibiaMotion.setTimeWarper(warper);

			// for fibulae

			MovingCenterRotationMotionField leftFibulaMotion = new MovingCenterRotationMotionField(leftFibulaTop, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees));
			leftFibulaMotion.setTimeWarper(warper);
			MovingCenterRotationMotionField rightFibulaMotion = new MovingCenterRotationMotionField(rightFibulaTop, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees));
			rightFibulaMotion.setTimeWarper(warper);

			// for patellae

			MovingCenterRotationMotionField leftPatellaMotion = new MovingCenterRotationMotionField(leftPatellaRotCenter, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees/2));
			leftPatellaMotion.setTimeWarper(warper);
			MovingCenterRotationMotionField rightPatellaMotion = new MovingCenterRotationMotionField(rightPatellaRotCenter, lowerBodyMotionField, new SimpleVector(1,0,0), General.toRadians(angleInDegrees/2));
			rightPatellaMotion.setTimeWarper(warper);

			SimpleMatrix FemurRotationCenterShifts_30 = new SimpleMatrix(6,3);
			System.out.println("Left Tibia Center Translation: " + lowerBodyMotionField.getPosition(leftTibiaTop, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(0, lowerBodyMotionField.getPosition(leftTibiaTop, 0, 0.55).getAbstractVector());
			System.out.println("Left Fibula Center Translation: " + lowerBodyMotionField.getPosition(leftFibulaTop, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(1, lowerBodyMotionField.getPosition(leftFibulaTop, 0, 0.55).getAbstractVector());
			System.out.println("Left Patella Center Translation: " + lowerBodyMotionField.getPosition(leftPatellaRotCenter, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(2, lowerBodyMotionField.getPosition(leftPatellaRotCenter, 0, 0.55).getAbstractVector());
			System.out.println("Right Tibia Center Translation: " + lowerBodyMotionField.getPosition(rightTibiaTop, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(3, lowerBodyMotionField.getPosition(rightTibiaTop, 0, 0.55).getAbstractVector());
			System.out.println("Right Fibula Center Translation: " + lowerBodyMotionField.getPosition(rightFibulaTop, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(4, lowerBodyMotionField.getPosition(rightFibulaTop, 0, 0.55).getAbstractVector());
			System.out.println("Right Patella Center Translation: " + lowerBodyMotionField.getPosition(rightPatellaRotCenter, 0, 0.55));
			FemurRotationCenterShifts_30.setRowValue(5, lowerBodyMotionField.getPosition(rightPatellaRotCenter, 0, 0.55).getAbstractVector());
			
			SimpleMatrix FemurRotationCenterShifts_60 = new SimpleMatrix(6,3);
			System.out.println("Left Tibia Center Translation: " + lowerBodyMotionField.getPosition(leftTibiaTop, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(0, lowerBodyMotionField.getPosition(leftTibiaTop, 0, 1).getAbstractVector());
			System.out.println("Left Fibula Center Translation: " + lowerBodyMotionField.getPosition(leftFibulaTop, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(1, lowerBodyMotionField.getPosition(leftFibulaTop, 0, 1).getAbstractVector());
			System.out.println("Left Patella Center Translation: " + lowerBodyMotionField.getPosition(leftPatellaRotCenter, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(2, lowerBodyMotionField.getPosition(leftPatellaRotCenter, 0, 1).getAbstractVector());
			System.out.println("Right Tibia Center Translation: " + lowerBodyMotionField.getPosition(rightTibiaTop, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(3, lowerBodyMotionField.getPosition(rightTibiaTop, 0, 1).getAbstractVector());
			System.out.println("Right Fibula Center Translation: " + lowerBodyMotionField.getPosition(rightFibulaTop, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(4, lowerBodyMotionField.getPosition(rightFibulaTop, 0, 1).getAbstractVector());
			System.out.println("Right Patella Center Translation: " + lowerBodyMotionField.getPosition(rightPatellaRotCenter, 0, 1));
			FemurRotationCenterShifts_60.setRowValue(5, lowerBodyMotionField.getPosition(rightPatellaRotCenter, 0, 1).getAbstractVector());

			
			System.out.println(FemurRotationCenterShifts_30);
			System.out.println(FemurRotationCenterShifts_60);
			
			
			// apply motion fields to splines.
			for (int i = splines.size()-1; i >= 0; i--){
				boolean constantTest = false;
				for (String s: constants){
					if (splines.get(i).getTitle().toLowerCase().contains(s)){
						constantTest = true;
					}
				}
				// constants remain constant.
				if (constantTest){
					System.out.println("Creating time-constant spline for " + splines.get(i).getTitle());
					TimeVariantSurfaceBSpline tSpline = new TimeVariantSurfaceBSpline(splines.get(i), new ConstantMotionField(), numberOfBSplineTimePoints, true);
					for (int t=0; t<numberOfBSplineTimePoints; t++){
						lowerBodyMotion.get(t).addAll(tSpline.getControlPoints(t));
					}
					// apply respective motion to the respective bones.
				} else if (splines.get(i).getTitle().toLowerCase().contains("lefttibia") || splines.get(i).getTitle().toLowerCase().contains("l_tibia")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftTibiaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("righttibia") || splines.get(i).getTitle().toLowerCase().contains("r_tibia")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightTibiaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("leftfibula") || splines.get(i).getTitle().toLowerCase().contains("l_fibula")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFibulaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("rightfibula") || splines.get(i).getTitle().toLowerCase().contains("r_fibula")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFibulaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("leftpatella") || splines.get(i).getTitle().toLowerCase().contains("l_patella")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftPatellaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("rightpatella") || splines.get(i).getTitle().toLowerCase().contains("r_patella")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightPatellaMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				}
			}

			lowerBodyMotionField = new DualMotionField(new HalfSpaceBoundingCondition(new Plane3D(rightFemurBottom, new SimpleVector(0,0,1))), lowerBodyMotionField, leftTibiaMotion);
			lowerBodyMotionField.setTimeWarper(warper);

		}
		else{
			for (int i = splines.size()-1; i >= 0; i--){
				if (splines.get(i).getTitle().toLowerCase().contains("lefttibia") || splines.get(i).getTitle().toLowerCase().contains("l_tibia")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("righttibia") || splines.get(i).getTitle().toLowerCase().contains("r_tibia")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("leftfibula") || splines.get(i).getTitle().toLowerCase().contains("l_fibula")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("rightfibula") || splines.get(i).getTitle().toLowerCase().contains("r_fibula")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("leftpatella") || splines.get(i).getTitle().toLowerCase().contains("l_patella")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), leftFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				} else if (splines.get(i).getTitle().toLowerCase().contains("rightpatella") || splines.get(i).getTitle().toLowerCase().contains("r_patella")){
					System.out.println("Creating affine-time-variant spline for " + splines.get(i).getTitle());
					variants.add(new TimeVariantSurfaceBSpline(splines.get(i), rightFemurMotion, numberOfBSplineTimePoints, true));
					splines.remove(i);
				}
			}

			lowerBodyMotionField = getSceneMotion(numberOfBSplineTimePoints, lowerBodyMotion, (int)Math.round(numberOfNeighborsLinInterp));//new PointBasedMotionField(getVariants(), (int)Math.round(sigma));
			//new MixedSurfaceBSplineMotionField(getVariants(), getSplines(), sigma);
			lowerBodyMotionField.setTimeWarper(warper);

			//lowerBodyMotionField = new MixedSurfaceBSplineMotionField(getVariants(), getSplines(), 10);//getSceneMotion(numberOfBSplineTimePoints, lowerBodyMotion, 5);
		}

		// 2nd estimate of the motion field to the lower body containing
		// motion of femurs, tibiae, patellae, fibulae, and time constants. 
		// use new motion field to interpolate the the remaining splines.
		for (int i = splines.size()-1; i >= 0; i--){
			boolean constantTest = false;
			for (String s: constants){
				if (splines.get(i).getTitle().toLowerCase().contains(s)){
					constantTest = true;
				}
			}
			if (!constantTest){
				System.out.println("Creating time-variant spline for " + splines.get(i).getTitle());
				variants.add(new TimeVariantSurfaceBSpline(splines.get(i), lowerBodyMotionField, numberOfBSplineTimePoints, true));
				splines.remove(i);
			}
		}
		//getSceneMotion(numberOfBSplineTimePoints, lowerBodyMotion, 5);
		/*
		lowerBodyMotionField = new ConstantShiftMotionField(lowerBodyMotionField, rightTibiaTop);
		lowerBodyMotionField.setTimeWarper(warper);

		for (int i = copyOfSplines.size()-1; i >= 0; i--){
			System.out.println("Creating time-variant spline for " + copyOfSplines.get(i).getTitle());
			variants.add(new TimeVariantSurfaceBSpline(copyOfSplines.get(i), lowerBodyMotionField, numberOfBSplineTimePoints, true));
			copyOfSplines.remove(i);
		}
		 */	
		createPhysicalObjects();
		lowerBodyMotionField = new MixedSurfaceBSplineMotionField(getVariants(), getSplines(), sigmaParzen);
		lowerBodyMotionField.setTimeWarper(warper);
	}

	@Override
	public PointND getPosition(PointND initialPosition, double initialTime, double time) {
		return lowerBodyMotionField.getPosition(initialPosition, initialTime, time);
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		return lowerBodyMotionField.getPositions(initialPosition, initialTime, times);
	}

	@Override
	public void setTimeWarper(TimeWarper warper){
		super.setTimeWarper(warper);
		lowerBodyMotionField.setTimeWarper(warper);
	}

	@Override
	public MotionField getMotionField() {
		return lowerBodyMotionField;
	};

	@Override
	public String getName() {
		return "XCat Squat";
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/