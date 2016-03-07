/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.trajectories;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.Projection.CameraAxisDirection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.io.SafeSerializable;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.jpop.FunctionOptimizer;
import edu.stanford.rsl.jpop.FunctionOptimizer.OptimizationMode;
import edu.stanford.rsl.jpop.GradientOptimizableFunction;

public class Trajectory implements SafeSerializable{

	private static final long serialVersionUID = 8005225046964549262L;

	protected int numProjectionMatrices;
	protected int detectorWidth;
	protected int detectorHeight;
	protected double pixelDimensionX;
	protected double pixelDimensionY;
	protected double sourceToDetectorDistance;
	protected double sourceToAxisDistance;
	protected double averageAngularIncrement;
	protected int projectionStackSize;
	protected double [] reconDimensions;
	protected double [] reconVoxelSizes;
	protected Projection [] projectionMatrices;
	protected double [] primaryAngles = null;
	protected double [] secondaryAngles = null;
	private String primaryAnglesString;
	private String secondaryAnglesString;
	protected double originInPixelsX;
	protected double originInPixelsY;
	protected double originInPixelsZ;
	protected CameraAxisDirection detectorUDirection;
	protected CameraAxisDirection detectorVDirection;
	protected double detectorOffsetU;
	protected double detectorOffsetV;
	protected SimpleVector rotationAxis;

	/**
	 * Copy Constructor that reads the values of another Trajectory object.
	 * @param source
	 */
	public Trajectory (Trajectory source){
		this.numProjectionMatrices = source.numProjectionMatrices;
		this.detectorWidth = source.detectorWidth;
		this.detectorHeight = source.detectorHeight;
		this.pixelDimensionX = source.pixelDimensionX;
		this.pixelDimensionY = source.pixelDimensionY;
		this.sourceToDetectorDistance = source.sourceToDetectorDistance;
		this.sourceToAxisDistance = source.sourceToAxisDistance;
		this.averageAngularIncrement = source.averageAngularIncrement;
		if (source.reconDimensions != null) {
			this.reconDimensions = Arrays.copyOf(source.reconDimensions, source.reconDimensions.length);
		} else {
			reconDimensions = new double[3];
		}
		if (source.reconVoxelSizes != null) {
			this.reconVoxelSizes = Arrays.copyOf(source.reconVoxelSizes, source.reconVoxelSizes.length);
		} else {
			reconVoxelSizes = new double[3];
		}
		if (source.projectionMatrices != null){
			this.projectionMatrices =  new Projection[source.projectionMatrices.length];
			for(int  i = 0; i < source.projectionMatrices.length; i++)
			{
				this.projectionMatrices[i] = new Projection(source.projectionMatrices[i]);
			}
		}
		if (source.primaryAngles != null){
			this.primaryAngles =  Arrays.copyOf(source.primaryAngles, source.primaryAngles.length);
		}
		
		if (source.secondaryAngles != null){
			this.secondaryAngles =  Arrays.copyOf(source.secondaryAngles, source.secondaryAngles.length);
		}
		
		this.projectionStackSize = source.projectionStackSize;
		this.originInPixelsX = source.originInPixelsX;
		this.originInPixelsY = source.originInPixelsY;
		this.originInPixelsZ = source.originInPixelsZ;
		this.detectorUDirection = source.detectorUDirection;
		this.detectorVDirection = source.detectorVDirection;
		if (source.rotationAxis != null) {
			this.rotationAxis = source.rotationAxis.clone();
		}
	}

	public Trajectory (){
		reconDimensions = new double[3];
		reconVoxelSizes = new double[3];
	}


	public Projection[] getProjectionMatrices() {
		return projectionMatrices;
	}

	public void setProjectionMatrices(Projection[] projectionMatrices) {
		this.projectionMatrices = projectionMatrices;
	}

	public int getDetectorWidth() {
		return detectorWidth;
	}

	public void setDetectorWidth(int detectorWidth) {
		this.detectorWidth = detectorWidth;
	}

	public int getDetectorHeight() {
		return detectorHeight;
	}

	public void setDetectorHeight(int detectorHeight) {
		this.detectorHeight = detectorHeight;
	}

	public double getPixelDimensionX() {
		return pixelDimensionX;
	}

	public void setPixelDimensionX(double pixelDimensionX) {
		this.pixelDimensionX = pixelDimensionX;
	}

	public double getPixelDimensionY() {
		return pixelDimensionY;
	}

	public void setPixelDimensionY(double pixelDimensionY) {
		this.pixelDimensionY = pixelDimensionY;
	}

	/**
	 * returns the source to detector distance in [mm]
	 * @return the source to detector distance
	 */
	public double getSourceToDetectorDistance() {
		return sourceToDetectorDistance;
	}

	public void setSourceToDetectorDistance(double sourceToDetectorDistance) {
		this.sourceToDetectorDistance = sourceToDetectorDistance;
	}

	public double getSourceToAxisDistance() {
		return sourceToAxisDistance;
	}

	public void setSourceToAxisDistance(
			double sourceToAxisDistance) {
		this.sourceToAxisDistance = sourceToAxisDistance;
	}

	public double getAverageAngularIncrement() {
		return averageAngularIncrement;
	}

	public void setAverageAngularIncrement(double averageAngularIncrement) {
		this.averageAngularIncrement = averageAngularIncrement;
	}

	public double[] getReconDimensions() {
		return reconDimensions;
	}

	public void setReconDimensions(double ... reconDimensions) {
		this.reconDimensions = reconDimensions;
	}

	public void setNumProjectionMatrices(int numProjectionMatrices) {
		this.numProjectionMatrices = numProjectionMatrices;
	}

	public double[] getReconVoxelSizes() {
		return reconVoxelSizes;
	}

	public void setReconVoxelSizes(double[] reconVoxelSizes) {
		this.reconVoxelSizes = reconVoxelSizes;
	}

	public Projection getProjectionMatrix(int i){
		if (i < projectionMatrices.length && i>=0) 
			return projectionMatrices[i];
		else {
			return null;
		}
	}

	public void setProjectionMatrix(int i, Projection projMat){
		if (i < projectionMatrices.length && i>=0) 
		{
			projectionMatrices[i] = projMat;
		}
	}
	
	public int getNumProjectionMatrices(){
		return numProjectionMatrices;
	}

	public double[] getPrimaryAngles() {
		return primaryAngles;
	}

	public void setPrimaryAngleArray(double[] primaryAngles) {
		this.primaryAngles = primaryAngles;
	}
	
	public double[] getSecondaryAngles(){
		return secondaryAngles;
	}
	
	public void setSecondaryAngleArray(double[] secondaryAngles) {
		this.secondaryAngles = secondaryAngles;
	}

	public int getProjectionStackSize() {
		return projectionStackSize;
	}

	public void setProjectionStackSize(int projectionStackSize) {
		this.projectionStackSize = projectionStackSize;
	}

	public int getReconDimensionX() {
		return (int)reconDimensions[0];
	}

	public int getReconDimensionY() {
		return (int)reconDimensions[1];
	}

	public int getReconDimensionZ() {
		return (int)reconDimensions[2];
	}

	public double getVoxelSpacingX() {
		return reconVoxelSizes[0];
	}

	public double getVoxelSpacingY() {
		return reconVoxelSizes[1];
	}

	public double getVoxelSpacingZ() {
		return reconVoxelSizes[2];
	}

	public void setReconDimensionX(int value) {
		reconDimensions[0] = value;
	}

	public void setReconDimensionY(int value) {
		reconDimensions[1] = value;
	}

	public void setReconDimensionZ(int value) {
		reconDimensions[2] = value;
	}

	public void setVoxelSpacingX(double value) {
		reconVoxelSizes[0] = value;
	}

	public void setVoxelSpacingY(double value) {
		reconVoxelSizes[1] = value;
	}

	public void setVoxelSpacingZ(double value) {
		reconVoxelSizes[2] = value;
	}

	@Override
	public void prepareForSerialization() {
		if (primaryAngles != null) {
			primaryAnglesString = "";
			if (secondaryAngles != null) secondaryAnglesString = "";
			for (int i=0; i< primaryAngles.length-1; i++){
				primaryAnglesString += primaryAngles[i] + " ";
				if (secondaryAngles != null)secondaryAnglesString += secondaryAngles[i] + " ";
			}
			primaryAnglesString += primaryAngles[primaryAngles.length-1];
			if (secondaryAngles != null) secondaryAnglesString += secondaryAngles[secondaryAngles.length-1];
			if (secondaryAngles != null) secondaryAngles = null;
			primaryAngles = null;
		}
		if (projectionMatrices !=null) {
			for (int i=0; i< projectionMatrices.length; i++){
				//projectionMatrices[i].prepareForSerialization();
			}
		}
	}

	/**
	 * @return the primaryAnglesString
	 */
	public String getPrimaryAnglesString() {
		return primaryAnglesString;
	}

	/**
	 * @param primaryAnglesString the primaryAnglesString to set
	 */
	public void setPrimaryAnglesString(String primaryAnglesString) {
		String [] entries = primaryAnglesString.split("\\s+");
		primaryAngles = new double [entries.length];
		for (int i = 0; i<entries.length; i++){
			primaryAngles[i] = Double.parseDouble(entries[i]);
		}
		this.primaryAnglesString = primaryAnglesString;
	}

	/**
	 * @return the secondaryAnglesString
	 */
	public String getSecondaryAnglesString() {
		return secondaryAnglesString;
	}

	/**
	 * @param secondaryAnglesString the secondaryAnglesString to set
	 */
	public void setSecondaryAnglesString(String secondaryAnglesString) {
		String [] entries = secondaryAnglesString.split("\\s+");
		secondaryAngles = new double [entries.length];
		for (int i = 0; i<entries.length; i++){
			secondaryAngles[i] = Double.parseDouble(entries[i]);
		}
		this.secondaryAnglesString = secondaryAnglesString;
	}

	/**
	 * @return the originInPixelsX
	 */
	public double getOriginInPixelsX() {
		return originInPixelsX;
	}

	// 

	/**
	 * @return the originX in world coordinates [mm]
	 */
	public double getOriginX() {
		return General.voxelToWorld(-originInPixelsX, getVoxelSpacingX(), 0);
	}

	/**
	 * @return the originY in world coordinates [mm]
	 */
	public double getOriginY() {
		return General.voxelToWorld(-originInPixelsY, getVoxelSpacingY(), 0);
	}

	/**
	 * @return the originZ in world coordinates [mm]
	 */
	public double getOriginZ() {
		return General.voxelToWorld(-originInPixelsZ, getVoxelSpacingZ(), 0);
	}
	
	/**
	 * Method to center the world origin around a phantom's min and max.
	 * @param phantom
	 */
	public void setOriginToPhantomCenter(AnalyticPhantom phantom){
		PointND min = phantom.getMin().clone();
		PointND max = phantom.getMax().clone();
		SimpleVector minVec = min.getAbstractVector();
		minVec.add(max.getAbstractVector());
		minVec.divideBy(2);
		SimpleVector maxVec = new SimpleVector(getVoxelSpacingX()*getReconDimensionX(),getVoxelSpacingY()*getReconDimensionY(), getVoxelSpacingZ()*getReconDimensionZ());
		maxVec.divideBy(2);
		minVec.subtract(maxVec);
		maxVec.multipliedBy(2);
		maxVec.add(minVec);
		setOriginInWorld(new PointND(minVec));	
	}

	public void setOriginInWorld(PointND originInWorld){
		originInPixelsX = General.worldToVoxel(-originInWorld.get(0), getVoxelSpacingX(), 0);
		originInPixelsY = General.worldToVoxel(-originInWorld.get(1), getVoxelSpacingY(), 0);
		originInPixelsZ = General.worldToVoxel(-originInWorld.get(2), getVoxelSpacingZ(), 0);
	}
	
	
	/**
	 * @param originInPixelsX the originInPixelsX to set
	 */
	public void setOriginInPixelsX(double originInPixelsX) {
		this.originInPixelsX = originInPixelsX;
	}

	/**
	 * @return the originInPixelsY
	 */
	public double getOriginInPixelsY() {
		return originInPixelsY;
	}

	/**
	 * @param originInPixelsY the originInPixelsY to set
	 */
	public void setOriginInPixelsY(double originInPixelsY) {
		this.originInPixelsY = originInPixelsY;
	}

	/**
	 * @return the originInPixelsZ
	 */
	public double getOriginInPixelsZ() {
		return originInPixelsZ;
	}

	/**
	 * @param originInPixelsZ the originInPixelsZ to set
	 */
	public void setOriginInPixelsZ(double originInPixelsZ) {
		this.originInPixelsZ = originInPixelsZ;
	}

	/**
	 * @param detectorUDirection the detectorUDirection to set
	 */
	public void setDetectorUDirection(CameraAxisDirection detectorUDirection) {
		this.detectorUDirection = detectorUDirection;
	}

	/**
	 * @return the detectorUDirection
	 */
	public CameraAxisDirection getDetectorUDirection() {
		return detectorUDirection;
	}

	/**
	 * @return the detectorVDirection
	 */
	public CameraAxisDirection getDetectorVDirection() {
		return detectorVDirection;
	}

	/**
	 * @param detectorVDirection the detectorVDirection to set
	 */
	public void setDetectorVDirection(CameraAxisDirection detectorVDirection) {
		this.detectorVDirection = detectorVDirection;
	}

	/**
	 * @return the detectorOffsetU
	 */
	public double getDetectorOffsetU() {
		return detectorOffsetU;
	}

	/**
	 * @param detectorOffsetU the detectorOffsetU to set
	 */
	public void setDetectorOffsetU(double detectorOffsetU) {
		this.detectorOffsetU = detectorOffsetU;
	}

	/**
	 * @return the detectorOffsetV
	 */
	public double getDetectorOffsetV() {
		return detectorOffsetV;
	}

	/**
	 * @param detectorOffsetV the detectorOffsetV to set
	 */
	public void setDetectorOffsetV(double detectorOffsetV) {
		this.detectorOffsetV = detectorOffsetV;
	}

	/**
	 * @param rotationAxis the rotationAxis to set
	 */
	public void setRotationAxis(SimpleVector rotationAxis) {
		this.rotationAxis = rotationAxis;
	}

	/**
	 * @return the rotationAxis
	 */
	public SimpleVector getRotationAxis() {
		return rotationAxis;
	}

	/**
	 * computes the iso center of the current trajectory. Current computation is based on heuristics, but works.
	 * Should be implemented properly in the future.
	 * Testing with 100 tries gives a standard deviation of 0.15 mm per coordinate.<br>
	 * Average error is about 0.5 mm.
	 * Use computeIsoCenter() instead!
	 * 
	 * @return the iso center of the scan trajectory.
	 * @see #computeIsoCenter()
	 */
	public PointND computeIsoCenterOld(){
		Random random = new Random(System.currentTimeMillis());
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i = 0; i < 150; i++){
			int random1 = random.nextInt(getProjectionMatrices().length);
			Projection proj1 = getProjectionMatrices()[random1];
			StraightLine line1 = new StraightLine(new PointND(proj1.computeCameraCenter()),proj1.computePrincipalAxis().normalizedL2());
			int random2 = random.nextInt(getProjectionMatrices().length);
			Projection proj2 = getProjectionMatrices()[random2];
			StraightLine line2 = new StraightLine(new PointND(proj2.computeCameraCenter()),proj2.computePrincipalAxis().normalizedL2());
			try {
				PointND p = line1.intersect(line2, 2000);
				if (p!= null){
					list.add(p);
				}
			} catch(Exception e){

			}
		}
		PointND center = new PointND(0,0,0);
		for (PointND p:list){
			center.getAbstractVector().add(p.getAbstractVector());
		}
		center.getAbstractVector().divideBy(list.size());
		return center;
	}
	
	/**
	 * Call of this Method will replace the entries in the internal primaryAngle Array with values that are computed
	 * from the projection matrices.
	 * Do not call this too often, the gradient descent is quite slow.
	 */
	public void updatePrimaryAngles(){
		// Internal Coordinates we want to use for visualization: (0,0,0) to (1,1,1);
		Trajectory trajectory = this;
		PointND min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
		PointND max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
		for (Projection p: trajectory.getProjectionMatrices()){
			PointND source = new PointND(p.computeCameraCenter());
			min.updateIfLower(source);
			max.updateIfHigher(source);
		}
		PointND center = trajectory.computeIsoCenter();
		SimpleVector range = max.getAbstractVector();
		range.add(min.getAbstractVector().negated());		
		
		for (int i=0;i<3;i++){
			if (range.getElement(i) < 0.1) range.setElementValue(i, 0.1);
		}
		// direction in the first halfspace
		SimpleVector normdir = new SimpleVector(1,0,0);
		// direction for the angular computation in the other halfspace
		SimpleVector normdir2 = new SimpleVector(-1,0,0);
		// compute normalized source position.
		SimpleVector scaledCoordinate = trajectory.getProjectionMatrix(0).computeCameraCenter();
		scaledCoordinate.add(center.getAbstractVector().negated());
		scaledCoordinate.normalizeL2();
		SimpleMatrix normRot = Rotations.getRotationMatrixFromAtoB(scaledCoordinate, normdir);
		for (int i = 0; i < trajectory.getProjectionMatrices().length; i++){
			// Normalize points and move them to the center
			Projection p = trajectory.getProjectionMatrices()[i];
			scaledCoordinate = p.computeCameraCenter();
			scaledCoordinate.add(center.getAbstractVector().negated());
			scaledCoordinate.normalizeL2();
			scaledCoordinate = SimpleOperators.multiply(normRot, scaledCoordinate);
			double angle = Rotations.getRotationFromAtoB(normdir, scaledCoordinate);
			if (scaledCoordinate.getElement(1) > 0){
				// Other half space:
				angle =  Math.PI + Rotations.getRotationFromAtoB(normdir2, scaledCoordinate);
			}
			// primary Angles are stored in Degrees.
			primaryAngles[i] = General.toDegrees(angle);
		}
		// set first angle to 0;
		boolean invert = primaryAngles[0] > primaryAngles[1];
		double substract = primaryAngles[0];
		if (invert) substract = primaryAngles[primaryAngles.length-1];
		
		for (int i=primaryAngles.length-1; i >= 0;i--){
			primaryAngles[i] -= substract;
			primaryAngles[i] = primaryAngles[i] % 360;
			if (primaryAngles[i] < 0) 
				primaryAngles[i] = 360 + primaryAngles[i];
		}
		
		
	}

	/**
	 * Computes the iso center of the current trajectory.
	 * Computation is based on a gradient descent procedure:<br>
	 * The objective function is<br>
	 * f(p, r) = (\Sum (x-p)^\top(x-p)-r^2)^2<br>
	 * where<br>
	 * p is the iso center and r the source to iso center distance. x are the different source positions.
	 * The objective function is basically the sphere equation and we seek to find the closest point that fulfills the equation at 0.
	 * Hence, we compute the square and use a gradient descent-type of solver.<br><br>
	 * The derivatives are:<br>
	 * f(p, r) / dp = 2 * (\Sum (x-p)^\top(x-p)-r^2) * (-2\Sum(x-p))<BR>
	 * f(p, r) / dr = 2 * (\Sum (x-p)^\top(x-p)-r^2) * (-2\Sum(r))
	 * 
	 * We do 15 attempts just to be on the save side.
	 * Still quite brute force, but better than guessing.<br>
	 * Testing with 100 tries gives a standard deviation of 2.208233228965341E-15 per coordinate<br>
	 * Average error is about 2.4312505535109182E-14.
	 * 
	 * @return the iso center of the scan trajectory.
	 */
	public PointND computeIsoCenter(){
		// create list with source positions
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (Projection p: this.getProjectionMatrices()){
			list.add(new PointND(p.computeCameraCenter()));
		}
		IsoCenterObjectiveFunction function = new IsoCenterObjectiveFunction(list);
		FunctionOptimizer fo = new FunctionOptimizer();
		fo.setDimension(4);
		fo.setOptimizationMode(OptimizationMode.Function);
		fo.setConsoleOutput(false);
		PointND center = null;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < 15; i++) {
			PointND firstGuess = computeIsoCenterOld();
			//firstGuess = new PointND(0,0,0);
			fo.setInitialX(new double [] {firstGuess.get(0),firstGuess.get(1),firstGuess.get(2),sourceToAxisDistance});
			double [] result = fo.optimizeFunction(function);
			if (function.evaluate(result, 0) < min) {
				center = new PointND(result[0],result[1],result[2]);
				min = function.evaluate(result, 0);
			}
		}
		//System.out.println(firstGuess + " " + center);
		return center;
	}

	private class IsoCenterObjectiveFunction implements GradientOptimizableFunction {

		ArrayList<PointND> sourcePositions;

		public IsoCenterObjectiveFunction(ArrayList<PointND> list){
			sourcePositions = list;
		}

		@Override
		public void setNumberOfProcessingBlocks(int number) {
			// not parallel
		}

		@Override
		public int getNumberOfProcessingBlocks() {
			return 1;
		}

		@Override
		public double evaluate(double[] x, int block) {
			double sum = 0;
			double r2 = x[3]*x[3];
			for (PointND p: sourcePositions){
				SimpleVector diff = new SimpleVector(p.get(0)-x[0],p.get(1)-x[1],p.get(2)-x[2]);
				sum += SimpleOperators.multiplyInnerProd(diff, diff) - r2;
			}
			System.out.println("function " + sum*sum + " " + new SimpleVector(x));
			return (sum*sum);
		}

		@Override
		public double[] gradient(double[] x, int block) {
			double [] grad = new double [4];
			double sum =0;
			double r2 = x[3]*x[3];
			System.out.println("x: " + new SimpleVector(x));
			try{
				for (PointND p: sourcePositions){
					SimpleVector diff = new SimpleVector(p.get(0)-x[0],p.get(1)-x[1],p.get(2)-x[2]);
					sum += (SimpleOperators.multiplyInnerProd(diff, diff) - r2);
					grad[0] += diff.getElement(0) * -2.0;
					grad[1] += diff.getElement(1) * -2.0;
					grad[2] += diff.getElement(2) * -2.0;
					grad[3] += -2.0 * x[3];
				}
				grad[0] *= sum * 2.0;
				grad[1] *= sum * 2.0;
				grad[2] *= sum * 2.0;
				grad[3] *= sum * 2.0;
			} catch (Exception e){
				e.printStackTrace();
			}
			System.out.println("grad: " + new SimpleVector(grad));
			return grad;
		}

	}



}
