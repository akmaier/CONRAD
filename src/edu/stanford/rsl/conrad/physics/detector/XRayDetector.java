/*
 * Copyright (C) 2014 Andreas Maier, Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.physics.detector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Edge;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.absorption.AbsorptionModel;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class XRayDetector extends PhysicalObject implements GUIConfigurable, Serializable{

	boolean configured = false;

	/**
	 * 
	 */
	private static final long serialVersionUID = -4281231525147924304L;
	protected AbsorptionModel model;
	protected Projection projection;
	
	/**
	 * Method to initialize or to reset parameters of a detector model. This method is called by the rendering classes before rendering starts.
	 * By default nothing is done here, but the method can be overridden in the corresponding child classes.
	 */
	public void init(){
		
	}
	

	/**
	 * Method that is notified once the rendering process using this AbsorptionModel is finished.
	 * This method is intended to be used with AbsorptionModels that collect information about the
	 * rendering process. For example a histogram of path lengths could be stored in an AbsortionModel
	 * and in this call the result could be stored in the registry or a certain file.
	 * <br>
	 * The default implementation just does nothing. 
	 */
	public void notifyEndOfRendering(){	
		model.notifyEndOfRendering();
	}

	/**
	 * This method creates the data container for the detector. By default a Grid2D is generated.
	 * More sophisticated absorption models can also use different grid containers, i.e., if we want
	 * to generate a material image for each material.
	 * @param width
	 * @param height
	 * @param the projection for the detector.
	 * @return the grid to store the values in.
	 */
	public Grid2D createDetectorGrid(int width, int height, Projection proj){
		return createDetectorGrid(width, height);
	}
	
	/**
	 * Generates and transforms the box shape of the detector
	 * @param proj The projection for the detector-source setup
	 * @param thickness	The thickness of the detector box
	 */
	public void generateDetectorShape(Projection proj, double thickness){
		projection = proj;
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		// Create Box in Detector Size
		Box detectorShape = new Box(traj.getDetectorWidth()*traj.getPixelDimensionX(), traj.getDetectorHeight()* traj.getPixelDimensionY(), thickness);
		// shift to origin.
		// and include correct detector offset.
		SimpleVector principalPoint = proj.computeOffset(new SimpleVector(traj.getDetectorWidth(), traj.getDetectorHeight()));
		detectorShape.applyTransform(new Translation((-traj.getDetectorWidth()*traj.getPixelDimensionX()/2)-(principalPoint.getElement(0)*traj.getPixelDimensionX()), (-traj.getDetectorHeight()* traj.getPixelDimensionY()/2)-(principalPoint.getElement(1)*traj.getPixelDimensionY()), -thickness/2));
		// default detector orientation (i.e. Box orientation)

		// principal ray direction to which we want to align the detector.
		SimpleVector targetOrientation = proj.computePrincipalAxis();
		targetOrientation.normalizeL2();
		// compute rotation matrix
		//SimpleMatrix rot = Rotations.getRotationMatrixFromAtoB(currentOrientation, targetOrientation);
		SimpleVector principalPointPixels = proj.getPrincipalPoint();
		SimpleVector principalPointWorld = proj.computeRayDirection(principalPointPixels);

		//determine x axis of the detector in world coordinates
		SimpleVector right = principalPointPixels.clone();
		right.add(new SimpleVector(1,0));
		SimpleVector axisRight = proj.computeRayDirection(right);
		axisRight.subtract(principalPointWorld);
		axisRight.normalizeL2();
		
		//determine y axis of the detector in world coordinates
		SimpleVector up = principalPointPixels.clone();
		up.add(new SimpleVector(0,1));
		SimpleVector axisUp = proj.computeRayDirection(up);
		axisUp.subtract(principalPointWorld);
		axisUp.normalizeL2();
		
		//build rotations matrix
		SimpleMatrix rot = new SimpleMatrix(3,3);
		rot.setColValue(0, axisRight);
		rot.setColValue(1, axisUp);
		rot.setColValue(2, targetOrientation);
		
				
		detectorShape.applyTransform(new ScaleRotate(rot));
		
		//move detector so its front plane has the correct distance to the source
		detectorShape.applyTransform(new Translation(targetOrientation.multipliedBy(thickness/2)));
		
		//move detector along the target direction to the correct distance
		SimpleVector source = proj.computeCameraCenter();
		source.add(targetOrientation.multipliedBy(traj.getSourceToDetectorDistance()));
		detectorShape.applyTransform(new Translation(source));
		this.setShape(detectorShape);
}
	
	/**
	 * Absorb a single photon.
	 * @param grid	The resulting image of the detector front plane.
	 * @param point	The points of absorbtion in world coordinates.
	 * @param energy	The energy of the photon
	 */
	public void absorbPhoton(Grid2D grid, PointND point, double energy){
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();

		SimpleVector pixel = new SimpleVector(0,0);
		projection.project(point.getAbstractVector(), pixel);

		if (pixel.getElement(0) > traj.getDetectorWidth()|| pixel.getElement(1) > traj.getDetectorHeight() || pixel.getElement(0) < 0 || pixel.getElement(1) < 0){
			//pixel outside of the grid
			//System.err.println("XRayDetector: pixel outside: " + pixel.getElement(0) + ", " +  pixel.getElement(1));
		} else {
			InterpolationOperators.addInterpolateLinear(grid, (int)pixel.getElement(0), (int)pixel.getElement(1), (float)energy);
		}
			
	}

	public Grid2D createDetectorGrid(int width, int height){
		Grid2D grid = new Grid2D(width, height);
		return grid;
	}
	
	/**
	 * Method to put the absorption information into the detector. This method will be called by the ray tracer
	 * to store the right information in the right detector pixel. By default, this method stores the
	 * line integral computed by evaluateLineIntegral(segments) in each detector pixel.
	 * 
	 * @param grid the image grid created by createDetectorGrid
	 * @param x the pixel x coordinate
	 * @param y the pixel y coordinate
	 * @param segments the segments that were passed by the ray.
	 */
	public void writeToDetector(Grid2D grid, int x, int y, ArrayList<PhysicalObject> segments){
		grid.putPixelValue(x, y, model.evaluateLineIntegral(segments));
	}

	@Override
	public void configure() throws Exception {
		model = (AbsorptionModel) UserUtil.queryObject("Select absorption model:", "Absorption Model Selection", AbsorptionModel.class);
		model.configure();
		configured = true;
	}

	
	public void configure(AbsorptionModel absorpMod) throws Exception {
		model = absorpMod;
		if (!model.isConfigured())
			model.configure();
		configured = true;
	}

	@Override
	public boolean isConfigured() {
		return configured;
	}

	/**
	 * Method that parses the segments and accumulates the path length for each material.
	 * @param segments the ray tracing segments
	 * @return a HashMap that can be queried for materials and returns the respective path length.
	 */
	public static HashMap<Material, Double> accumulatePathLenghtForEachMaterial(ArrayList<PhysicalObject> segments){
		HashMap<Material, Double> localMap = new HashMap<Material, Double>();
		for (PhysicalObject segment: segments){
			Material mat = segment.getMaterial();
			Double value = localMap.get(mat);
			if (value == null){
				value = new Double(0);
			}
			Edge edge = (Edge) segment.getShape();
			value += edge.getLength();
			localMap.put(mat, value);
		}
		return localMap;
	}

	public String toString(){
		String name = "Generic X-Ray Detector";
		if (model!= null) name+=" " + model.toString();
		return name;
	}

	/**
	 * @return the model
	 */
	public AbsorptionModel getModel() {
		return model;
	}

	/**
	 * @param model the model to set
	 */
	public void setModel(AbsorptionModel model) {
		this.model = model;
	}

	/**
	 * @param configured the configured to set
	 */
	public void setConfigured(boolean configured) {
		this.configured = configured;
	}

}
