package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
/**
 * 
 * @author Tobias Miksch
 * 
 * Class that contains the results of a ray tracer call and saves them.
 * These can then be further processed and are used to calculate the complex indirect illumination.
 */
public class XRayHitVertex {

	private PointND startPoint;
	private PointND endPoint;
	
	private SimpleVector rayDir;

	private double distance;
	private PhysicalObject currentLineSegment;
	
	private double photoAbsorption;
	private double comptonAbsorption;
	
	private double energyEV;
	private double totalDistanceFromStartingPoint;
	
	//Default Constructor
	public XRayHitVertex() {
		
	}
	
	public XRayHitVertex(PointND startPoint, PointND endPoint, SimpleVector rayDir, PhysicalObject currentLineSegment,
			double photoAbsorption, double comptonAbsorption, double energyEV, double totalDistanceFromStartingPoint) {
		
		this.startPoint = startPoint;
		this.endPoint = endPoint;
		this.rayDir = rayDir.normalizedL2();
		this.currentLineSegment = currentLineSegment;
		this.photoAbsorption = photoAbsorption;
		this.comptonAbsorption = comptonAbsorption;
		this.energyEV = energyEV;
		this.distance = startPoint.euclideanDistance(endPoint);
		this.setDistanceFromStartingPoint(totalDistanceFromStartingPoint);
	}
	
	// Method to easily access different points inbetween the vertex
	public PointND getPointOnStraightLine(double factor) {
		StraightLine ray = new StraightLine(startPoint, rayDir);
		return ray.evaluate(factor * distance);
	}

	//Getter and Setter Methods
	public PointND getStartPoint() {
		return startPoint;
	}

	public void setStartPoint(PointND startPoint) {
		this.startPoint = startPoint;
	}
	public PointND getEndPoint() {
		return endPoint;
	}
	public void setEndPoint(PointND endPoint) {
		this.endPoint = endPoint;
	}
	
	public SimpleVector getRayDir() {
		return rayDir;
	}

	public void setRayDir(SimpleVector rayDir) {
		this.rayDir = rayDir;
	}
	
	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public PhysicalObject getCurrentLineSegment() {
		return currentLineSegment;
	}

	public void setCurrentLineSegment(PhysicalObject currentLineSegment) {
		this.currentLineSegment = currentLineSegment;
	}

	public double getPhotoAbsorption() {
		return photoAbsorption;
	}

	public void setPhotoAbsorption(double photoAbsorption) {
		this.photoAbsorption = photoAbsorption;
	}

	public double getComptonAbsorption() {
		return comptonAbsorption;
	}

	public void setComptonAbsorption(double comptonAbsorption) {
		this.comptonAbsorption = comptonAbsorption;
	}

	public double getEnergyEV() {
		return energyEV;
	}

	public void setEnergyEV(double energyEV) {
		this.energyEV = energyEV;
	}
	
	public double getDistanceFromStartingPoint() {
		return totalDistanceFromStartingPoint;
	}

	public void setDistanceFromStartingPoint(double totalDistanceFromStartingPoint) {
		this.totalDistanceFromStartingPoint = totalDistanceFromStartingPoint;
	}
}
