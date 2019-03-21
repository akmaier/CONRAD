/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
/**
 * 
 * @author Tobias Miksch
 * 
 * Class that contains the results of a ray tracer call and saves them in form of a virtual ray light.
 * These can then be further processed and are used to calculate the complex indirect illumination.
 */
public class XRayVRay {

	private boolean valid;
	private PointND vrlStart;
	private PointND vrlEnd;
	private SimpleVector direction;
	private double distance;
	
	private double photoAbsorption;
	private double comptonAbsorption;
	
	private double energyEV;
	private double lightScaling;
	private int scattercount;
	
	public XRayVRay() {
		this.setValid(false);
	}


	public XRayVRay(PointND vplStart, PointND vplEnd, double energyEV, boolean valid) {
		this.valid = valid;
		if(valid) {
			this.vrlStart = vplStart;
			this.vrlEnd = vplEnd;
			this.distance = vplStart.euclideanDistance(vplEnd);
			setDirection(SimpleOperators.subtract(vplEnd.getAbstractVector(), vplStart.getAbstractVector()));
			this.energyEV = energyEV;
		}
	}
	
	public XRayVRay(PointND vplStart, PointND vplEnd, double energyEV, double photoAbsorption, double comptonAbsorption, boolean valid) {
		this.valid = valid;
		if(valid) {
			this.vrlStart = vplStart;
			this.vrlEnd = vplEnd;
			this.distance = vplStart.euclideanDistance(vplEnd);
			setDirection(SimpleOperators.subtract(vplEnd.getAbstractVector(), vplStart.getAbstractVector()));
			this.energyEV = energyEV;
			
			this.photoAbsorption = photoAbsorption;
			this.comptonAbsorption = comptonAbsorption;
		}
	}
	
	public double getTransmittance(double factor) {
		if(factor > 1.0 || factor < 0.0) {
			System.err.println("Factor of the function \"getTransmittance\" should not be > 1.0!");
		}
		return XRayTracerSampling.getTransmittanceOfMaterial(photoAbsorption, comptonAbsorption, factor * distance);
	}

	//Method to easily access different points on the VRL
	public PointND getPointByFraction(double factor) {
		if(factor > 1.0 || factor < 0.0) {
			System.err.println("Factor of the function \"getPointByFraction\" should not be > 1.0!");
		}
		StraightLine ray = new StraightLine(vrlStart, direction);
		return ray.evaluate(factor * distance);
	}

	//Method to easily access different points on the VRL
		public PointND getPointByDistance(double distanceToSample) {
			if(distanceToSample > distance) {
				System.err.println("Parameter of the function \"getPointByDistance\" should not be bigger then the lenght of the ray");
			}
			StraightLine ray = new StraightLine(vrlStart, direction);
			return ray.evaluate(distanceToSample);
		}
	
	public PointND getVrlStart() {
		return vrlStart;
	}


	public void setVrlStart(PointND vplStart) {
		this.vrlStart = vplStart;
	}


	public PointND getVrlEnd() {
		return vrlEnd;
	}


	public void setVrlEnd(PointND vplEnd) {
		this.vrlEnd = vplEnd;
	}


	public double getEnergyEV() {
		return energyEV;
	}


	public void setEnergyEV(double energyEV) {
		this.energyEV = energyEV;
	}


	public boolean isValid() {
		return valid;
	}


	public void setValid(boolean valid) {
		this.valid = valid;
	}


	public double getDistance() {
		return distance;
	}


	public void setDistance(double distance) {
		this.distance = distance;
	}


	public SimpleVector getDirection() {
		return direction;
	}


	public void setDirection(SimpleVector direction) {
		this.direction = direction.normalizedL2();
	}

	public void setDirection(PointND start, PointND end) {
		setDirection(SimpleOperators.subtract(end.getAbstractVector(), start.getAbstractVector()));
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


	public double getLightScaling() {
		return lightScaling;
	}


	public void setLightScaling(double lightScaling) {
		this.lightScaling = lightScaling;
	}


	public int getScattercount() {
		return scattercount;
	}


	public void setScattercount(int scattercount) {
		this.scattercount = scattercount;
	}
}
