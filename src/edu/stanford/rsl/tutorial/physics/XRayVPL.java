/*
 * Copyright (C) 2014 - Andreas Maier, Tobias Miksch
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.physics;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.materials.Material;

/**
 * Class that contains the results of a ray tracer call and saves them in form of a virtual point light.
 * These can then be further processed and are used to calculate the complex indirect illumination.
 *
 * @author Tobias Miksch
 * 
 */
public class XRayVPL {

	private PointND vplPos;
	private double energyEV;
	private double lightpower;
	private SimpleVector direction;
	private int scattercount;
	
	public XRayVPL() {
		
	}
	public XRayVPL(PointND vplPos, double energyInEV) {
		this.vplPos = vplPos;
		this.energyEV = energyInEV;
	}
	
	public XRayVPL(PointND vplPos, double energyEV, int scattercount, PointND vplOrigin) {
		this.vplPos = vplPos;
		this.energyEV = energyEV;
		this.scattercount = scattercount;
		setDirection(vplOrigin, vplPos);
		this.lightpower = 1.0;
	}
	
	public XRayVPL(PointND vplPos, double energyEV, int scattercount, SimpleVector direction) {
		this.vplPos = vplPos;
		this.energyEV = energyEV;
		this.scattercount = scattercount;
		setDirection(direction);
		this.lightpower = 1.0;
	}

	public void setDirection(SimpleVector direction) {
		this.direction = direction.normalizedL2();
	}
	
	public void setDirection(PointND start, PointND end) {
		setDirection(SimpleOperators.subtract(end.getAbstractVector(), start.getAbstractVector()));
	}
	
	public SimpleVector getDirection(){
		return direction;
	}
	
	public PointND getVplPos() {
		return vplPos;
	}

	public void setVplPos(PointND vplPos) {
		this.vplPos = vplPos;
	}

	public double getEnergyEV() {
		return energyEV;
	}

	public void setEnergyEV(double energyEV) {
		this.energyEV = energyEV;
	}

	public int getScattercount() {
		return scattercount;
	}

	public void setScattercount(int scattercount) {
		this.scattercount = scattercount;
	}
	public double getLightpower() {
		return lightpower;
	}
	public void setLightpower(double lightpower) {
		this.lightpower = lightpower;
	}

}
