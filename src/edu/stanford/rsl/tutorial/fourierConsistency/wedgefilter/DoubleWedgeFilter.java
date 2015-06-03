/*
 * Copyright (C) 2014 Marcel Pohlmann
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.fourierConsistency.wedgefilter;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;

public abstract class DoubleWedgeFilter extends Grid2D{
	protected double rp;				// Maximum radius
	
	public DoubleWedgeFilter(int[] size, double[] spacing, double[] origin, double rp) {
		super(new float[size[1]*size[0]], size[0], size[1]);
		
		this.setSpacing(spacing);
		this.setOrigin(origin);
		this.rp = rp;
	}
	
	public void setParameterRp(double rp) {
		this.rp = rp;
		
		this.update();
	}
	
	public double getParameterRp() {
		return this.rp;
	}
	
	protected abstract void update();
	
}