package edu.stanford.rsl.conrad.volume3d.operations;

import edu.stanford.rsl.conrad.volume3d.VolumeOperator.FILTER_TYPE;

public class InitializeSquaredCosine extends VoxelOperation {

	private float[] fMax;
	private float[] fDelta;
	private float[] direction;
	private FILTER_TYPE filterType;
	private int exponent;

	
	@Override
	protected void performVoxelOperation() {
		float r_dot = 0.0f;
		float r_abs = 0.0f;


		float pos = -fMax[0] + (float) indexX * fDelta[0];
		r_dot += direction[0] * pos;
		r_abs += pos * pos;
		pos = -fMax[1] + (float) indexY * fDelta[1];
		r_dot += direction[1] * pos;
		r_abs += pos * pos;
		pos = -fMax[2] + (float) indexZ * fDelta[2];
		r_dot += direction[2] * pos;
		r_abs += pos * pos;


		r_abs = (float) Math.sqrt(r_abs);   /* LW 990320 */

		
		
		if (r_abs != 0 && !(filterType==FILTER_TYPE.QUADRATIC && r_dot < 0) ) {
			
			float tmp = r_dot/r_abs;
			vol.data[indexX][indexY][indexZ] = 1;
			for (int exp_loop=0; exp_loop<exponent; exp_loop++) 
				vol.data[indexX][indexY][indexZ] *= tmp*tmp;

		} else
			vol.data[indexX][indexY][indexZ] = 0;

	}

	@Override
	public ParallelVolumeOperation clone() {
		InitializeSquaredCosine clone = new InitializeSquaredCosine();
		clone.fMax = fMax;
		clone.fDelta = fDelta;
		clone.direction = direction;
		clone.filterType = filterType;
		clone.exponent = exponent;
		return clone;
	}

	public float[] getfMax() {
		return fMax;
	}

	public void setfMax(float[] fMax) {
		this.fMax = fMax;
	}

	public float[] getfDelta() {
		return fDelta;
	}

	public void setfDelta(float[] fDelta) {
		this.fDelta = fDelta;
	}

	public float[] getDirection() {
		return direction;
	}

	public void setDirection(float[] direction) {
		this.direction = direction;
	}

	public FILTER_TYPE getFilterType() {
		return filterType;
	}

	public void setFilterType(FILTER_TYPE filterType) {
		this.filterType = filterType;
	}

	public int getExponent() {
		return exponent;
	}

	public void setExponent(int exponent) {
		this.exponent = exponent;
	}

	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/