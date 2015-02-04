package edu.stanford.rsl.conrad.volume3d.operations;

import edu.stanford.rsl.conrad.volume3d.VolumeOperator.FILTER_TYPE;

public class InitializeSquaredCosineR extends VoxelOperation {

	private float[] fMax;
	private float[] fDelta;
	private float[] direction;
	private FILTER_TYPE filterType;
	private int exponent;
	private float B;
	private float ri;
	
	public float getB() {
		return B;
	}

	public void setB(float b) {
		B = b;
	}

	public float getRi() {
		return ri;
	}

	public void setRi(float ri) {
		this.ri = ri;
	}

	@Override
	protected void performVoxelOperation() {
		float r_dot = 0.0f;
		float r_abs = 0.0f;

		float pos = -fMax[0] + indexX * fDelta[0];
		r_dot += direction[0] * pos;
		r_abs += pos * pos;
		pos = -fMax[1] + indexY * fDelta[1];
		r_dot += direction[1] * pos;
		r_abs += pos * pos;
		pos = -fMax[2] + indexZ * fDelta[2];
		r_dot += direction[2] * pos;
		r_abs += pos * pos;

		r_abs = (float) Math.sqrt(r_abs);
		
		if (r_abs != 0 && !(filterType==FILTER_TYPE.QUADRATIC && r_dot < 0) ) {

			float tmp = r_dot/r_abs;
			vol.data[indexX][indexY][indexZ] = 1;
			for (int exp_loop=0; exp_loop<exponent; exp_loop++) 
				vol.data[indexX][indexY][indexZ] *= tmp*tmp;

			tmp = (float) Math.log(r_abs/ri);
			vol.data[indexX][indexY][indexZ] *= Math.exp( -4.0 / Math.log( (float) 2 ) /  (B*B)*tmp*tmp);
			//System.out.println("1: " + r_abs + " " + filterType + " " + r_dot + " " + vol.data[indexX][indexY][indexZ]);
		} else {
			vol.data[indexX][indexY][indexZ] = 0;
			//System.out.println("2: " + r_abs + " " + filterType + " " + r_dot + " " + vol.data[indexX][indexY][indexZ]);
		}

	}

	@Override
	public ParallelVolumeOperation clone() {
		InitializeSquaredCosineR clone = new InitializeSquaredCosineR();
		clone.fMax = fMax;
		clone.fDelta = fDelta;
		clone.direction = direction;
		clone.filterType = filterType;
		clone.exponent = exponent;
		clone.B = B;
		clone.ri = ri;
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