package edu.stanford.rsl.conrad.volume3d.operations;

public class InitializeHighPass extends VoxelOperation {

	private float[] fMax;
	private float[] fDelta;
	private float lpUpper;
	private float hpLower;
	private float hpUpper;

	@Override
	protected void performVoxelOperation() {
		float r_abs = 0;

		int dim_loop = 0;
		float tmp;
		float pos = -fMax[dim_loop] + (float) indexX * fDelta[dim_loop];
		r_abs += pos * pos;
		dim_loop=1;
		pos = -fMax[dim_loop] + (float) indexY * fDelta[dim_loop];
		r_abs += pos * pos;
		dim_loop=2;
		pos = -fMax[dim_loop] + (float) indexZ * fDelta[dim_loop];
		r_abs += pos * pos;

		r_abs = (float) Math.sqrt(r_abs);

		if (r_abs <= lpUpper) {

			tmp=(float) Math.cos(Math.PI*r_abs/(2.0*lpUpper));
			vol.data[indexX][indexY][indexZ] =	(float) (1.0 - tmp*tmp);

		} else if (lpUpper<r_abs && r_abs<=hpLower) {

			vol.data[indexX][indexY][indexZ] = 1;

		} else if (hpLower<r_abs && r_abs<=hpUpper) {

			tmp=(float) Math.cos(Math.PI*(r_abs-hpLower)/(2.0*(hpUpper-hpLower)));
			vol.data[indexX][indexY][indexZ] = tmp*tmp;

		} else
			vol.data[indexX][indexY][indexZ] = 0;

	}

	@Override
	public ParallelVolumeOperation clone() {
		InitializeHighPass clone = new InitializeHighPass();
		clone.fMax = fMax;
		clone.fDelta = fDelta;
		clone.lpUpper = lpUpper;
		clone.hpLower = hpLower;
		clone.hpUpper = hpUpper;
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

	public float getLpUpper() {
		return lpUpper;
	}

	public void setLpUpper(float lpUpper) {
		this.lpUpper = lpUpper;
	}

	public float getHpLower() {
		return hpLower;
	}

	public void setHpLower(float hpLower) {
		this.hpLower = hpLower;
	}

	public float getHpUpper() {
		return hpUpper;
	}

	public void setHpUpper(float hpUpper) {
		this.hpUpper = hpUpper;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/