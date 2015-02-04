package edu.stanford.rsl.conrad.volume3d.operations;

public class InitializeGaussian extends VoxelOperation {

	private float[] fMax;
	private float[] fDelta;
	private float sigma;


	@Override
	protected void performVoxelOperation() {
		float r_abs_sq = 0.0f;

		double pos = - (double) fMax[0] + (double) indexX * (double) fDelta[0];
		r_abs_sq += pos * pos;
		pos = - (double) fMax[1] + (double) indexY * (double) fDelta[1];
		r_abs_sq += pos * pos;
		pos = - (double) fMax[2] + (double) indexZ * (double) fDelta[2];
		r_abs_sq += pos * pos;

		vol.data[indexX][indexY][indexZ] = (float) Math.exp(- (double) 0.5*r_abs_sq*sigma*sigma);


	}

	@Override
	public ParallelVolumeOperation clone() {
		InitializeGaussian clone = new InitializeGaussian();
		clone.fMax = fMax;
		clone.fDelta = fDelta;
		clone.sigma = sigma;
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

	public float getSigma() {
		return sigma;
	}

	public void setSigma(float sigma) {
		this.sigma = sigma;
	}


}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/