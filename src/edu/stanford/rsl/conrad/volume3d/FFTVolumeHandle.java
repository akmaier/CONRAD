package edu.stanford.rsl.conrad.volume3d;


/**
 * Class to wrap different FFT libraries
 * @author akmaier
 *
 */
public abstract class FFTVolumeHandle {

	protected boolean debug = false;

	public FFTVolumeHandle(VolumeOperator operator){
		this.operator = operator;
	}

	protected VolumeOperator operator;
	/**
	 * Performs a forward Fast Fourier Transform of the Volume
	 * @param vol the Volume
	 */
	public abstract void forwardTransform(Volume3D vol);

	/**
	 * Performs a normalized inverse Fast Fourier Transform of the Volume
	 * @param vol the Volume
	 */
	public abstract void inverseTransform(Volume3D vol);

	public void setVolumeOperator(VolumeOperator operator){
		this.operator = operator;
	}

	/**
	 * Cleans up the memory. Relevant for FFT implementations which depend on native code.
	 */
	public abstract void cleanUp();

	/**
	 * Sets the maximal number of threads used for the FFT.
	 * @param number
	 */
	public abstract void setThreadNumber(int number);

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/