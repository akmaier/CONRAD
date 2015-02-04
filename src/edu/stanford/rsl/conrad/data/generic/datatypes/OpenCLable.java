/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.datatypes;

public interface OpenCLable<T> {
	float[] getAsFloatArray();
	T deriveFromFloatArray(float[] input);
}
