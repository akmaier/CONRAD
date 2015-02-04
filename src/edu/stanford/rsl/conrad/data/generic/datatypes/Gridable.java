/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.datatypes;


public abstract class Gridable<T> implements Comparable<T>, OpenCLable<T> {
	
	protected T defaultVariable = null;
	
	public abstract T add(T in1);
	public abstract T sub(T in1);
	public abstract T mul(T in1);
	public abstract T div(T in1);
	public abstract T min(T in1);
	public abstract T max(T in1);
	public abstract T add(double in1);
	public abstract T sub(double in1);
	public abstract T mul(double in1);
	public abstract T div(double in1);
	public abstract T clone();
	public abstract T getNewInstance();
}
