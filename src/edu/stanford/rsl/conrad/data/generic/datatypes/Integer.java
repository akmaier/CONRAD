/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.datatypes;

public class Integer extends Gridable<Integer>{

	private int data = 0;

	public Integer () {
		data=0;
	}

	public Integer(int val) {
		data=val;
	}

	public Integer(Integer input) {
		data = input.data;
	}

	public Integer add(Integer op) {
		return new Integer(op.data+this.data);
	}

	public Integer sub(Integer op) {
		return new Integer(this.data-op.data);
	}

	public Integer mul(Integer op) {
		return new Integer(this.data*op.data);
	}

	public Integer div(Integer op) {
		return new Integer(this.data/op.data);
	}

	public Integer min(Integer op){
		if(this.data < op.data)
			return new Integer(this);
		else
			return new Integer(op);
	}

	public Integer max(Integer op){
		if(this.data < op.data)
			return new Integer(this);
		else
			return new Integer(op);
	}

	public String toString() {
		return ""+data;
	}

	@Override
	public int compareTo(Integer arg0) {
		if (this.data < arg0.data)
			return -1;
		else if (this.data > arg0.data)
			return 1;
		else
			return 0;
	}

	@Override
	public Integer getNewInstance() {
		return new Integer();
	}

	@Override
	public Integer clone() {
		return new Integer(this);
	}
	
	@Override
	public float[] getAsFloatArray() {
		return new float[]{data};
	}

	@Override
	public Integer deriveFromFloatArray(float[] input) {
		return new Integer((int)input[0]);
	}

	@Override
	public Integer add(double in1) {
		return new Integer((int)(data+in1));
	}

	@Override
	public Integer sub(double in1) {
		return new Integer((int)(data-in1));
	}

	@Override
	public Integer mul(double in1) {
		return new Integer((int)(data*in1));
	}

	@Override
	public Integer div(double in1) {
		return new Integer((int)(data/in1));
	}
}
