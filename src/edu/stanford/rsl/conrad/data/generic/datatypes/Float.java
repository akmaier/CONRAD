/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.datatypes;

public class Float extends Gridable<Float>{

	private float data = 0;

	public Float () {
		data=0;
	}

	public Float(float val) {
		data=val;
	}

	public Float(Float input) {
		data = input.data;
	}

	public Float add(Float op) {
		return new Float(op.data+this.data);
	}

	public Float sub(Float op) {
		return new Float(this.data-op.data);
	}

	public Float mul(Float op) {
		return new Float(this.data*op.data);
	}

	public Float div(Float op) {
		return new Float(this.data/op.data);
	}

	public Float min(Float op){
		if(this.data < op.data)
			return new Float(this);
		else
			return new Float(op);
	}

	public Float max(Float op){
		if(this.data < op.data)
			return new Float(this);
		else
			return new Float(op);
	}

	public String toString() {
		return ""+data;
	}

	@Override
	public int compareTo(Float arg0) {
		if (this.data < arg0.data)
			return -1;
		else if (this.data > arg0.data)
			return 1;
		else
			return 0;
	}

	@Override
	public Float getNewInstance() {
		return new Float();
	}

	@Override
	public Float clone() {
		return new Float(this);
	}

	@Override
	public float[] getAsFloatArray() {
		return new float[]{data};
	}

	@Override
	public Float deriveFromFloatArray(float[] input) {
		return new Float(input[0]);
	}

	@Override
	public Float add(double in1) {
		return new Float((int)(data+in1));
	}

	@Override
	public Float sub(double in1) {
		return new Float((int)(data-in1));
	}

	@Override
	public Float mul(double in1) {
		return new Float((int)(data*in1));
	}

	@Override
	public Float div(double in1) {
		return new Float((int)(data/in1));
	}
}
