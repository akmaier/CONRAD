/*
 * Copyright (C) 2014 - Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.generic.datatypes;

public class Double extends Gridable<Double>{

	private double data = 0;

	public Double () {
		data=0;
	}

	public Double(double val) {
		data=val;
	}

	public Double(Double input) {
		data = input.data;
	}

	public Double add(Double op) {
		return new Double(op.data+this.data);
	}

	public Double sub(Double op) {
		return new Double(this.data-op.data);
	}

	public Double mul(Double op) {
		return new Double(this.data*op.data);
	}

	public Double div(Double op) {
		return new Double(this.data/op.data);
	}

	public Double min(Double op){
		if(this.data < op.data)
			return new Double(this);
		else
			return new Double(op);
	}

	public Double max(Double op){
		if(this.data < op.data)
			return new Double(this);
		else
			return new Double(op);
	}

	public String toString() {
		return ""+data;
	}

	@Override
	public int compareTo(Double arg0) {
		if (this.data < arg0.data)
			return -1;
		else if (this.data > arg0.data)
			return 1;
		else
			return 0;
	}

	@Override
	public Double getNewInstance() {
		return new Double();
	}

	@Override
	public Double clone() {
		return new Double(this);
	}

	@Override
	public float[] getAsFloatArray() {
		return new float[]{(float)data};
	}

	@Override
	public Double deriveFromFloatArray(float[] input) {
		return new Double(input[0]);
	}

	@Override
	public Double add(double in1) {
		return new Double((int)(data+in1));
	}

	@Override
	public Double sub(double in1) {
		return new Double((int)(data-in1));
	}

	@Override
	public Double mul(double in1) {
		return new Double((int)(data*in1));
	}

	@Override
	public Double div(double in1) {
		return new Double((int)(data/in1));
	}

}
