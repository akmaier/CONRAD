package edu.stanford.rsl.conrad.data.generic.datatypes;

/** Copyright (c) 2013 the authors listed at the following URL, and/or
the authors of referenced articles or incorporated external code:
http://en.literateprograms.org/Complex_numbers_(Java)?action=history&offset=20120912160954

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Complex_numbers_(Java)?oldid=18617
 */

public class Complex extends Gridable<Complex>{

	private final double[] data = new double[2];

	public Complex () {
		data[0] = 0;
		data[1] = 0;
	}

	public Complex(double re, double im) {
		data[0] = re;
		data[1] = im;
	}

	public Complex(float[] realImagArray, int offset){
		data[0] = realImagArray[offset];
		data[1] = realImagArray[offset+1];
	}

	public static Complex fromPolar(double magnitude, double angle) {
		Complex result = new Complex();
		result.setReal(magnitude * Math.cos(angle));
		result.setImag(magnitude * Math.sin(angle));
		return result;
	}

	public Complex(Complex input) {
		data[0] = input.getReal();
		data[1] = input.getImag();
	}
	public double getReal() {
		return data[0];
	}

	public double getImag() {
		return data[1];
	}

	public void setReal(double re) {
		data[0] = re;
	}

	public void setImag(double im) {
		data[1] = im;
	}   

	public Complex add(Complex op) {
		Complex result = new Complex();
		result.setReal(data[0] + op.getReal());
		result.setImag(data[1] + op.getImag());
		return result;
	}

	public Complex add(double op) {
		Complex result = new Complex();
		result.setReal(data[0] + op);
		result.setImag(data[1]);
		return result;
	}

	public Complex sub(Complex op) {
		Complex result = new Complex();
		result.setReal(data[0] - op.getReal());
		result.setImag(data[1] - op.getImag());
		return result;
	}

	public Complex sub(double op) {
		Complex result = new Complex();
		result.setReal(data[0] - op);
		result.setImag(data[1]);
		return result;
	}

	public Complex mul(Complex op) {
		Complex result = new Complex();
		result.setReal(data[0] * op.getReal() - data[1] * op.getImag());
		result.setImag(data[0] * op.getImag() + data[1] * op.getReal());
		return result;
	}

	public Complex mul(double op) {
		Complex result = new Complex();
		result.setReal(data[0] * op);
		result.setImag(data[1] * op);
		return result;
	}

	public Complex div(Complex op) {
		Complex result = new Complex(this);
		result = result.mul(op.getConjugate());
		double opNormSq = op.getReal()*op.getReal()+op.getImag()*op.getImag();
		result.setReal(result.getReal() / opNormSq);
		result.setImag(result.getImag() / opNormSq);
		return result;
	}

	public Complex div(double op) {
		Complex result = new Complex();
		result.setReal(data[0] / op);
		result.setImag(data[1] / op);
		return result;
	}

	public Complex min(Complex op){
		if(this.getMagn() < op.getMagn())
			return new Complex(this);
		else
			return new Complex(op);
	}

	public Complex max(Complex op){
		if(this.getMagn() > op.getMagn())
			return new Complex(this);
		else
			return new Complex(op);
	}

	public double getMagn() {
		return Math.sqrt(data[0] * data[0] + data[1] * data[1]);
	}

	public double getAngle() {
		return Math.atan2(data[1], data[0]);
	}

	public Complex getConjugate() {
		return new Complex(data[0], data[1] * (-1));
	}

	public String toString() {
		if (data[0] == 0) {
			if (data[1] == 0) {
				return "0";
			} else {
				return (data[1] + "i");
			}
		} else {
			if (data[1] == 0) {
				return String.valueOf(data[0]);
			} else if (data[1] < 0) {
				return(data[0] + "" + data[1] + "i");
			} else {
				return(data[0] + "+" + data[1] + "i");
			}
		}
	}
	
	@Override
	public int compareTo(Complex arg0) {
		if (this.getMagn() < arg0.getMagn())
			return -1;
		else if (this.getMagn()  > arg0.getMagn())
			return 1;
		else
			return 0;
	}

	public static void main(String argv[]) {
		Complex a = new Complex(3, 4);
		Complex b = new Complex(1, -100);
		System.out.println(a.getMagn());
		System.out.println(b.getAngle());
		System.out.println(a.mul(b));
		System.out.println(a.div(b));
		System.out.println(a.add(b).mul(b));
	}

	@Override
	public Complex getNewInstance() {
		return new Complex(0, 0);
	}

	@Override
	public Complex clone() {
		return new Complex(this);
	}

	@Override
	public float[] getAsFloatArray() {
		return new float[]{(float)data[0],(float)data[1]};
	}

	@Override
	public Complex deriveFromFloatArray(float[] input) {
		return new Complex(data[0],data[1]);
	}

}
