package edu.stanford.rsl.conrad.numerics;


import java.io.Serializable;
import java.util.Scanner;
import java.util.Vector;


/**
 * @author Andreas Keil
 * @author Andreas Maier (only very few additions) 
 */
public class SimpleVector implements Serializable {

	private static final long serialVersionUID = -8563837337889104489L;

	
	/////////////////////////////////////////////////
	// Properties                                  //
	/////////////////////////////////////////////////
	
	protected int len;
	protected double[] buf;
	

	/////////////////////////////////////////////////
	// Constructors                                //
	/////////////////////////////////////////////////
	
	/**
	 * Creates a null vector. vector must be initialized before use;
	 */
	public SimpleVector() {
		this.init(0);
	}
	/**
	 * Creates a len dimensional vector
	 * @param len is dimension of vector.
	 */
	public SimpleVector(final int len) {
		this.init(len);
	}
	
	/**
	 * Creates a new vector from otherVec. The entries of this vector are element wise equal to other vec
	 * @param otherVec 
	 */
	public SimpleVector(final SimpleVector otherVec) {
		this.init(otherVec);
	}
	
	/**
	 * Creates a new vector initialized by an ordered list of numbers
	 * @param otherBuffer
	 */
	public SimpleVector(final double... otherBuffer) {
		this.init(otherBuffer);
	}
	
	/**
	 * Creates a new vector initialized by an ordered list of numbers (float)
	 * @param otherBuffer
	 */
	public SimpleVector(final float... otherBuffer) {
		this.init(otherBuffer);
	}
	
	/**
	 * Creates a new vector from a string equivalent
	 * @param str
	 */
	public SimpleVector(final String str) {
		this.init(str);
	}
	/**
	 * Creates a new vector from a jama equivalent
	 * @param other
	 */
	public SimpleVector(final Jama.Matrix other) {
		this(other.getColumnPackedCopy());
	}


	/////////////////////////////////////////////////
	// Methods                                     //
	/////////////////////////////////////////////////
	
	/**
	 * Initialize vector to [0,0,...,0] and length len.
	 * @param len is dimension of vector
	 */
	public void init(final int len) {
		assert len >= 0 : new IllegalArgumentException("Length has to be greater than or equal to zero!");
		if (this.len != len) {
			this.len = len;
			this.buf = new double[this.len];
		}
	}
	
	/**
	 * Initialize vector with otherVec
	 * @param otherVec
	 */
	public void init(final SimpleVector otherVec) {
		if (this.len != otherVec.len) {
			this.len = otherVec.len;
			this.buf = new double[this.len];
		}
		System.arraycopy(otherVec.buf, 0, this.buf, 0, this.len);
	}
	
	/**
	 * Initialize vector with a comma-separated list of double array
	 * @param otherBuffer
	 */
	public void init(final double... otherBuffer) {
		assert otherBuffer.length >= 0 : new IllegalArgumentException("Length has to be greater than or equal to zero!");
		this.len = otherBuffer.length;
		this.buf = new double[this.len];
		System.arraycopy(otherBuffer, 0, this.buf, 0, this.len);
	}
	
	/**
	 * Initialize vector with a comma-separated list of double array (float)
	 * Private: For internal use only
	 * @param otherBuffer
	 */
	public void init(final float... otherBuffer) {
		assert otherBuffer.length >= 0 : new IllegalArgumentException("Length has to be greater than or equal to zero!");
		this.len = otherBuffer.length;
		this.buf = new double[this.len];
		for (int i = 0; i < otherBuffer.length; i++) {
			buf[i] = otherBuffer[i];
		}
	}
	
	/**
	 * Initialize vector with string equivalent
	 * @param str
	 */
	public void init(final String str) {
		final String strTrim = str.trim();
		if ((strTrim.charAt(0) != '[') || (strTrim.charAt(strTrim.length() - 1) != ']')) throw new RuntimeException("Error parsing matrix string!");
		Scanner scanner = new Scanner(strTrim.substring(1, strTrim.length()-1).trim());
		scanner.useDelimiter("\\s*[\\s,;]\\s*");
		Vector<Double> list = new Vector<Double>();
		while (scanner.hasNext()) {
			// TODO: Getting the next number as String and then converting it to a double is a workaround for a locale-Mac-related conversion problem. Maybe replace this by
			// list.add(scanner.nextDouble());
			// once this issue is resolved in java.util.Scanner (if it is really a bug and not a feature)
			list.add(Double.parseDouble(scanner.next()));
		}
		this.init(list.size());
		for (int i = 0; i < list.size(); ++i)
			this.buf[i] = list.elementAt(i);
	}
	
	@Override
	public SimpleVector clone() {
		return new SimpleVector(this);
	}
	
	/**
	 * Copies the elements of this vector to a double array
	 * @return double array containing ordered values
	 */
	public double[] copyAsDoubleArray(){
		double[] array = new double[this.len];
		this.copyTo(array);
		return array; 
	}
	
	/**
	 * Copies the elements of this vector to a float array. Warning: Possible loss of Data !
	 * @return float array containing ordered values
	 */
	public float[] copyAsFloatArray(){
		float[] array = new float[this.len];
		this.copyTo(array);
		return array; 
	}
	
	/**
	 * Copies the element of this vector  to the double array provided
	 * @param other is array to be populated
	 */
	public void copyTo(final double[] other) {
		assert (this.len == other.length) : new IllegalArgumentException("Copying is only possible to an array of the same size!");
		System.arraycopy(this.buf, 0, other, 0, this.len);
	}
	
	/**
	 * Copies the element of this vector  to the float array provided
	 * @param other is array to be populated
	 */
	public void copyTo(final float[] other) {
		assert (this.len == other.length) : new IllegalArgumentException("Copying is only possible to an array of the same size!");
		for(int i = 0; i < this.getLen();i++)
		{
			other[i] = (float) this.buf[i];
		}
	}
	
	/**
	 * @return the dimension of this vector
	 */
	public int getLen() {
		return this.len;
	}
	
	/** Sets all entries to the given value. */
	public void fill(final double value) {
		java.util.Arrays.fill(this.buf, value);
	}
	
	/** Sets all entries to 0.0. */
	public void zeros() {
		this.fill(0.0);
	}
	
	/** Sets all entries to 1.0. */
	public void ones() {
		this.fill(1.0);
	}
	
	/** Rounds all entries to the next smaller integer */
	public void floor() {
		for (int i = 0; i < this.len; i++) {
			this.setElementValue(i, Math.floor(getElement(i)));
		}
	}
	
	/** Rounds all entries to the next higher integer */
	public void ceil() {
		for (int i = 0; i < this.len; i++) {
			this.setElementValue(i, Math.ceil(getElement(i)));
		}
	}

	/**
	 * Assigns random values to the entries of the vector.
	 * Values are uniformly distributed in the given interval [min, max).
	 * @param min  The lower bound of the interval the values are drawn from.
	 * @param max  The upper bound of the interval the values are drawn from. Note that value max
	 *             itself is excluded from the interval and therefore never assigned.
	 */
	public void randomize(final double min, final double max) {
		for (int i = 0; i < this.len; ++i)
			this.buf[i] = (max-min)*Math.random() + min;
	}
	
	/**
	 * Retrieve vector element at index i
	 * @param i index to be retrieved
	 * @return element at index i
	 */
	public double getElement(final int i) {
		return this.buf[i];
	}
	
	/**
	 * Replaces the vector element at index i
	 * @param i index to be replaced
	 * @param val is new value
	 */
	public void setElementValue(final int i, final double val) {
		this.buf[i] = val;
	}
	
	public SimpleVector getSubVec(final int firstInd, final int size) {
		final SimpleVector subVector = new SimpleVector(size);
		System.arraycopy(this.buf, firstInd, subVector.buf, 0, size);
		return subVector;
	}
	
	public void setSubVecValue(final int firstInd, final SimpleVector subVector) {
		System.arraycopy(subVector.buf, 0, this.buf, firstInd, subVector.len);
	}

	public void addToElement(final int i, final double addend) {
		this.buf[i] += addend;
	}
	public void subtractFromElement(final int i, final double subtrahend) {
		this.buf[i] -= subtrahend;
	}
	public void multiplyElementBy(final int i, final double factor) {
		this.buf[i] *= factor;
	}
	public void divideElementBy(final int i, final double divisor) {
		this.buf[i] /= divisor;
	}
	
	/**
	 * Method to add a scalar to this vector in place.
	 * @param addend the other vector
	 */
	public void add(final double addend) {
		for (int i = 0; i < this.len; i++){
			this.buf[i] += addend;
		}
	}
	
	public void subtract(final double subtrahend) {
		for (int i = 0; i < this.len; i++){
			this.buf[i] -= subtrahend;
		}
	}

	public void multiplyBy(final double factor) {
		for (int i = 0; i < this.len; ++i)
			this.buf[i] *= factor;
	}

	/**
	 * Returns a scaled instance of the vector. All elements in the scaled instance are multiplied by s
	 * @param factor the scalar
	 * @return the scaled instance
	 */
	public SimpleVector multipliedBy(final double factor) {
		final SimpleVector result = new SimpleVector(this.len);
		for (int i = 0; i < this.len; ++i)
			result.buf[i] = this.buf[i] * factor;
		return result;
	}
	
	public void divideBy(final double divisor) {
		this.multiplyBy(1.0 / divisor);
	}
	
	/**
	 * Returns a scaled instance of the vector. All elements in the scaled instance are divided by s
	 * @param divisor the scalar
	 * @return the scaled instance
	 */
	public SimpleVector dividedBy(final double divisor) {
		return this.multipliedBy(1.0 / divisor);
	}
	
	/**
	 * Method to add other vectors to this vector in place.
	 * @param addends  The other vectors.
	 */
	public void add(final SimpleVector... addends) {
		assert addends.length >= 1 : new IllegalArgumentException("At least one other vector has to be given!");
		for (SimpleVector addend : addends) {
			assert addend.len == this.len;
			for (int i = 0; i < this.len; ++i)
				this.buf[i] += addend.buf[i];
		}
	}
	
	public void subtract(final SimpleVector... subtrahends) {
		assert subtrahends.length >= 1 : new IllegalArgumentException("At least one other vector has to be given!");
		for (SimpleVector subtrahend : subtrahends) {
			assert subtrahend.len == this.len;
			for (int i = 0; i < this.len; ++i)
				this.buf[i] -= subtrahend.buf[i];
		}
	}

	public void multiplyElementWiseBy(final SimpleVector... factors) {
		assert factors.length >= 1 : new IllegalArgumentException("At least one other vector must be given as parameter!");
		for (SimpleVector factor : factors) {
			assert factor.len == this.len : new IllegalArgumentException("Vector lenghts must match!");
			for (int i = 0; i < this.len; ++i)
				this.buf[i] *= factor.buf[i];
		}
	}
	
	public void divideElementWiseBy(final SimpleVector... divisors) {
		assert divisors.length >= 1 : new IllegalArgumentException("At least one other vector must be given as parameter!");
		for (SimpleVector divisor : divisors) {
			assert divisor.len == this.len : new IllegalArgumentException("Vector lenghts must match!");
			for (int i = 0; i < this.len; ++i)
				this.buf[i] /= divisor.buf[i];
		}
	}

	public void negate() {
		for (int i = 0; i < this.len; ++i)
			this.buf[i] = -this.buf[i];
	}
	
	public SimpleVector negated() {
		final SimpleVector result = new SimpleVector(this.len);
		for (int i = 0; i < this.len; ++i)
			result.buf[i] = -this.buf[i];
		return result;
	}
	
	public void absolute() {
		for (int i = 0; i < this.len; ++i)
			this.buf[i] = Math.abs(this.buf[i]);
	}
	
	public SimpleVector absoluted() {
		final SimpleVector result = new SimpleVector(this.len);
		for (int i = 0; i < this.len; ++i)
			result.buf[i] = Math.abs(this.buf[i]);
		return result;
	}
	
	public double min() {
		double result = Double.MAX_VALUE;
		for (int i = 0; i < this.len; ++i)
			result = Math.min(result, this.buf[i]);
		return result;
	}
	
	public double max() {
		double result = Double.MIN_VALUE;
		for (int i = 0; i < this.len; ++i)
			result = Math.max(result, this.buf[i]);
		return result;
	}
	
	public SimpleMatrix transposed() {
		final SimpleMatrix result = new SimpleMatrix(1, this.len);
		for (int i = 0; i < result.cols; ++i)
			result.buf[0*result.cols+i] = this.buf[i];
		return result;
	}

	public static enum VectorNormType {
		/** A vectors' L_1 norm is the sum of its absolute values. */
		VEC_NORM_L1,
		/** A vectors' L_2 norm is the square root of the sum of squares of its entries. */
		VEC_NORM_L2,
		/** A vectors' L_infinity norm is the maximum of its absolute values. */
		VEC_NORM_LINF
	}
	public double norm(final VectorNormType normType) {
		double result = 0.0;
		switch (normType) {
		case VEC_NORM_L1:
			for (int i = 0; i < this.len; ++i)
				result += Math.abs(this.buf[i]);
			return result;
		case VEC_NORM_L2:
			for (int i = 0; i < this.len; ++i)
				result += this.buf[i]*this.buf[i];
			return Math.sqrt(result);
		case VEC_NORM_LINF:
			for (int i = 0; i < this.len; ++i) {
				double viAbs = Math.abs(this.buf[i]);
				if (viAbs > result) result = viAbs;
			}
			return result;
		default:
			throw new RuntimeException("Vector norm type not implemented yet!");
		}
	}

	public double normL2() {
		return this.norm(VectorNormType.VEC_NORM_L2);
	}
	
	public void normalizeL2() {
		this.divideBy(this.normL2());
	}
	
	public SimpleVector normalizedL2() {
		return this.dividedBy(this.normL2());
	}


	/////////////////////////////////////////////////
	// Serialization and Persistence               //
	/////////////////////////////////////////////////

	public String getVectorSerialization() {
		return this.toString();
	}
	
	public void setVectorSerialization(final String str) {
		this.init(str);
	}
	
	@Override
	public String toString() {
		String result = new String();
		result += "[";
		for (int i = 0; i < this.len; ++i) {
			if (i != 0) result += "; ";
			result += new Double(this.buf[i]);
		}
		result += "]";
		return result;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Kail, Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
