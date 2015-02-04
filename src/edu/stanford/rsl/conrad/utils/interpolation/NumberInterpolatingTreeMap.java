package edu.stanford.rsl.conrad.utils.interpolation;

import java.util.TreeMap;

import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;

/**
 * This is a generic implementation of {@link LinearInterpolatingDoubleArray}
 * used in modeling an interpolated table. <div>In this implementation:</div>
 * <blockquote class="webkit-indent-blockquote"
 * style="margin: 0 0 0 40px; border: none; padding: 0px;"> <div> Keys represent
 * Input and Values represent the Output. </div></blockquote> <blockquote
 * class="webkit-indent-blockquote"
 * style="margin: 0 0 0 40px; border: none; padding: 0px;"> <div> Entries added
 * to this data structure are automatically sorted by key value.
 * </div></blockquote> <div>The default interpolation scheme used is this class
 * is linear, however developers can create custom interpolation schemes.</div>
 * 
 * @see LinearInterpolatingDoubleArray
 * @see Interpolator
 * @see TreeMap
 * @author Rotimi X Ojo *
 */
public class NumberInterpolatingTreeMap extends TreeMap<Number, Number> {

	private static final long serialVersionUID = -5328843102124736133L;
	private Interpolator interpolator = new LinearInterpolator();

	/**
	 * Returns the intepolated value of key
	 * @param key is the input to the intepolating function
	 * @return throws {@link RuntimeException} if provided key is out of bounds
	 */
	public Number interpolateValue(Number key) {
		if (containsKey(key)) {
			return get(key);
		}
		

		if ((key.doubleValue() < firstKey().doubleValue())
				|| (key.doubleValue() > lastKey().doubleValue())) {
			throw new RuntimeException("Cannot interpolate outside range: "
					+ key + " Range: [ " + firstKey() + ", " + lastKey() + " ]");
		}

		double dkey = key.doubleValue();
		Number floor = floorKey(dkey);
		Number ceiling = ceilingKey(dkey);

		double floorVal = get(floor).doubleValue();
		double ceilingVal = get(ceiling).doubleValue();

		interpolator.setXPoints(floor.doubleValue(), ceiling.doubleValue());
		interpolator.setYPoints(floorVal, ceilingVal);
		return interpolator.InterpolateYValue(key.doubleValue());
	}
	
	/**
	 * Set the interpolating scheme used by the class. A linear interpolator is used by default
	 * @param interpolator is interpolator used by the class
	 */
	public void setInterpolator(Interpolator interpolator) {
		this.interpolator = interpolator;
	}
	
	/**
	 * Retrieve the intepolator used by the class
	 * @return the interpolator used by the class
	 */
	public Interpolator getInterpolator() {
		return interpolator;
	}
	
	/**
	 * Insert a array of input and corresponding output to data structure. input-output pairs are automatially sorted by key value
	 * @param keys is an array of input values.
	 * @param values is corresponding array of output values.
	 */
	public void put(Number[] keys, Number[] values) {
		for (int i = 0; i < keys.length; i++) {
			put(keys[i], values[i]);
		}
	}
	
	/**
	 * Insert a array of input and corresponding output to data structure. input-output pairs are automatially sorted by key value
	 * @param keys is an array of input values.
	 * @param values is corresponding array of output values.
	 */
	public void put(double[] keys, double[] values) {
		for (int i = 0; i < keys.length; i++) {
			put(keys[i], values[i]);
		}
	}

}
/*
 * Copyright (C) 2010-2014 Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/