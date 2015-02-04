/*
 * Copyright (C) 2010-2014 - Andreas Maier, Andreas Keil 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data;


/**
 * The Grid*D classes are to be used as wrappers for already allocated float buffers.
 * A Grid then defines the properties of such a buffer (like the spacing and origin of
 * the grid, how much of it is a boundary, ...) and provides some methods like interpolation
 * and boundary filling. The Grids also keep a reference to their buffer. Note that a
 * Grid's buffer, as well as its number of boundary pixels are fixed after the creation of
 * this object. Therefore, whenever any size parameter should change, a new Grid has to
 * be instantiated.
 * 
 * @author Andreas Keil
 */
public abstract class Grid {
	
	/** The Grid's size (excluding boundary pixels). */
	protected int[] size;
	/** The pixel spacing, i.e. the distance between two neighboring pixel centers. */
	protected double[] spacing;
	/** The Grid's origin, given in real world units (usually mm) of the first pixel's center (excluding borders). */
	protected double[] origin;

	
	/**
	 * @return The array's size (excluding borders).
	 */
	public int[] getSize() {
		return this.size;
	}

	/**
	 * @return The array's spacings in all dimensions.
	 */
	public double[] getSpacing() {
		return this.spacing;
	}
	/**
	 * Set the array's spacings.
	 */
	public void setSpacing(double... spacing) {
		assert spacing.length == this.size.length : "The given spacing's dimension (" + spacing.length + ") does not match this Grid's dimension!";
		for (double s : spacing) assert s >= 0.0 : "Spacing values have to be non-negative!";
		if(this.spacing == null) this.spacing = new double[size.length];
		System.arraycopy(spacing, 0, this.spacing, 0, spacing.length);
	}

	/**
	 * @return The array origin's world coordinates, given in real world units (usually mm) of the first pixel's center (excluding borders), in all dimensions.
	 */
	public double[] getOrigin() {
		return this.origin;
	}
	/**
	 * Set the array origin's world coordinates, given in real world units (usually mm) of the first pixel's center (excluding borders), in all dimensions.
	 */
	public void setOrigin(double... origin) {
		assert origin.length == this.size.length : "The given origin's dimension (" + origin.length + ") does not match this Grid's dimension!";
		if(this.origin == null) this.origin = new double[size.length];
		System.arraycopy(origin, 0, this.origin, 0, origin.length);
	}
	
	/**
	 * Show the object in an imageJ window
	 */
	public abstract void show();
	
	/**
	 * Show the object in an imageJ window with title
	 */
	public abstract void show(String s);
	
	/**
	 * Deep copy the object
	 */
	public abstract Grid clone();
	
	
	/**
	 * Get number of float elements
	 */
	public int getNumberOfElements(){
		int elements = this.size[0];
		for (int i=1; i < this.size.length; ++i)
			elements*=this.size[i];
		return elements;
	}
	

	
//	// THE FOLLOWING METHODS TAKE A LONGER TIME TO EXECUTE THAN THEIR DIMENSION-SPECIFIC IMPLEMENTATIONS
//	// (ABOUT 20x-30x LONGER IN SOME TEST ANDREAS KEIL DID).  THAT'S WHY THEY ARE COMMENTED OUT.
//	public double[] indexToPhysical(double... index) {
//		assert index.length == this.size.length : "The given index's dimension (" + index.length + ") does not match this Grid's dimension!";
//		double[] physical = new double[index.length];
//		for (int i = 0; i < this.size.length; ++i) physical[i] = index[i] * this.spacing[i] + this.origin[i];  
//		return physical;
//	}
//	public double[] physicalToIndex(double... physical) {
//		assert physical.length == this.size.length : "The given coordinate's dimension (" + physical.length + ") does not match this Grid's dimension!";
//		double[] index = new double[physical.length];
//		for (int i = 0; i < this.size.length; ++i) index[i] = (physical[i] - this.origin[i]) / this.spacing[i];  
//		return index;
//	}

	/**
	 * Serialize the grid's content to a String.
	 * @return the serialized String, created so that each dimension is wrapped in brackets and entries within one dimension are separated by commas, e.g., [[a, b, c], [d, e, f]]
	 */
	@Override
	public abstract String toString();
	
}


