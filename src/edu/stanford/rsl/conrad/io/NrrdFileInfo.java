/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.io;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.io.FileInfo;

public class NrrdFileInfo extends FileInfo {
	public int dimension=0;
	public int[] sizes;
	public String encoding="";
	public String[] centers=null;
	public double[] spacing;
	private SimpleVector spaceOrigin;
	private SimpleMatrix spaceDirections;
	
	/**
	 * Getter for the space directions. Assures that both, the space direction matrix or the spacing array have been allocated and set.
	 * If either of those is null, the default value which is one is used for all directions.
	 * @return The space direction matrix.
	 */
	public SimpleMatrix getSpaceDirections() {
		if(spacing == null){
			this.spacing = new double[dimension];
			for(int i = 0; i < dimension; i++){
				spacing[i] = 1;
			}
		}
		if(this.spaceDirections == null){
			this.spaceDirections = new SimpleMatrix(dimension, dimension);
			for(int i = 0; i < dimension; i++){
				spaceDirections.setElementValue(i, i, spacing[i]);
			}
		}
		return spaceDirections;
	}
	
	public void setSpaceDirections(SimpleMatrix spaceDirections) {
		this.spaceDirections = spaceDirections;
	}
	
	/**
	 * Getter for the space origin. If it hasn't been set before, uses the default value which is (0,0,0).
	 * @return
	 */
	public SimpleVector getSpaceOrigin() {
		if(spaceOrigin == null){
			spaceOrigin = new SimpleVector(dimension);
		}
		return spaceOrigin;
	}
	public void setSpaceOrigin(SimpleVector spaceOrigin) {
		this.spaceOrigin = spaceOrigin;
	}

	// Additional compression modes for fi.compression
	public static final int GZIP = 1001;
	public static final int ZLIB = 1002;
	public static final int BZIP2 = 1003;
	
	// Additional file formats for fi.fileFormat
	public static final int NRRD = 1001;
	public static final int NRRD_TEXT = 1002;
	public static final int NRRD_HEX = 1003;
	
	
	

}
