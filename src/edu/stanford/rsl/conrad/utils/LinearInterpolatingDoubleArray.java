package edu.stanford.rsl.conrad.utils;

/**
 * Class to perform linear interpolation along an arbitrary mesh given the function values at each mesh point.
 * 
 * @author akmaier
 *
 */
public class LinearInterpolatingDoubleArray {

	private double [] mesh;
	private double [] values;
	private double min;
	private double max;
	
	
	public LinearInterpolatingDoubleArray(){
		
	}
	/**
	 * Constructs the interpolating Array. Requires two double arrays.
	 * @param mesh the mesh with the gridding of the values
	 * @param values the values at each mesh point.
	 */
	public LinearInterpolatingDoubleArray(double [] mesh, double [] values){
		setMesh(mesh);
		setValues(values);				
	}
	
	protected void setValues(double[] values) {
		this.values = values;		
	}
	
	public double [] getValues() {
		return values;
		
	}

	protected void setMesh(double [] mesh){
		this.mesh = mesh;
		// Compute the minimum and the maximum of mesh.
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(mesh);
		min = minmax[0];
		max = minmax[1];
	}
	
	public double [] getMesh(){
		return mesh;
	}
	
	/**
	 * Interpolated the value at meshPoint from the given array.
	 * @param meshPoint the value to interpolate at
	 * @return the interpolated value
	 */
	public double getValue(double meshPoint){
		if ((meshPoint < min)||(meshPoint > max)){
			throw new RuntimeException("Cannot extrapolate outside mesh: " + meshPoint + " Range: [ " + min + ", " + max + " ]");
		}
		int lowerIndex = 0;
		for (int i = 1; i < mesh.length; i++){
			if (mesh[i] > meshPoint){
				lowerIndex = i - 1;
				break;
			}
		}
		// distance between mesh points
		double gridDist = mesh[lowerIndex+1] - mesh[lowerIndex];
		// distance between values
		double valueDist = values[lowerIndex+1] - values[lowerIndex];
		// distance to left grid point
		double distLeft = meshPoint - mesh[lowerIndex];
		// interpolated value;
		double revan = values[lowerIndex] + ((distLeft) * (valueDist/gridDist));
		return revan;
	}
	
	public int size() {
		return mesh.length;
	}
	public void setMap(double [] mesh, double [] values) {
		setMesh(mesh);
		setValues(values);			
	}
	

	
}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/