package edu.stanford.rsl.conrad.utils;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.HashMap;


public abstract class DoubleArrayUtil {

	private static HashMap<Integer, double[]> arrayBuffer = new HashMap<Integer, double[]>();
	private static int max = -1;

	/**
	 * Stores an array for later visualization at index imageNumber
	 * @param imageNumber the number
	 * @param array the array
	 */
	public static void saveForVisualization(int imageNumber, double [] array){
		arrayBuffer.put(new Integer(imageNumber), array);
		if (imageNumber > max) max = imageNumber;
	}

	/**
	 * Performs a 1-D convolution of the input array with the kernel array.<BR>
	 * New array will be only of size <br>
	 * <pre>
	 * output.lenght = input.length - (2 * (kernel.length/2));
	 * </pre>
	 * (Note that integer arithmetic is used here)<br>
	 * @param input the array to be convolved
	 * @param kernel the kernel
	 * @return the output array.
	 */
	public static double [] convolve(double [] input, double [] kernel){
		int offset = ((kernel.length) / 2);
		double [] revan = new double [input.length - (2* offset)];
		double weightSum = 0;
		for (int j = 0; j < kernel.length; j++){
			weightSum += kernel[j];
		}
		if (weightSum == 0) weightSum = 1; 
		for (int i = offset; i < input.length-offset;i++){
			double sum = 0;
			for (int j = -offset; j <= offset; j++){
				sum += kernel[offset+j] * input[i+j];
			}
			sum /= weightSum;
			revan [i-offset] = sum;
		}
		return revan;
	}

	/**
	 * Displays the arrays stored with "saveForVisualization" as ImagePlus.
	 * @param title the title of the ImagePlus
	 * @return the reference to the ImagePlus
	 * 
	 * @see #saveForVisualization(int imageNumber, double [] array)
	 */
	public static ImagePlus visualizeBufferedArrays(String title){
		if (max >= 0) {
			int height = max+1;
			int width = arrayBuffer.get(new Integer(0)).length;
			FloatProcessor flo = new FloatProcessor(width, height);
			for (int j = 0; j <= max; j++){
				double [] array = arrayBuffer.get(new Integer(j));
				for (int i = 0; i < array.length; i++){
					flo.putPixelValue(i, max - j, array[i]);
				}
			}
			return VisualizationUtil.showImageProcessor(flo, title);
		} else {
			return null;
		}
	}

	/**
	 * Forces an complex double array to be symmetric. Left / first half of the array is mirrored to the right / second half
	 * @param array the complex array
	 */
	public static void forceSymmetryComplexDoubleArray(double [] array){
		// Force Symmetry
		int width = array.length / 2;
		for (int i = 0; i < (width/2); i++){
			array[(width) + (2 * i)] = array[(width)-(2*i)];
			array[(width) + (2 * i)+1] = array[(width)-(2*i)+1];
		}
	}

	/**
	 * tests if any of the values in the given array is NaN
	 * @param array
	 * @return true if the array contains at least one entry with NaN
	 */
	public static boolean isNaN(double [] array){
		boolean revan = true;
		for (int i = 0; i < array.length; i++){
			if (revan && Double.isNaN(array[i])) revan = false;
		}
		return !revan;
	}

	/**
	 * Forces a real double array to be symmetric. Left / first half of the array is mirrored to the right / second half
	 * @param array the real array
	 */
	public static void forceSymmetryRealDoubleArray(double [] array){
		// Force Symmetry
		int width = array.length;
		for (int i = 0; i < (width/2); i++){
			array[(width/2) + (i)] = array[(width/2)-(i)];
		}
	}

	/**
	 * returns the closest index in the array to the given value
	 * @param x the value
	 * @param array the array
	 * @return the desired index in the array
	 */
	public static int findClosestIndex(double x, double [] array){
		double min = Double.MAX_VALUE;
		int pos = 0;
		for (int i = 0; i < array.length; i++){
			double dist = Math.abs(array[i]-x);
			if (dist < min){
				min = dist;
				pos = i;
				if (dist == 0) break;
			}
		}
		return pos;
	}
	
	
	/**
	 * computes the covariance between two arrays
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the correlation
	 */
	public static double covarianceOfDoubleArrays(double [] x, double [] y){
		double meanX = computeMean(x);
		double meanY = computeMean(y);
		double covariance = 0;
		for (int i=0; i< x.length; i++){
			covariance += ((x[i] - meanX) * (y[i] - meanY)) / x.length;
		}
		return covariance / x.length;
	}

	/**
	 * computes the correlation coefficient between two arrays after Pearson
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the correlation
	 */
	public static double correlateDoubleArrays(double [] x, double [] y){
		double meanX = computeMean(x);
		double meanY = computeMean(y);
		double covariance = 0;
		double varX = 0, varY = 0;
		for (int i=0; i< x.length; i++){
			varX += Math.pow(x[i] - meanX, 2) / x.length;
			varY += Math.pow(y[i] - meanY, 2) / y.length;
			covariance += ((x[i] - meanX) * (y[i] - meanY)) / x.length;
		}
		if (varX == 0) varX = CONRAD.SMALL_VALUE;
		if (varY == 0) varY = CONRAD.SMALL_VALUE;
		return covariance / (Math.sqrt(varX) * Math.sqrt(varY));
	}
	
	/**
	 * computes the correlation coefficient between two arrays after Pearson
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the correlation
	 */
	public static double computeSSIMDoubleArrays(double [] x, double [] y){
		double meanX = computeMean(x);
		double meanY = computeMean(y);
		double covariance = 0;
		double varX = 0, varY = 0;
		for (int i=0; i< x.length; i++){
			varX += Math.pow(x[i] - meanX, 2) / x.length;
			varY += Math.pow(y[i] - meanY, 2) / y.length;
			covariance += ((x[i] - meanX) * (y[i] - meanY)) / x.length;
		}
		if (varX == 0) varX = CONRAD.SMALL_VALUE;
		if (varY == 0) varY = CONRAD.SMALL_VALUE;
		return (2*covariance *2*meanX*meanY) / ((varX+varY) * (Math.pow(meanY, 2) + Math.pow(meanX, 2)));
	}

	
	/**
	 * computes the concordance correlation coefficient between two arrays
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the correlation
	 */
	public static double concordanceCorrelateDoubleArrays(double [] x, double [] y){
		double meanX = computeMean(x);
		double meanY = computeMean(y);
		double covariance = 0;
		double varX = 0, varY = 0;
		for (int i=0; i< x.length; i++){
			varX += Math.pow(x[i] - meanX, 2) / x.length;
			varY += Math.pow(y[i] - meanY, 2) / y.length;
			covariance += ((x[i] - meanX) * (y[i] - meanY)) / x.length;
		}
		if (varX == 0) varX = CONRAD.SMALL_VALUE;
		if (varY == 0) varY = CONRAD.SMALL_VALUE;
		return (2* covariance) / (varX + varY + Math.pow(meanX - meanY, 2));
	}
	
	/**
	 * computes the mean square error of array x to array y
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the mean square error
	 */
	public static double computeMeanSquareError(double [] x, double [] y){
		double sum = 0;
		for (int i=0; i< x.length; i++){
			sum += Math.pow(x[i] - y[i], 2);
		}
		return sum / x.length;
	}
	
	/**
	 * computes the root mean square error of array x to array y
	 *
	 * @param x the one array
	 * @param y the other array
	 * @return the root mean square error.
	 */
	public static double computeRootMeanSquareError(double [] x, double [] y){
		return Math.sqrt(computeMeanSquareError(x, y));
	}

	public static void suppressCenter(double [] weights, int threshold){
		for (int i = 1; i < weights.length; i++){
			if (!((i < threshold) || ((weights.length - i) < threshold))) {
				weights[i] = (weights[threshold] + weights[weights.length-threshold]) /2;
			}
		}
	}

	/**
	 * Removes outliers from the array which differ more than threshold from the last value.
	 * @param weights the weight
	 * @param threshold the threshold
	 */
	public static void removeOutliers(double [] weights, double threshold){
		for (int i = 1; i < weights.length; i++){
			if (Math.abs(weights[i] - weights[i - 1]) > threshold) weights[i] = weights[i - 1] + threshold * Math.signum(- weights[i - 1] + weights[i]);
		}
	}

	/**
	 * computes the mean of the array "values" on the interval [start, end].
	 * @param values the array
	 * @param start the start index
	 * @param end the end index
	 * @return the mean value
	 */
	public static double computeMean (double [] values, int start, int end){
		double revan = 0;
		for(int i = start; i <= end; i++){
			revan += values[i];
		}
		revan /= end - start + 1;
		return revan;
	}

	/**
	 * Computes the average increment of the array
	 * @param array the array
	 * @return the average increment
	 */
	public static double computeAverageIncrement(double [] array){
		double increment = 0;
		for (int i = 1; i < array.length; i++){
			double value = Math.abs(array[i-1] - array[i]);
			if (value > 180) { 
				value -= 360;
				value = Math.abs(value);
			}
			increment += value;
		}
		return increment / (array.length - 1);
	}

	/**
	 * Performs mean filtering of the array.
	 * @param weights the array
	 * @param context the context to be used for smoothing (from -context/2 to context/2)
	 * @return the smoothed array
	 */
	public static double[] meanFilter(double [] weights, int context){
		double meanFiltered [] = new double [weights.length];
		double mean = 0;
		for (int i = 0; i < weights.length; i++) {
			if (i > context / 2){
				mean -= weights[i - (context/2)];
			} if (i < (weights.length -1) - (context / 2)){
				mean += weights[i + (context /2)];
			}
			if (i < context/2){
				meanFiltered[i] = computeMean(weights, 0, i);
			} else if ((i+ context/2) >= weights.length){
				meanFiltered[i] = computeMean(weights, i, weights.length - 1);
			} else {
				meanFiltered[i] = computeMean(weights, i - context/2, i + context/2);
			}
		}
		return meanFiltered;
	}

	/**
	 * Gaussian smoothing of the elements of the array "weights"
	 * @param weights the array
	 * @param sigma the standard deviation
	 * @return the smoothed array
	 */
	public static double[] gaussianFilter(double [] weights, double sigma){
		double meanFiltered [] = new double [weights.length];
		int center = (int) Math.floor(sigma * 1.5) + 1;
		double kernel [] = new double [(int) Math.ceil(center*2 +1)];
		double kernelSum = 0;
		for (int j = 0; j < (center*2) + 1; j++){
			kernel[j] = Math.exp(-0.5 * Math.pow((center-j) / sigma, 2))/sigma/Math.sqrt(2*Math.PI);
			kernelSum += kernel[j];
		}
		for (int i =0; i< meanFiltered.length; i++){
			double sum = 0;		
			for (int j = -center; j <= center; j++){
				// Out of bounds at left side
				if (i+j < 0)
					sum += kernel[j+center] * weights[0];
				// Out of bounds at right side
				else if (i+j > weights.length-1)
					sum += kernel[j+center] * weights[weights.length-1];
				// Convolution applied inside the valid part of the signal
				else
					sum += kernel[j+center] * weights[i+j];
			}
			meanFiltered[i] = sum / kernelSum;
		}
		return meanFiltered;
	}

	/**
	 * Computes the standard deviation given an array and its mean value
	 * @param array the array
	 * @param mean the mean value of the array
	 * @return the standard deviation
	 */
	public static double computeStddev(double[] array, double mean){
		double stddev = 0;
		for (int i = 0; i < array.length; i++){	
			stddev += Math.pow(array[i] - mean, 2);
		}
		return Math.sqrt(stddev / array.length);
	}

	/**
	 * Computes the mean value of a given array
	 * @param array the array
	 * @return the mean value as double
	 */
	public static double computeMean(double[] array){
		double mean = 0;
		for (int i = 0; i < array.length; i++){	
			mean += array[i];
		}
		return mean / array.length;
	}
	
	/**
	 * Computes the median value of a given array
	 * @param array the array
	 * @return the median value as double
	 */
	public static double computeMedian(double[] array){
		double [] sorted = Arrays.copyOf(array, array.length);
		Arrays.sort(sorted);
		return sorted[sorted.length/2];
	}

	/**
	 * Returns the minimal and the maximal value in a given array
	 * @param array the array
	 * @return an array with minimal and maximal value
	 */
	public static double [] minAndMaxOfArray(double [] array){
		double [] revan = new double [2];
		revan[0] = Double.MAX_VALUE;
		revan[1] = -Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++){
			if (array[i] < revan[0]) {
				revan[0] = array[i];
			}
			if (array[i] > revan[1]) {
				revan[1] = array[i];
			}
		}
		return revan;		
	}

	/**
	 * Returns the minimal value in a given array
	 * @param array the array
	 * @return the minimal value
	 */
	public static double minOfArray(double [] array){
		double min = Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++){
			if (array[i] < min) {
				min = array[i];
			}

		}
		return min;		
	}
	
	/**
	 * Returns the maximal value in a given array
	 * @param array the array
	 * @return the minimal value
	 */
	public static double maxOfArray(double [] array){
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < array.length; i++){
			if (array[i] > max) {
				max = array[i];
			}

		}
		return max;		
	}


	/**
	 * forces monotony onto the input array
	 * @param array the array
	 * @param rising force rising monotony?
	 */
	protected void forceMonotony(double [] array, boolean rising){
		double lastValid = array[0];
		for (int i=0;i< array.length; i++){
			double value = array[i];
			if (rising) {
				if (value < lastValid) {
					value = lastValid;
				} else {
					lastValid = value;
				}
			} else {
				if (value > lastValid) {
					value = lastValid;
				} else {
					lastValid = value;
				}
			}
			array[i] = value;
		}
	}

	/**
	 * Adds one array to the first array
	 * @param sum the first array
	 * @param toAdd the array to add
	 */
	public static void add(double[] sum, double[] toAdd) {
		for (int i =0; i < sum.length; i++){
			sum[i] += toAdd[i];
		}
	}

	/**
	 * Adds a constant to the first array
	 * @param sum the first array
	 * @param toAdd the constant to add
	 */
	public static double [] add(double[] sum, double toAdd) {
		for (int i =0; i < sum.length; i++){
			sum[i] += toAdd;
		}
		return sum;
	}

	/**
	 * Divides all entries of array by divident.
	 * @param array the array
	 * @param divident the number used for division.
	 */
	public static double [] divide(double[] array, double divident) {
		for (int i =0; i < array.length; i++){
			array[i] /= divident;
		}
		return array;
	}

	/**
	 * Multiplies all entries of array by factor.
	 * @param array the array
	 * @param factor the number used for multiplication.
	 */
	public static double [] multiply(double[] array, double factor) {
		for (int i =0; i < array.length; i++){
			array[i] *= factor;
		}
		return array;
	}

	/**
	 * Multiplies all entries of the two arrays element by element.<bR>
	 * Works in place and overwrites array.
	 * @param array the array
	 * @param array2 the other array.
	 */
	public static void multiply(double[] array, double[] array2) {
		for (int i =0; i < array.length; i++){
			array[i] *= array2[i];
		}
	}

	/**
	 * Uses Math.exp() on all elements of the array
	 * Works in place and overwrites array.
	 * @param array the array
	 */
	public static void exp(double[] array) {
		for (int i =0; i < array.length; i++){
			array[i] = Math.exp(array[i]);
		}
	}

	/**
	 * Divides all entries of the two arrays element by element.<bR>
	 * Works in place and overwrites array.
	 * @param array the array
	 * @param divident the other array.
	 */
	public static double [] divide(double[] array, double[] divident) {
		for (int i =0; i < array.length; i++){
			array[i] /= divident[i];
		}
		return array;
	}

	/**
	 * Uses Math.log() on all elements of the array
	 * Works in place and overwrites array.
	 * @param array the array
	 */
	public static void log(double[] array) {
		for (int i =0; i < array.length; i++){
			array[i] = Math.log(array[i]);
		}
	}

	/**
	 * Prints the contents of the double array on standard out.
	 * @param array
	 * @param nf the NumberFormat
	 */
	public static void print(double[] array, NumberFormat nf) {
		System.out.print("[");
		for (int i =0; i < array.length; i++){
			System.out.print(" " + nf.format(array[i]));
		}
		System.out.println(" ]");
	}

	/**
	 * Prints the array on standard out and denotes the arrays name.
	 * @param name the name
	 * @param array the array
	 * @param nf the number format
	 */
	public static void print(String name, double[] array, NumberFormat nf) {
		System.out.print(name + " = ");
		print(array, nf);
	}

	/**
	 * Prints the array on standard out. Uses NumberFormat.getInstance() for number formatting
	 * @param name the name 
	 * @param array the array
	 */
	public static void print(String name, double [] array){
		print(name, array, NumberFormat.getInstance());
	}

	/**
	 * Prints the array on standard out. Uses NumberFormat.getInstance() for number formatting
	 * @param array the array
	 */
	public static void print (double [] array){
		print(array, NumberFormat.getInstance());
	}

	/**
	 * calls Math.pow for each element of the array
	 * @param array
	 * @param exp the exponent.
	 * @return reference to the input array
	 */
	public static double [] pow(double[] array, double exp) {
		for (int i =0; i < array.length; i++){
			array[i] = Math.pow(array[i], exp);
		}
		return array;
	}

	public static double[] min(double [] array, double min) {
		for (int i =0; i < array.length; i++){
			array[i] = Math.min(array[i], min);
		}
		return array;
	}
	
	/**
	 * Converts the array to a String representation. Calls toString(array, " ").
	 * @param array the array
	 * @return the String representation
	 * @see #toString(double[],String)
	 */
	public static String toString(double [] array){
		return toString(array, " ");
	}

	/**
	 * Converts the array to a String representation. delimiter is used to connect the elements of the array.
	 * @param array the array
	 * @param delimiter the delimiter
	 * @return the String representation
	 */
	public static String toString(double[] array, String delimiter) {
		String revan = array[0] + delimiter;
		for(int i=1;i<array.length -1;i++){
			revan += array[i] + delimiter;
		}
		revan += array[array.length-1];
		return revan;
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/