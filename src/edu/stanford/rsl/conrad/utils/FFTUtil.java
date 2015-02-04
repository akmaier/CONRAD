package edu.stanford.rsl.conrad.utils;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;


/**
 * This class is a wrapper for the FFT as implemented in JTransforms.
 * 
 * 
 * @author Andreas Maier
 * 
 */

public abstract class FFTUtil {

	public static void init1DFFT(int width){
		DoubleFFT_1D fft = new DoubleFFT_1D(FFTUtil.getNextPowerOfTwo(width));
		double [] test = new double [FFTUtil.getNextPowerOfTwo(width) * 2];
		fft.complexForward(test);
		fft.complexInverse(test, true);
	}
	
	public static double []  fft(double []  array){
		DoubleFFT_1D fft = new DoubleFFT_1D(FFTUtil.getNextPowerOfTwo(array.length));
		double [] test = new double [FFTUtil.getNextPowerOfTwo(array.length) * 2];
		for (int i = 0; i < array.length; i++){
			test[i*2] = array[i];
		}
		fft.complexForward(test);
		return test;
	}
	
	public static double []  fft(double []  array, int padding){
		DoubleFFT_1D fft = new DoubleFFT_1D(FFTUtil.getNextPowerOfTwo(array.length + padding));
		double [] test = new double [FFTUtil.getNextPowerOfTwo(array.length+padding) * 2];
		for (int i = 0; i < array.length; i++){
			test[i*2] = array[i];
		}
		fft.complexForward(test);
		return test;
	}

	/**
	 * Computes the complex signum for the given array. 
	 * @param array the array
	 */
	public static double [] complexSignum (double [] array){
		double [] revan = new double [array.length];
		for (int i=0;i<array.length/2;i++){
			double abs = abs(i, array);
			if (abs != 0) {
				revan[i*2] = array[i*2] / abs;
				revan[(i*2)+1] = array[(i*2)+1] / abs;
			} else {
				revan[i*2] = 0;
				revan[(i*2)+1] = 0;
			}
		}
		return revan;
	}

	/**
	 * Computes the signum for the given complex array. 
	 * @param array the array
	 */
	public static double [] cSignum (double [] array){
		double [] revan = new double [array.length];
		for (int i=0;i<array.length/2;i++){
			double abs = abs(i, array);
			if (abs != 0) {
				double val = 1;
				if (array[i*2] < 0 ||(array[i*2]==0 && array[(i*2)+1]<0)) val =-1;
				revan[i*2] = val;
				revan[(i*2)+1] = 0;
			} else {
				revan[i*2] = 0;
				revan[(i*2)+1] = 0;
			}
		}
		return revan;
	}


	/**
	 * Multiplies the array with the complex number defined as two double values;
	 * @param array the array
	 * @param realOne the real part of the complex number
	 * @param imagOne the imaginary part of the complex number
	 */
	public static void timesComplexNumber(double [] array, double realOne, double imagOne){
		for (int i=0;i<array.length/2;i++){
			double [] newValue = multiplyComplex(array[i*2], array[(i*2)+1], realOne, imagOne);
			array[i*2] = newValue[0];
			array[(i*2)+1] = newValue[1];
		}
	}

	public static double[] complexOneOverPiT(int width){
		double [] revan = new double [width];
		for(int i=0; i< width/2; i++){
			revan[i*2] = 1/((i-(width/4))*Math.PI);
			if ((i-(width/4))==0){
				revan[i*2] =10000;
			}
		}
		return revan;
	}

	public static double[] hilbertKernel(int width){
		double [] revan = new double [width];
		for(int i=0; i< width/2; i++){
			revan[i*2] = Math.signum(i-(width/4));
		}
		return revan;
	}

	/**
	 * Computes the 1-D Hilbert Transform of the given Array. Uses the Fast Fourier transform to compute the transform
	 * @param array the array
	 * @param nTimes the size of the zero padding:<pre>
	 *            n = 1 zero pad to next power of 2
	 *            n = 2 zero pad to 2 times the next power of 2</pre>
	 * @return the Hilbert transformed array
	 */
	public static double [] hilbertTransform(double [] array, int nTimes){
		int newDimension = FFTUtil.getNextPowerOfTwo(array.length);
		// Write the data to JTransforms complex format
		double [] fftData = new double[newDimension*2*nTimes];
		for (int i = 0 ; i < array.length; i++){
			fftData[i*2] = array[i];
		}
		// FFT
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension*nTimes);
		fft.complexForward(fftData);
		// compute Hilbert Transform
		for (int i=0;i<fftData.length/2;i++){
			double sign = Math.signum((fftData.length/4)-i);
			if (sign == 0) {
			} else {
				if (i==0) sign =0;
				if(i == (fftData.length/2) -1) sign =0;
				double real = sign * -1.0 * fftData[(2*i)+1]; 
				double imag = sign * 1.0 * fftData[(2*i)];
				fftData[2*i] = real;
				fftData[(2*i)+1] = imag;
			}
		}
		// iFFT
		fft.complexInverse(fftData, true);
		// rewrite as real array.
		double [] revan = new double[array.length];
		for (int i = 0 ; i < array.length; i++){
			revan[i] = 
				//abs( i, fftData);
				fftData[2*i];
		}
		return revan;
	}

	public static double [] discreteHilbertTransform(double [] array, int nTimes){
		double [] revan = new double [array.length];
		for (int i =0; i<array.length; i++){
			double sum = 0;
			if(i % 2 == 0){ // i even
				for (int j = 0; j < array.length; j++){
					if (j % 2 == 1){ // j odd
						sum += array[j] / (i - j);
					}
				}
				sum *= 2.0 / Math.PI;
			} else { // i odd
				for (int j = 0; j < array.length; j++){
					if (j % 2 == 0){ // j even
						sum += array[j] / (i - j);
					}
				}
				sum *= 2.0 / Math.PI;
			}
			revan[i] = sum;
		}
		return revan;
	}
	/**
	 * Exact finite discrete Hilbert transformation after Kak SC. Hilbert Transformation for discrete data. Int J Electronics 34(2):177-83. 1973. Eq. 14
	 * @param array the Array
	 * @param nTimes Approximation for infinity
	 * @return the array with the Hilbert transform
	 */
	public static double [] exactFiniteHilbertTransform(double [] array, int nTimes){
		double [] revan = new double [array.length];
		for (int n =0; n<array.length; n++){
			double sum1 = 0;
			for (int k = 0; k< array.length; k++){
				if (n-k !=0){
					double cos = Math.cos(Math.PI*(n-k));
					double sum2 = (1.0 - cos) / (Math.PI*(n-k));
					for (int p = 1; p < nTimes; p++){
						sum2 += (2.0 / Math.PI) *
						(((n-k)*(1.0-(Math.pow(-1.0, p) * cos))) / 
								(Math.pow(n-k,2) - (Math.pow(p,2)*Math.pow(array.length, 2)))
						);
					}
					sum1 += array[k] * sum2;
					//System.out.println(sum1);
				}
			}
			revan[n] = sum1;
		}
		return revan;
	}


	/**
	 * Estimates an inverse of a Ramp Filter. Works in place. The array in the argument list will be changed!
	 * @param filter the filter
	 * @param max the maximal scaling (1.0 is a good number)
	 * @return the inverted RampFilter
	 */
	public static double [] invertRampFilter(double [] filter, double max){
		double maximum = -Double.MAX_VALUE;
		for (int i=0; i< filter.length/2; i++){
			// Compute Inverse
			double value = 1.0 / abs(i, filter);
			// set to real part
			filter[2*i] = value;
			// set complex part to 0
			filter[(2*i) + 1] = 0;
			// update maximum if needed
			if (value > maximum) maximum = value;		
		}
		// compute scaling factor
		double scale = max / maximum;
		// scale the inverted filter
		boolean scaling = true;
		if (scaling) {
			for (int i=0; i< filter.length/2; i++){
				if (Double.isInfinite(filter[2*i])) {
					filter [2*i] = 0;
				} else {
					filter [2*i] *= scale;
				}
			}
		}
		return filter;
	}

	/**
	 * Removes a ramp filter from the ImageProcessor. Note that frequencies which were multiplied with 0 cannot
	 * be restored. This has to be estimated differently. (If you apply the same filter later on again, it should
	 * not matter anyway, as the same frequencies will be multiplied with 0 again.)
	 * 
	 * @param imp the filtered ImageProcessor
	 * @param ramp the ramp
	 * @param max the maximum for the inverted ramp filter. (try 1.0)
	 * @return the unfiltered ImageProcessor
	 */
	public static Grid2D removeRampFilter(Grid2D imp, RampFilter ramp, double max){
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = imp.getWidth();
		originalHeight = imp.getHeight();
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		// Write the data to JTransforms format
		double [] fftData = new double[newDimension*2];
		double [] filter = ramp.getRampFilter1D(newDimension);
		invertRampFilter(filter, max);
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		Grid2D revan = new Grid2D(originalWidth, originalHeight);
		revan.setOrigin(imp.getOrigin());
		revan.setSpacing(imp.getSpacing());
		for (int j = 0;  j < imp.getHeight(); j++){
			for (int i = 0 ; i < imp.getWidth(); i++){
				fftData[i*2] = imp.getPixelValue(i, j);
			}
			// Perform forward transform
			fft.complexForward(fftData);
			// Filter
			fftData = FFTUtil.multiplyAbsolute(fftData, filter);
			// Perform backward transform
			fft.complexInverse(fftData, true);
			for (int i = 0 ; i< revan.getWidth(); i++){
				double value = fftData[i*2];
				revan.putPixelValue(i, j, value);
			}
		}
		boolean adjust = false;
		if (adjust){
			double [] scale = new double [revan.getWidth()];
			double [] stats = new double [2];
			stats[0] = NumericPointwiseOperators.mean(revan);
			stats[1] = NumericPointwiseOperators.stddev(revan, stats[0]);
			System.out.println("Mean :"  + stats[0]);
			double avgBorder = 0;
			for (int i = 0; i < revan.getHeight(); i++){
				avgBorder += (revan.getPixelValue(0, i) + revan.getPixelValue(revan.getWidth()-1, i)) / (2* revan.getHeight());
			}
			double factor = ((stats[0] - Math.abs(stats[0] - avgBorder)) / avgBorder) - 1;
			System.out.println("Factor :"  + factor);
			for (int i = 0; i < scale.length; i++){
				scale[i] =  (1 + (factor * Math.cos(i * 2 * Math.PI / revan.getWidth())));
			}
			for (int j = 0;  j < imp.getHeight(); j++){
				for (int i = 0 ; i< revan.getWidth(); i++){
					double value = revan.getPixelValue(i, j) * scale[i];
					revan.putPixelValue(i, j, value);
				}
			}
		}
		return revan;
	}	


	/**
	 * low pass filters the given array. All indices with a distance  greater or equal to cutOffIndex from the highest frequency are set to 0;
	 * @param array the real double array
	 * @param cutOffIndex the index to start cutting
	 * @return the low pass filtered array.
	 */
	public static double [] lowPassFilterRealDoubleArray (double [] array, int cutOffIndex){
		int newDimension = FFTUtil.getNextPowerOfTwo(array.length);
		// Write the data to JTransforms complex format
		double [] fftData = new double[newDimension*2];
		for (int i = 0 ; i < array.length; i++){
			fftData[i*2] = array[i];
		}
		// FFT
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		fft.complexForward(fftData);
		// low pass data
		for (int i = 0 ; i < cutOffIndex; i++){
			int center = newDimension;
			fftData[center + (i*2)] = 0;
			fftData[center + (i*2) + 1] = 0;
			fftData[center - (i*2)] = 0;
			fftData[center - (i*2) + 1] = 0;
		}
		// iFFT
		fft.complexInverse(fftData, true);
		// rewrite as real array.
		double [] revan = new double[newDimension];
		for (int i = 0 ; i < array.length; i++){
			revan[i] = fftData[2*i];
		}
		return revan;
	}
	


	/**
	 * Applies a ramp filter to the an ImageProcessor
	 * @param imp the ImageProcessor to be filtered
	 * @param ramp the ramp
	 * @return the filtered ImageProcessor
	 */
	public static Grid2D applyRampFilter_ECC(Grid2D imp, RampFilter ramp){
		int originalWidth;
		int originalHeight;
		int maxN;
		boolean error = false;
		originalWidth = imp.getWidth();
		originalHeight = imp.getHeight();
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		// Write the data to JTransforms format
		double [] fftData = new double[newDimension*2];
		double [] filter = ramp.getRampFilter1D(newDimension);
		if (DoubleArrayUtil.isNaN(filter)){
			System.out.println("NaN found in Filter!");
		}
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		Grid2D revan = new Grid2D(originalWidth, originalHeight);
		revan.setOrigin(imp.getOrigin());
		revan.setSpacing(imp.getSpacing());
		for (int j = 0;  j < imp.getHeight(); j++){
			for (int i = 0 ; i < newDimension *2; i++){
				fftData[i] = 0;
			}
			if (DoubleArrayUtil.isNaN(fftData)){
				System.out.println("Never happens!");
			}
			for (int i = 0 ; i < imp.getWidth(); i++){
				fftData[i*2] = imp.getPixelValue(i, j);
			}
			if (DoubleArrayUtil.isNaN(fftData)){
				System.out.println("NaN found in input Data! line " + j);
				for (int i = 0 ; i < imp.getWidth(); i++){
					if (Double.isNaN(fftData[i*2])){
						System.out.println("index " + i);
					}
				}
				error = true;
			}
			// Perform forward transform
			fft.complexForward(fftData);
			if (DoubleArrayUtil.isNaN(fftData)){
				System.out.println("NaN found after FFT!");
			}
			// Filter
			fftData = FFTUtil.multiplyAbsolute(fftData, filter);
			if (DoubleArrayUtil.isNaN(fftData)){
				System.out.println("NaN found after filter application!");
			}
			// Perform backward transform
			fft.complexInverse(fftData, true);
			if (DoubleArrayUtil.isNaN(fftData)){
				System.out.println("NaN found after iFFT!");
			}
			for (int i = 0 ; i< revan.getWidth(); i++){
				double value = fftData[i*2];
				revan.putPixelValue(i, j, value);
			}
		}
		if (error) {
			//ImagePlus gi = ImageUtil.wrapGrid3D(imp, "");
			//gi.setTitle("Errors");
			//gi.show();
		}
		return revan;
	}

	/**
	 * Applies a ramp filter to the an ImageProcessor
	 * @param imp the ImageProcessor to be filtered
	 * @param ramp the ramp
	 * @return the filtered ImageProcessor
	 */
	public static Grid2D applyRampFilter(Grid2D imp, RampFilter ramp){
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = imp.getWidth();
		originalHeight = imp.getHeight();
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN); // Satisfy Nyquist?
		// Write the data to JTransforms format
		double [] fftData = new double[newDimension*2];	// Zero padding?
		double [] filter = ramp.getRampFilter1D(newDimension);		
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		Grid2D revan = new Grid2D(originalWidth, originalHeight);
		revan.setOrigin(imp.getOrigin());
		revan.setSpacing(imp.getSpacing());
		for (int j = 0;  j < imp.getHeight(); j++){
			for (int i = 0 ; i < newDimension *2; i++){
				fftData[i] = 0;
			}
			for (int i = 0 ; i < imp.getWidth(); i++){
				fftData[i*2] = imp.getPixelValue(i, j);
			}
			// Perform forward transform
			fft.complexForward(fftData);
			// Filter
			fftData = FFTUtil.multiplyAbsolute(fftData, filter);
			// Perform backward transform
			fft.complexInverse(fftData, true);
			for (int i = 0 ; i< revan.getWidth(); i++){
				double value = fftData[i*2];
				revan.putPixelValue(i, j, value);
			}
		}
		return revan;
	}

	/**
	 * Applies a ramp filter to the a detector row
	 * @param detectorRow the row
	 * @param ramp the ramp
	 * @return the filtered row
	 */
	public static double [] applyRampFilter(double [] detectorRow, RampFilter ramp){
		int originalWidth;
		int maxN;
		originalWidth = detectorRow.length;
		maxN = originalWidth;
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		
		// Write the data to JTransforms format
		double [] fftData = new double[newDimension*2];
		double [] filter = ramp.getRampFilter1D(newDimension);		
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		double [] revan = new double[originalWidth];
		for (int i = 0 ; i < newDimension *2; i++){
			fftData[i] = 0;
		}
		for (int i = 0 ; i < detectorRow.length; i++){
			fftData[i*2] = detectorRow[i];
		}
		// Perform forward transform
		fft.complexForward(fftData);
		// Filter
		fftData = FFTUtil.multiplyAbsolute(fftData, filter);
		// Perform backward transform
		fft.complexInverse(fftData, true);
		for (int i = 0 ; i< revan.length; i++){
			double value = fftData[i*2];
			revan[i] = value;
		}
		return revan;
	}

	/**
	 * Applies a 2D filter to the an ImageProcessor
	 * @param imp the ImageProcessor to be filtered
	 * @param filter the filter
	 * @return the filtered ImageProcessor
	 */
	public static Grid2D apply2DFilter(Grid2D imp, Grid2D filter){
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = imp.getWidth();
		originalHeight = imp.getHeight();
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		// Write the data to JTransforms format
		double [][] fftData = new double[newDimension][newDimension*2];
		DoubleFFT_2D fft = new DoubleFFT_2D(newDimension, newDimension);
		Grid2D revan = new Grid2D(originalWidth, originalHeight);
		revan.setOrigin(imp.getOrigin());
		revan.setSpacing(imp.getSpacing());
		for (int j = 0;  j < imp.getHeight(); j++){
			for (int i = 0 ; i < imp.getWidth(); i++){
				fftData[i][j*2] = imp.getPixelValue(i, j);
			}
		}
		// Perform forward transform
		fft.complexForward(fftData);
		// Filter
		for (int j = 0;  j < filter.getHeight(); j++){
			for (int i = 0 ; i < filter.getWidth(); i++){
				fftData[i][j*2] *= filter.getPixelValue(i, j);
				fftData[i][(j*2)+1] *= filter.getPixelValue(i, j);
			}
		}
		// Perform backward transform
		fft.complexInverse(fftData, true);
		for (int j = 0;  j < imp.getHeight(); j++){
			for (int i = 0 ; i< revan.getWidth(); i++){
				double value = fftData[i][j*2];
				revan.putPixelValue(i, j, value);
			}
		}
		return revan;
	}


	/**
	 * Divides two complex values
	 * @param realOne real part one
	 * @param imagOne imaginary part one
	 * @param realTwo real part two
	 * @param imagTwo imaginary part two
	 * @return an array of two values: first entry is real, second imaginary
	 */
	public static double [] divideComplex(double realOne, double imagOne, double realTwo, double imagTwo){
		double [] revan = new double [2];
		double denominator = Math.pow(realTwo, 2) + Math.pow(imagTwo, 2);
		revan [0] = ((realOne * realTwo) + (imagOne * imagTwo)) / denominator;
		revan [1] = ((imagOne * realTwo) - (realOne * imagTwo)) / denominator;
		return revan;
	}

	/**
	 * Divides two arrays of complex numbers. The absolute value of the second array is used to divide the complex first array.
	 * @param one the first array
	 * @param two the second array
	 * @return the array of null if the lengths of the arrays don't match
	 */
	public static double [] divideAbsolute(double [] one, double [] two){
		double [] revan = null;
		if (one.length==two.length) {
			revan = new double [one.length];
			for (int i = 0; i < one.length/2; i++){
				revan[2*i] = (one[2*i] / abs(i, two));
				revan[(2*i)+1] = (one[(2*i)+1] / abs(i, two));
			}
		}
		return revan;
	}

	/**
	 * Multiplies two complex values
	 * @param realOne real part one
	 * @param imagOne imaginary part one
	 * @param realTwo real part two
	 * @param imagTwo imaginary part two
	 * @return an array of two values: first entry is real, second imaginary
	 */
	public static double [] multiplyComplex(double realOne, double imagOne, double realTwo, double imagTwo){
		double [] revan = new double [2];
		revan [0] = (realOne * realTwo) - (imagOne * imagTwo);
		revan [1] = (imagOne * realTwo) + (realOne * imagTwo);
		return revan;
	}

	/**
	 * Multiplies two arrays of complex numbers pairwise
	 * @param one the first array
	 * @param two the second array
	 * @return the array of null if the lengths of the arrays don't match
	 */
	public static double [] multiplyComplex(double [] one, double [] two){
		double [] revan = null;
		if (one.length==two.length) {
			revan = new double [one.length];
			for (int i = 0; i < one.length/2; i++){
				revan[2*i] = (one[2*i] * two[2*i]) - (one[(2*i)+1] * two[(2*i)+1]);
				revan[(2*i)+1] = (one[(2*i)+1] * two[2*i]) + (one[2*i] * two[(2*i)+1]);
			}
		}
		return revan;
	}

	/**
	 * Multiplies two arrays of complex numbers. The absolute value of the second array is multiplied to the complex first array.
	 * @param one the first array
	 * @param two the second array
	 * @return the array of null if the lengths of the arrays don't match
	 */
	public static double [] multiplyAbsolute(double [] one, double [] two){
		double [] revan = null;
		if (one.length==two.length) {
			revan = new double [one.length];
			for (int i = 0; i < one.length/2; i++){
				revan[2*i] = (one[2*i] * abs(i, two));
				revan[(2*i)+1] = (one[(2*i)+1] * abs(i, two));
			}
		}
		return revan;
	}
	/**
	 * shift zero frequency to center, or vice verse, 1D.
	 * @param data the double data to be shifted
	 * @param bComplex true: complex; false: real
	 * @param bSign true: fftshift; false: ifftshift
	 * @return the fft shifted array
	 */

	public static double []  fftshift(double [] data, boolean bComplex, boolean bSign)
	{
		double [] revan = new double [data.length];

		int step = 1;
		if (bComplex) step = 2;
		int len = data.length/step;
		int p = 0;
		if(bSign) 
			p = (int) Math.ceil(len/2.0);
		else
			p = (int) Math.floor(len/2.0);

		int i=0;
		if (step==1){
			for (i=p;i<len;i++){
				revan[i-p] = data[i];
			}
			for (i=0;i<p;i++){
				revan[i+len-p] = data[i];
			}
		}
		else{
			for (i=p;i<len;i++){
				revan[(i-p)*2] = data[i*2];
				revan[(i-p)*2+1] = data[i*2+1];
			}
			for (i=0;i<p;i++){
				revan[(i+len-p)*2] = data[i*2];
				revan[(i+len-p)*2+1] = data[i*2+1];
			}			
		}			
		return revan;
	}
	
	/**
	 * shift zero frequency to center, or vice verse, 1D.
	 * @param data the float data to be shifted
	 * @param bComplex true: complex; false: real
	 * @param bSign true: fftshift; false: ifftshift
	 * @return the fft shifted array
	 */

	public static float []  fftshift(float [] data, boolean bComplex, boolean bSign)
	{
		float [] revan = new float [data.length];

		int step = 1;
		if (bComplex) step = 2;
		int len = data.length/step;
		int p = 0;
		if(bSign) 
			p = (int) Math.ceil(len/2.0);
		else
			p = (int) Math.floor(len/2.0);

		int i=0;
		if (step==1){
			for (i=p;i<len;i++){
				revan[i-p] = data[i];
			}
			for (i=0;i<p;i++){
				revan[i+len-p] = data[i];
			}
		}
		else{
			for (i=p;i<len;i++){
				revan[(i-p)*2] = data[i*2];
				revan[(i-p)*2+1] = data[i*2+1];
			}
			for (i=0;i<p;i++){
				revan[(i+len-p)*2] = data[i*2];
				revan[(i+len-p)*2+1] = data[i*2+1];
			}			
		}			
		return revan;
	}
	
	/**
	 * shift zero frequency to center, or vice verse, 2D.
	 * @param data the double data to be shifted
	 * @param bComplex true: complex; false: real
	 * @param bSign true: fftshift; false: ifftshift
	 * @return the fft shifted array
	 */

	public static double[][]  fftshift(double [][] data, boolean bComplex, boolean bSign)
	{
		int step = 1;
		if (bComplex) step = 2;
		int height = data.length;
	
		int width = data[0].length/step;
		
		double [][] revan = new double [data.length][data[0].length];

		int pH = 0;
		int pW = 0;
		if(bSign) {
			pH = (int) Math.ceil(height/2.0);
			pW = (int) Math.ceil(width/2.0);
		}
		else{
			pH = (int) Math.floor(height/2.0);
			pW = (int) Math.floor(width/2.0);
		}
		int i=0;
		int j=0;
		if (step==1){
			for(j=pH;j<height;j++){
				for (i=pW;i<width;i++){				
					revan[j-pH][i-pW] = data[j][i];
				}
				for (i=0;i<pW;i++){
					revan[j-pH][i+width-pW] = data[j][i];
				}				
			}
			for(j=0;j<pH;j++){
				for (i=pW;i<width;i++){				
					revan[j+height-pH][i-pW] = data[j][i];
				}
				for (i=0;i<pW;i++){
					revan[j+height-pH][i+width-pW] = data[j][i];
				}				
			}
		}
		else{
			for(j=pH;j<height;j++){
				for (i=pW;i<width;i++){				
					revan[j-pH][(i-pW)*2] = data[j][i*2];
					revan[j-pH][(i-pW)*2+1] = data[j][i*2+1];
				}
				for (i=0;i<pW;i++){
					revan[j-pH][(i+width-pW)*2] = data[j][i*2];
					revan[j-pH][(i+width-pW)*2+1] = data[j][i*2+1];
				}				
			}
			for(j=0;j<pH;j++){
				for (i=pW;i<width;i++){				
					revan[j+height-pH][(i-pW)*2] = data[j][i*2];
					revan[j+height-pH][(i-pW)*2+1] = data[j][i*2+1];					
				}
				for (i=0;i<pW;i++){
					revan[j+height-pH][(i+width-pW)*2] = data[j][i*2];
					revan[j+height-pH][(i+width-pW)*2+1] = data[j][i*2+1];
				}				
			}		
		}			
		return revan;
	}
	
	/**
	 * shift zero frequency to center, or vice verse, 2D.
	 * @param data the double data to be shifted
	 * @param bComplex true: complex; false: real
	 * @param bSign true: fftshift; false: ifftshift
	 * @return the fft shifted array
	 */

	public static float[][]  fftshift(float [][] data, boolean bComplex, boolean bSign)
	{
		int step = 1;
		if (bComplex) step = 2;
		int height = data.length;
	
		int width = data[0].length/step;
		
		float [][] revan = new float [data.length][data[0].length];

		int pH = 0;
		int pW = 0;
		if(bSign) {
			pH = (int) Math.ceil(height/2.0);
			pW = (int) Math.ceil(width/2.0);
		}
		else{
			pH = (int) Math.floor(height/2.0);
			pW = (int) Math.floor(width/2.0);
		}
		int i=0;
		int j=0;
		if (step==1){
			for(j=pH;j<height;j++){
				for (i=pW;i<width;i++){				
					revan[j-pH][i-pW] = data[j][i];
				}
				for (i=0;i<pW;i++){
					revan[j-pH][i+width-pW] = data[j][i];
				}				
			}
			for(j=0;j<pH;j++){
				for (i=pW;i<width;i++){				
					revan[j+height-pH][i-pW] = data[j][i];
				}
				for (i=0;i<pW;i++){
					revan[j+height-pH][i+width-pW] = data[j][i];
				}				
			}
		}
		else{
			for(j=pH;j<height;j++){
				for (i=pW;i<width;i++){				
					revan[j-pH][(i-pW)*2] = data[j][i*2];
					revan[j-pH][(i-pW)*2+1] = data[j][i*2+1];
				}
				for (i=0;i<pW;i++){
					revan[j-pH][(i+width-pW)*2] = data[j][i*2];
					revan[j-pH][(i+width-pW)*2+1] = data[j][i*2+1];
				}				
			}
			for(j=0;j<pH;j++){
				for (i=pW;i<width;i++){				
					revan[j+height-pH][(i-pW)*2] = data[j][i*2];
					revan[j+height-pH][(i-pW)*2+1] = data[j][i*2+1];					
				}
				for (i=0;i<pW;i++){
					revan[j+height-pH][(i+width-pW)*2] = data[j][i*2];
					revan[j+height-pH][(i+width-pW)*2+1] = data[j][i*2+1];
				}				
			}		
		}			
		return revan;
	}
	/**
	 * Get Lowpass and HighPass Images for "real" image using "real" filters
	 * @param data image in real
	 * @param LowPassFilter Low pass filter in real
	 * @param HighPassFilter High pass filter in real
	 * @param LowPassImage Low pass image in real
	 * @param HighPassImage High Pass image in real
	 */
	public static void  GetLowandHighPassImage(float [][] data, 
			float [][] LowPassFilter, 
			float [][] HighPassFilter, 
			float [][] LowPassImage, 
			float [][] HighPassImage)
	{
		int nHeight = data.length;
		int nWidth = data[0].length;
		double [][] temp1 = new double[nHeight][nWidth*2]; // for high pass 
		double [][] temp2 = new double[nHeight][nWidth*2]; // for low pass
		
		for (int i=0;i<nHeight;i++){
			for(int j=0;j<nWidth;j++){
				temp1[i][2*j] = data[i][j];
			}
		}
		//FloatFFT_2D fft2 = new FloatFFT_2D(nHeight,nWidth);
		DoubleFFT_2D fft2 = new DoubleFFT_2D(nHeight,nWidth);
		fft2.complexForward(temp1);
		
		for (int i=0;i<nHeight;i++){
			for(int j=0;j<nWidth;j++){
				temp2[i][2*j] = temp1[i][2*j] * LowPassFilter[i][j];
				temp1[i][2*j] = temp1[i][2*j] * HighPassFilter[i][j];
				temp2[i][2*j+1] = temp1[i][2*j+1] * LowPassFilter[i][j];	
				temp1[i][2*j+1] = temp1[i][2*j+1] * HighPassFilter[i][j];
			}
		}
	
		temp1 = fftshift(temp1,true,false);
		fft2.complexInverse(temp1,true);
		fft2.complexInverse(temp2,true);
		for (int i=0;i<nHeight;i++){
			for(int j=0;j<nWidth;j++){
				LowPassImage[i][j] = (float)temp2[i][2*j];// Math.sqrt(temp2[i][2*j]*temp2[i][2*j]+temp2[i][2*j+1]*temp2[i][2*j+1]);
				HighPassImage[i][j] = (float)temp1[i][2*j];// Math.sqrt(temp1[i][2*j]*temp1[i][2*j]+temp1[i][2*j+1]*temp1[i][2*j+1]);
			}
		}
		
	}

	/**
	 * Returns the 2D power spectrum of a given image processor.
	 * 
	 * @param imp the ImageProcessor
	 * @return the power spectrum as ImageProcessor 
	 */
	public static Grid2D getPowerSpectrum(Grid2D imp){
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = imp.getWidth();
		originalHeight = imp.getHeight();
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		// Write the data to JTransforms format
		double [][] fftData = new double[newDimension][newDimension*2];
		DoubleFFT_2D fft = new DoubleFFT_2D(newDimension, newDimension);
		for (int i = 0 ; i< imp.getWidth(); i++){
			for (int j = 0;  j< imp.getHeight(); j++){
				fftData[i][j*2] = imp.getPixelValue(i, j);
			}
		}
		// Perform forward transform
		fft.complexForward(fftData);
		// Filter
		Grid2D revan = new Grid2D(newDimension, newDimension);
		for (int i = 0 ; i< revan.getWidth(); i++){
			for (int j = 0;  j< revan.getHeight(); j++){
				double value = FFTUtil.abs(i, j, fftData);
				revan.putPixelValue(i, j, value);
			}
		}
		return revan;
	}

	/**
	 * Returns true if the number is a power of two
	 * 
	 * @param value the input number.
	 */
	public static boolean isPowerOfTwo(int value){
		return Integer.bitCount(value)==1;
	}
	
	/**
	 * Returns the NEXT power of 2 given a certain integer value 
	 * For power of twos it also returns the next power of 2, e.g. for 512 -> 1024
	 * 
	 * Code was partially taken from ij.plugin.FFT.java::pad().
	 * Thanks for the inspiration!
	 * 
	 * @param value the input number.
	 * @return the next power of two.
	 */
	public static int getNextPowerOfTwo(int value){
		if (isPowerOfTwo(value)) 
			return value*2;
		else
		{
			int i = 2;
			while (i <= value) {
				i *= 2;
			}
			return i*2;
		}
	}

	/**
	 * Estimates the applied filter given an input and an output image.
	 * 
	 * @param before the input image
	 * @param after the output image
	 * @param threshold the maximal value which may appear in the estimate. (to avoid outliers.)
	 * @return the estimate of the applied filter
	 * @throws Exception may occur.
	 */
	public static Grid2D estimateFilter2D(Grid2D before, Grid2D after, double threshold) throws Exception{
		if ((before.getWidth() != after.getWidth()|| (after.getHeight() != before.getHeight()))){
			throw new Exception ("Image dimensions do not match!");
		}
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = before.getWidth();
		originalHeight = before.getHeight();
		// Padding required? Pad if necessary
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		Grid2D filterEstimate = new Grid2D(newDimension, newDimension);
		double [][] fftBefore = new double [newDimension][newDimension*2];
		double [][] fftAfter = new double [newDimension][newDimension*2];
		DoubleFFT_2D fft = new DoubleFFT_2D(newDimension, newDimension);
		double outlier = 0;
		for (int j = 0;  j < before.getHeight(); j++){ // for all image rows
			// Convert to JTransforms format
			for (int i = 0 ; i < newDimension; i++){
				fftBefore[i][j*2] = before.getPixelValue(i, j);
				fftBefore[i][(j*2)+1] = 0;
				fftAfter[i][j*2] = after.getPixelValue(i, j);
				fftAfter[i][(j*2)+1] = 0;
			}
		}
		// Perform the FFTs
		fft.complexForward(fftBefore);
		fft.complexForward(fftAfter);
		// estimate the filter
		for (int j = 0;  j < newDimension; j++){ // for all image rows
			for (int i = 0 ; i < newDimension; i++){
				double beforeValue = abs(i, j, fftBefore);
				double add = 0;
				if (beforeValue != 0) {
					add = abs(0, divideComplex(fftAfter[i][2*j], fftAfter[i][(2*j)+1], fftBefore[i][2*j], fftBefore[i][(2*j)+1]));
					if (add < threshold) {
						filterEstimate.putPixelValue(i, j, add);
					} else {
						outlier++;
					}
				} else {
					outlier++;
				}
			}
		}
		//System.out.println("Outliers per frame: " + outlier);
		return filterEstimate;
	}


	/**
	 * Estimates the applied ramp filter given an input and an output image.
	 * 
	 * @param after the output image
	 * @param before the input image
	 * @return the estimate of the applied filter
	 * @throws Exception may occur.
	 */
	public static double [] estimateFilter(Grid2D after, Grid2D before, double threshold, boolean meanSquare) throws Exception{
		if ((before.getWidth() != after.getWidth()|| (after.getHeight() != before.getHeight()))){
			throw new Exception ("Image dimensions do not match!");
		}
		int originalWidth;
		int originalHeight;
		int maxN;
		originalWidth = before.getWidth();
		originalHeight = before.getHeight();
		// Padding required? Pad if necessary
		maxN = Math.max(originalWidth, originalHeight);
		int newDimension = FFTUtil.getNextPowerOfTwo(maxN);
		double [] filterEstimate = new double [newDimension];
		double [] fftBefore = new double [newDimension*2];
		double [] fftAfter = new double [newDimension*2];
		DoubleFFT_1D fft = new DoubleFFT_1D(newDimension);
		double outlier = 0;
		for (int j = 0;  j < before.getHeight(); j++){ // for all image rows
			// Perform the fft transform on both images
			for (int i = 0 ; i < newDimension; i++){
				fftBefore[i*2] = before.getPixelValue(i, j);
				fftBefore[(i*2)+1] = 0;
				fftAfter[i*2] = after.getPixelValue(i, j);
				fftAfter[(i*2)+1] = 0;
			}
			fft.complexForward(fftBefore);
			fft.complexForward(fftAfter);
			// estimate the filter
			if (meanSquare) {
				double [] input = new double [newDimension];
				double [] output = new double [newDimension];
				for (int i = 0 ; i < newDimension; i++){
					double in = abs(i,fftBefore);
					double out = abs(i, fftAfter);
					// Just consider observations which are not 0;
					if ((Math.abs(in) > 0.01)){
						// Least Square estimate
						input[i] += in * in;
						output[i] += in * out;
					}
				}
				for (int i = 0 ; i < newDimension; i++){
					filterEstimate[i] = output[i] / input[i];
					if ((filterEstimate[i] > threshold)||(Double.isNaN(filterEstimate[i]))) filterEstimate[i] = -1;
				}
			} else {
				for (int i = 0 ; i < newDimension; i++){
					double beforeValue = abs(i, fftBefore);
					double add = 0;
					if (beforeValue != 0) {
						add = abs(0, divideComplex(fftAfter[2*i], fftAfter[(2*i)+1], fftBefore[2*i], fftBefore[(2*i)+1]));
						if (add < threshold) {
							filterEstimate[i] += add / before.getHeight();
						} else {
							outlier++;
						}
					} else {
						outlier++;
					}
				}	
			}
		}
		outlier /= before.getHeight(); // Outliers per row
		if (!meanSquare) System.out.println("Outliers per row: " + outlier);
		return filterEstimate;
	}

	/**
	 * Computes the absolute value of the complex number at position pos in the array
	 * @param pos the position
	 * @param array the array which contains the values
	 * @return the absolute value
	 */
	public static double abs (int pos, double[] array){
		return Math.sqrt(Math.pow(array[pos *2], 2) + Math.pow(array[(2*pos)+1], 2));
	}

	/**
	 * Computes the absolute values of the complex number at posx, posy in the 2D array array.
	 * 
	 * @param posx x position
	 * @param posy y position
	 * @param array the array
	 * @return the absolute value of the complex number
	 */
	public static double abs (int posx, int posy, double[][] array){
		return Math.sqrt(Math.pow(array[posx][posy *2], 2) + Math.pow(array[posx][(2*posy)+1], 2));
	}

	/**
	 * Prints the absolute values in the given array
	 * @param array the array of complex values
	 */
	public static void printAbsolute(double [] array){
		System.out.println("Begin Absolute Out");
		for (int i = 0; i < array.length / 2; i++){
			System.out.println(FFTUtil.abs(i, array));
		}
		System.out.println("End Absolute Out");
	}

	/**
	 * Prints the complex values in the given array.
	 * @param array the array
	 */
	public static void printComplex(double [] array){
		System.out.println("Begin Compex Out");
		for (int i = 0; i < array.length / 2; i++){
			System.out.println(array[i*2] + " " + array[(i*2)+1]);
		}
		System.out.println("End Compex Out");
	}
	


}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/