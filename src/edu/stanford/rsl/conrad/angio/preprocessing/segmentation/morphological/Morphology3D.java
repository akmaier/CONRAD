/*
 * Copyright (C) 2010-2018 Maximilian Dankbar
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological;

import java.util.Arrays;

import edu.stanford.rsl.conrad.angio.preprocessing.segmentation.morphological.tools.StructuringElement3D;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;

/**
 * Performs morphological operators in 3D.
 * @author Maximilian Dankbar
 *
 */
public class Morphology3D {
	/**
	 * Percentile filter. Sets each pixel equal to the value of the pth
	 * percentile of its neighborhood.  Neighborhood is defined using a 
	 * structuring element s.
	 * <p>
	 * This is the core functionality of most of the morphological 
	 * operations, which at their core are rank-order filters.
	 *
	 @param im   An ImagePlus object containing the image to be operated on.
	 *
	 @param s    StructuringElement containing the shape of the neighborhood.
	 *
	 @param perc Percentile to be used.
	 *
	 @param symmetric Determines whether the boundary conditions are symmetric, or padded with background.
	 *
	 @return An ImagePlus containing the filtered image.
	 *
	 */
	public static ImagePlus percentileFilter(ImagePlus im, StructuringElement3D s,
			double perc, boolean symmetric) {	
		ImagePlus out;
		ImageStack stackOut = new ImageStack(im.getWidth(), im.getHeight());
		for(int z = 0; z < im.getStackSize(); ++z){
			stackOut.addSlice(im.getStack().getProcessor(z + 1).createProcessor(im.getWidth(), im.getHeight()));
		}
		out = new ImagePlus("percentile output", stackOut);

		int width = out.getWidth();
		int height = out.getHeight();

		// Send image to structuring element for speed and set symmetry.
		s.setObject(im, symmetric);
		for(int z = 0; z < out.getStackSize(); ++z){
			ImageProcessor op;
			op = out.getStack().getProcessor(z + 1);
			
			for(int x=0; x<width; x++) {
				for (int y=0; y<height; y++) {
					op.set(x,y, (int)Math.round(
						percentile( 
						s.window(x,y,z), perc ) ) );
				}
			}
		}
		return out;
	}
	
	/*
	 * Calculates the value of the perc-th percentile of values[],
	 * using the calculation technique from the Engineering
	 * Statistics Handbook published by NIST, section 7.2.5.2.
	 *
	 */
	private static double percentile(int[] values, double perc) {
		Arrays.sort(values);

		// n is the rank of the percentile value in the sorted array.
		double n = (perc/100.0)*(values.length - 1) + 1.0;

		if (Math.round(n)==1) {
			return values[0];
		} else if (Math.round(n)==values.length) {
			return values[values.length-1];
		} else {
			int k = (int)n;
			double d = n-k;
			return values[k-1] + d*(values[k] - values[k-1]);
		}
	}
	
	/**
	 * Assume symmetric bc if not supplied.
	 *
	 */
	public static ImagePlus percentileFilter(ImagePlus im, StructuringElement3D s,
			double perc) {
		return percentileFilter(im, s, perc, true);
	}
	
	/**
	 * Morphological erosion.  Compares each pixel to its neighborhood as 
	 * defined by a structuring element, s.  That pixel is then set to be 
	 * equal to the value found in its neighborhood which is closest to the
	 * background value.
	 * <p>
	 * If the background is white (255) then this corresponds to a local
	 * maximum filter.
	 * <p>
	 * If the background is black (0) then this corresponds to a local 
	 * minimum filter.
	 * <p>
	 * The structuring element defines the color of the background.
	 *
	 @param im An ImagePlus object containing the image to be operated on.
	 *
	 @param s Structuring Element defining the neighborhood.
	 *
	 @return An ImagePlus containing the filtered image.
	 *
	 */
	public static ImagePlus erode(ImagePlus im, StructuringElement3D s) {
		if (s.isBgWhite()) {
			return percentileFilter(im, s, 100.0);
		} else {
			return percentileFilter(im, s, 0.0);
		}
	}

	/**
	 * Morphological dilation.  Compares each pixel to its neighborhood as 
	 * defined by a structuring element, s.  That pixel is then set to be 
	 * equal to the value found in its neighborhood which is closest to the
	 * foreground value.
	 * <p>
	 * Note that this is the inverse operation to erosion.
	 * <p>
	 * If the background is white (255) then this corresponds to a local
	 * minimum filter.
	 * <p>
	 * If the background is black (0) then this corresponds to a local 
	 * maximum filter.
	 * <p>
	 * The structuring element defines the color of the background.
	 @param im An ImagePlus object containing the image to be operated on.
	 *
	 @param s Structuring Element defining the neighborhood.
	 *
	 @return An ImagePlus containing the filtered image.
	 *
	 */
	public static ImagePlus dilate(ImagePlus im, StructuringElement3D s) {
		if (s.isBgWhite()) {
			return percentileFilter(im, s, 0.0);
		} else {
			return percentileFilter(im, s, 100.0);
		}
	}

	/**
	 * Morphological opening.  Corresponds to an erosion followed
	 * by a dilation.
	 *
	 */
	public static ImagePlus open(ImagePlus im, StructuringElement3D s) {
		return dilate( erode( im, s ), s);
	}

	/**
	 * Morphological closing. Corresponds to a dilation followed by 
	 * an erosion.
	 *
	 */
	public static ImagePlus close(ImagePlus im, StructuringElement3D s) {
		return erode( dilate( im, s), s );
	}
}
