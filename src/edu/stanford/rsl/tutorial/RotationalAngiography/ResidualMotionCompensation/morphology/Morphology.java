package edu.stanford.rsl.tutorial.RotationalAngiography.ResidualMotionCompensation.morphology;

import ij.*;
import ij.process.*;
import java.lang.Math;
import java.util.Arrays;
//import org.ajdecon.morphology.StructuringElement;

/**
 * Morphology. Utility class which performs image processing operations from mathematical morphology
 * based on StructuringElement objects which define the "shape" of the operation. Images are
 * passed as ImageJ ImagePlus objects.
 *
 * For more information on mathematical morphology: http://en.wikipedia.org/wiki/Mathematical_morphology
 * For ImageJ: http://rsbweb.nih.gov/ij/ 
 *
 @author Adam DeConinck
 @version 0.1
 *
 */
public class Morphology {

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
	public static ImagePlus percentileFilter(ImagePlus im, StructuringElement s,
			double perc, boolean symmetric) {

		ImagePlus in = new ImagePlus("percentile input", im.getProcessor().convertToByte(true) );
		ImagePlus out = new ImagePlus("percentile output", in.getProcessor().createImage() );
		ImageProcessor op = out.getProcessor();

		int width = out.getWidth();
		int height = out.getHeight();

		// Send image to structuring element for speed and set symmetry.
		s.setObject(in, symmetric);
		for(int x=0; x<width; x++) {
			for (int y=0; y<height; y++) {
				op.set(x,y, (int)Math.round(
					percentile( 
					s.window(x,y), perc ) ) );
//				System.out.print(x);System.out.print(" ");System.out.print(y);System.out.println();
			}
		}
		return out;
	}

	/**
	 * Assume symmetric bc if not supplied.
	 *
	 */
	public static ImagePlus percentileFilter(ImagePlus im, StructuringElement s,
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
	public static ImagePlus erode(ImagePlus im, StructuringElement s) {
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
	public static ImagePlus dilate(ImagePlus im, StructuringElement s) {
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
	public static ImagePlus open(ImagePlus im, StructuringElement s) {
		return dilate( erode( im, s ), s);
	}

	/**
	 * Morphological closing. Corresponds to a dilation followed by 
	 * an erosion.
	 *
	 */
	public static ImagePlus close(ImagePlus im, StructuringElement s) {
		return erode( dilate( im, s), s );
	}

	/** 
	 * Morphological gradient.  Difference between the dilation and the erosion.
	 *
	 */
	public static ImagePlus gradient(ImagePlus im, StructuringElement s) {
		ImagePlus d = dilate(im,s);
		ImagePlus e = erode(im,s);

		int width=im.getWidth(); int height=im.getHeight();
		ImageProcessor dp = d.getProcessor();
		ImageProcessor ep = e.getProcessor();

		for (int x=0;x<width;x++) {
			for (int y=0; y<height; y++) {
				dp.set(x,y, (int)dp.getPixel(x,y) - (int)ep.getPixel(x,y) );
			}
		}
		return d;
	}

	/**
	 * Morphological hit-or-miss transform. Takes one image and two structuring
	 * elements. Result is the set of positions, where the first structuring element fits
	 * in the foreground of the input, and the second misses it completely.
	 
	 @param im ImagePlus object containing the image to be operated on.
	 @param c Structuring element for the foreground.
	 @param d Structuring element for the background.
	 @return ImagePlus containing the filtered image.
	 *
	 */
	public static ImagePlus hitOrMiss(ImagePlus im, 
			StructuringElement c, StructuringElement d) {

		/* Note that both fg and bg are returned from erode, so are 
		 * guaranteed to be 8-bit images with ByteProcessors. */
		ImagePlus fg = erode(im,c);
		im.getProcessor().invert();
		ImagePlus bg = erode( new ImagePlus( " ", im.getProcessor() ), d);

		int width=fg.getWidth();
		int height=fg.getHeight();
		ImageProcessor ip = fg.getProcessor();

		for (int x=0; x<width; x++) {
			for (int y=0; y<width; y++) {
				ip.set( x, y, ip.getPixel(x,y) & bg.getProcessor().getPixel(x,y) );
			}
		}
		return fg;
	}

	/**
	 * Morphological top-hat transform. Can be "white" or "black" top-hat.
	 * White corresponds to difference between image and its opening.
	 * Black corresponds to difference between closing and image.
	 *
	 */
	public static ImagePlus topHat(ImagePlus im, StructuringElement s, boolean white) {

		int width = im.getWidth();
		int height = im.getHeight();

		ImagePlus result = new ImagePlus("top hat", im.getProcessor());
		ImageConverter ic = new ImageConverter(result);
		ic.convertToGray8();
		ImageProcessor rip = result.getProcessor();

		if (white) {
			ImageProcessor o = open(im, s).getProcessor();
			
			for (int x=0; x<width; x++) {
				for (int y=0; y<height; y++) {
					rip.set(x,y, (rip.getPixel(x,y) - o.getPixel(x,y)));
				}
			}
		} else {
			ImageProcessor c = close(im, s).getProcessor();

			for (int x=0; x< width; x++) {
				for (int y=0; y<height; y++) {
					rip.set(x,y, (c.getPixel(x,y) - rip.getPixel(x,y)));
				}
			}
		}
		return result;
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
}
