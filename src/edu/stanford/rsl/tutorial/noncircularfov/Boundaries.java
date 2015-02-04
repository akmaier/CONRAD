/*
 * Copyright (C) 2014 Madgalena Herbst
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

package edu.stanford.rsl.tutorial.noncircularfov;

import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.ImageUtil;

/**
 * Example that shows the boundaries and corresponding lines in the sinogram
 * 
 * @author Magdalena
 * 
 */
public class Boundaries {

	public static void main(String[] args) {

		double gammaM = 2 * 11.768288932020647 * Math.PI / 180, maxT = 500, deltaT = 1.0, focalLength = (maxT / 2.0 - 0.5)
				* deltaT / Math.tan(gammaM), maxBeta = 360 * Math.PI / 180, // +gammaM*2,
		deltaBeta = maxBeta / 360;

		// Load a saved sinogram, then it is not necessary to perform the
		// projection every time again. Remember to set the scanning parameter
		// according to the sinogram
		String path = "C:/Users/Magdalena/Pictures/Sinogram_Ellipse_0.7_0.3.zip";
		Grid2D sino = ImageUtil.wrapImagePlusSlice(IJ.openImage(path), 1, true);
		sino.setSpacing(deltaT, deltaBeta);

		new ImageJ();
		int width = sino.getWidth();
		int height = sino.getHeight();
		Grid2D bounds = new Grid2D(width, height);
		Grid2D mask = new Grid2D(width, height);

		int detSize = 250; // detector size in pixel

		// find the boundaries of the sinogram = the first non-zero pixel
		// left boundary + left boundary shifted by detectorsize
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (sino.getAtIndex(x, y) > 0) {
					bounds.setAtIndex(x, y, 1.f);
					bounds.setAtIndex(x + detSize, y, 0.5f);
					mask.setAtIndex(x + detSize, y, 1.f);
					break;
				}
			}

			// right boundary
			for (int x = width - 1; x >= 0; x--) {
				if (sino.getAtIndex(x, y) > 0) {
					bounds.setAtIndex(x, y, 1.f);
					break;
				}
			}
		}

		Correspondences c = new Correspondences(focalLength, sino);
		Grid2D boundsInv = c.findRedundancies(mask, mask);
		Grid2D point = new Grid2D(width, height);

		// this was to check the intersection points and the corresponding
		// points, values have to be changed dependent on the sinogram and the
		// detector size
		point.setAtIndex(118, 333, 1.f);
		point.setAtIndex(428, 53, 1.f);
		Grid2D pointD = c.findCorrespondences(point, point);
		NumericPointwiseOperators.addBy(bounds, boundsInv);
		sino.show("Sinogram");
		bounds.show("Boundauries and corresponding lines");
		pointD.show("Intersection points and corresponding points");

	}

}
