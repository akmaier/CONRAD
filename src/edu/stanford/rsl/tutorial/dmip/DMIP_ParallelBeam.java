package edu.stanford.rsl.tutorial.dmip;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Box;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.tutorial.phantoms.SheppLogan;
import ij.ImageJ;;

/**
 * Exercise 5 of Diagnostic Medical Image Processing (DMIP)
 * @author Bastian Bier
 *
 */

public class DMIP_ParallelBeam {
	
	
	public enum RampFilterType {NONE, RAMLAK, SHEPPLOGAN};
	
	
	/**
	 * Forward projection of the phantom onto the detector
	 * Rule of thumb: Always sample in the domain where you expect the output!
	 * Thus, we sample at the detector pixel positions and sum up the informations along one ray
	 * 
	 * @param grid the image
	 * @param maxTheta the angular range in radians
	 * @param deltaTheta the angular step size in radians
	 * @param maxS the detector size in [mm]
	 * @param deltaS the detector element size in [mm]
	 */
	
	public Grid2D projectRayDriven(Grid2D grid, double maxTheta, double deltaTheta, double maxS, double deltaS) {
		
		int maxSIndex = (int) (maxS / deltaS + 1);
		int maxThetaIndex = (int) (maxTheta / deltaTheta + 1);
		
		final double samplingRate = 3.d; // # of samples per pixel
		Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex);
		sino.setSpacing(deltaS, deltaTheta);

		// set up image bounding box in WC
		Translation trans = new Translation(
				-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2, -1);
		Transform inverse = trans.inverse();

		Box b = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
		b.applyTransform(trans);

		for(int e=0; e<maxThetaIndex; ++e){
			// compute theta [rad] and angular functions.
			double theta = deltaTheta * e;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			for (int i = 0; i < maxSIndex; ++i) {
				// compute s, the distance from the detector edge in WC [mm]
				double s = deltaS * i - maxS / 2;
				// compute two points on the line through s and theta
				// We use PointND for Points in 3D space and SimpleVector for directions.
				PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
				PointND p2 = new PointND(-sinTheta + (s * cosTheta),
						(s * sinTheta) + cosTheta, .0d);
				// set up line equation
				StraightLine line = new StraightLine(p1, p2);
				// compute intersections between bounding box and intersection line.
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (2 != points.size()){
					if(points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
					}
					if(points.size() == 0)
						continue;
				}

				PointND start = points.get(0); // [mm]
				PointND end = points.get(1);   // [mm]

				// get the normalized increment
				SimpleVector increment = new SimpleVector(
						end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				double sum = .0;
				start = inverse.transform(start);

				// compute the integral along the line.
				for (double t = 0.0; t < distance * samplingRate; ++t) {
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));

					double x = current.get(0) / grid.getSpacing()[0],
							y = current.get(1) / grid.getSpacing()[1];

					if (grid.getSize()[0] <= x + 1
							|| grid.getSize()[1] <= y + 1
							|| x < 0 || y < 0)
						continue;

					sum += InterpolationOperators.interpolateLinear(grid, x, y);
				}

				// normalize by the number of interpolation points
				sum /= samplingRate;
				// write integral value into the sinogram.
				sino.setAtIndex(i, e, (float)sum);
			}
		}
		return sino;
	}
	
	/**
	 * Sampling of projections is defined in the constructor.
	 * Backprojection of the projections/sinogram
	 * Rule of thumb: Always sample in the domain where you expect the output!
	 * Here, we want to reconstruct the volume, thus we sample in the reconstructed grid!
	 * The projections are created pixel driven!
	 * 
	 * @param sino  the sinogram
	 * @param imageSizeX
	 * @param imageSizeY
	 * @param pxSzXMM
	 * @param pxSzYMM	 
	 */
	
	public Grid2D backprojectPixelDriven(Grid2D sino, int imageSizeX, int imageSizeY, float pxSzXMM, float pxSzYMM) {
		
		int maxThetaIndex = sino.getSize()[1];
		double deltaTheta = sino.getSpacing()[1];
		int maxSIndex = sino.getSize()[0];
		double deltaS = sino.getSpacing()[0];
		double maxS = (maxSIndex-1) * deltaS;
		
		Grid2D grid = new Grid2D(imageSizeX, imageSizeY);
		grid.setSpacing(pxSzXMM, pxSzYMM);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);

		// loop over the projection angles
		for (int i = 0; i < maxThetaIndex; i++) {
			// compute actual value for theta
			double theta = deltaTheta * i;
			// precompute sine and cosines for faster computation
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);
			// get detector direction vector
			SimpleVector dirDetector = new SimpleVector(sinTheta,cosTheta);
			// loops over the image grid
			for (int x = 0; x < grid.getSize()[0]; x++) {
				for (int y = 0; y < grid.getSize()[1]; y++) {
					// compute world coordinate of current pixel
					double[] w = grid.indexToPhysical(x, y);
					// wrap into vector
					SimpleVector pixel = new SimpleVector(w[0], w[1]);
					//  project pixel onto detector
					double s = SimpleOperators.multiplyInnerProd(pixel, dirDetector);
					// compute detector element index from world coordinates
					s += maxS/2; // [mm]
					s /= deltaS; // [GU]
					// get detector grid
					Grid1D subgrid = sino.getSubGrid(i);
					// check detector bounds, continue if out of array
					if (subgrid.getSize()[0] <= s + 1
							||  s < 0)
						continue;
					// get interpolated value
					float val = InterpolationOperators.interpolateLinear(subgrid, s);
					// sum value to sinogram
					grid.addAtIndex(x, y, val);
				}

			}
		}
		// apply correct scaling
		NumericPointwiseOperators.divideBy(grid, (float) (maxThetaIndex / Math.PI));
		return grid;
	}
	
	
	/**
	 * Filtering the sinogram with a high pass filter
	 * 
	 * The ramp filters are defined in the spatial domain but 
	 * they are applied in the frequency domain.
	 * Remember: a convolution in the spatial domain corresponds to a 
	 * multiplication in the frequency domain.
	 * 
	 * Both, the sinogram and the ramp filter are transformed into
	 * the frequency domain and multiplied there.
	 * 
	 * @param sinogram  a line of the sinogram
	 *  
	 */
	public Grid1D rampFiltering(Grid1D sinogram, RampFilterType filter){
		
		double deltaS = 1;
		
		// Initialize the ramp filter
		// Define the filter in the spatial domain on the full padded size!
		Grid1DComplex ramp = new Grid1DComplex(sinogram.getSize()[0]);
		
		int paddedSize = ramp.getSize()[0];
		
		if(filter == RampFilterType.RAMLAK)
		{
			// TODO: implement the ram-lak filter in the spatial domain 
			
		}
		else if(filter == RampFilterType.SHEPPLOGAN)
		{
			// TODO: implement the Shepp-Logan filter in the spatial domain
			
		}
		else
		{
			// if no filtering is used
			return sinogram;
		}
		
		// TODO: Transform ramp filter into frequency domain
		
		
		Grid1DComplex sinogramF = new Grid1DComplex(sinogram,true);
		// TODO: Transform the input sinogram signal into the frequency domain
		
		
		// TODO: Multiply the ramp filter with the transformed sinogram
		
		
		// TODO: Backtransformation

		
		// Crop the image to its initial size
		Grid1D ret = new Grid1D(sinogram);
		ret = sinogramF.getRealSubGrid(0, sinogram.getSize()[0]);
		
		return ret;
	}
	
	public static void main(String[] args)
	{
		ImageJ ij = new ImageJ();
		
		DMIP_ParallelBeam parallel = new DMIP_ParallelBeam();
		
		// 0. Parameters
		
		// size of the phantom
		int phantomSize = 128; 
		// projection image range 	
		double angularRange = Math.PI; 	
		// number of projection images	
		int projectionNumber = 180;	
		// angle in between adjacent projections
		double angularStepSize 	= angularRange / projectionNumber;
		// detector size in [mm]
		float detectorSize = 200; 
		// size of a detector Element [mm]
		float detectorSpacing = 1.0f;	
		// filterType: NONE, RAMLAK, SHEPPLOGAN
		RampFilterType filter = RampFilterType.NONE;				
		
		// 1. Create the Shepp Logan Phantom
		SheppLogan sheppLoganPhantom = new SheppLogan(phantomSize);
		sheppLoganPhantom.show();
		
		// 2. Acquire forward projection images with a parallel projector
		Grid2D sinogram = parallel.projectRayDriven(sheppLoganPhantom, angularRange, angularStepSize, detectorSize, detectorSpacing);
		sinogram.show("The Sinogram");
		Grid2D filteredSinogram = new Grid2D(sinogram);
		
		// 4. Ramp Filtering
		for (int theta = 0; theta < sinogram.getSize()[1]; ++theta) 
		{
			// Filter each line of the sinogram independently
			Grid1D tmp = parallel.rampFiltering(sinogram.getSubGrid(theta), filter);
			
			for(int i = 0; i < tmp.getSize()[0]; i++)
			{
				filteredSinogram.putPixelValue(i, theta, tmp.getAtIndex(i));
			}
		}
		
		filteredSinogram.show("Filtered Sinogram");
		
		// 5. Reconstruct the object with the information in the sinogram	
		Grid2D reco = parallel.backprojectPixelDriven(filteredSinogram, phantomSize, phantomSize, detectorSpacing, detectorSpacing);
		reco.show("Reconstruction");
	}
	
}
