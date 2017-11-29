package edu.stanford.rsl.tutorial.mipda;

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
import ij.ImageJ;

/**
 * Ramp Filter for Parallel Beam Reconstruction
 * Programming exercise for module "Parallel Beam Reconstruction"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch, Bastian Bier, Ashwini Jadhav, Anna Gebhard, Mena Abdelmalek
 *
 */

public class ExercisePB {
	
	public enum RampFilterType {NONE, RAMLAK, SHEPPLOGAN};
	boolean filterShownOnce = false;
	
	RampFilterType filterType = RampFilterType.NONE; //TODO: Select one of the following values: NONE, RAMLAK, SHEPPLOGAN
	// (make this choice when you have finished the rest of the exercise)
	
	SheppLogan sheppLoganPhantom;
	
	// parameters for the phantom
	final int phantomSize = 256; // size of the phantom
	final float world_unit = 1.0f; // pixel dimension
	
	// parameters for projection
	final int projectionNumber = 180;// number of projection images
	final double angularRange = Math.PI; // projection image range 		
	final double angularStepLength = angularRange / projectionNumber; // angle in between adjacent projections
	//
	final float detectorSize = 250;	// detector size in pixel
	final float detectorSpacing = 1.0f; // size of a detector Element [mm]
	final float detectorLength = detectorSize*detectorSpacing;
	
	// parameters for filtering
	int paddedSize = 0; // determined later by phantomSize and Grid1DComplex
	
	// parameters for reconstruction
	final float recoPixelDim = world_unit; // isotropic image pixels
	final int recoSize = (int) ((int) phantomSize*world_unit/recoPixelDim);

	// outputs
	Grid2D sinogram;
	Grid1DComplex ramp;
	Grid2D filteredSinogram;	
	Grid2D recoImage;


	public static void main(String[] args) {
		
		//ImageJ ij = new ImageJ(); // optional TODO: uncomment if you want to analyze the output images
		
		ExercisePB parallel = new ExercisePB(); // Step 1: creates the Shepp-Logan phantom as example image
		parallel.get_sheppLoganPhantom().show("Original Image (Shepp-Logan)");
		
		// Step 2: acquire forward projection images with a parallel projector
		parallel.sinogram = parallel.parallelProjection();
		parallel.get_sinogram().show("The Sinogram");
		
		// Step 3: ramp filtering
		parallel.filteredSinogram = parallel.filterSinogram(); // All TODO-s here!
		parallel.get_filteredSinogram().show("Filtered Sinogram");
 
		// Step 4: reconstruct the object by backprojection of the information in the filtered sinogram
		parallel.recoImage = parallel.backprojectFilteredSinogram();
		parallel.get_recoImage().show("Reconstructed Image");
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
	 * @param first input is a line of the sinogram
	 *  
	 */
	public Grid1D rampFiltering(Grid1D projection, RampFilterType filterType){
		
		// initialize the ramp filter
		ramp = new Grid1DComplex(projection.getSize()[0]);
		
		// define the filter in the spatial domain on the fully padded size
		paddedSize = ramp.getSize()[0];

		double deltaS = projection.getSpacing()[0];
		
		// RampFilterType: RAMLAK, SHEPPLOGAN, NONE
		switch (filterType) {
		
		case RAMLAK: // RamLak filter
			ramlak(ramp, paddedSize, deltaS); // TODO: go to this method and implement it
			break;
			
		case SHEPPLOGAN: // SheppLogan filter
			sheppLogan(ramp, paddedSize, deltaS); // TODO: go to this method and implement it
			break;
			
		default: // NONE (or other)
			return projection;			
		}
		
		if (!filterShownOnce)
			ramp.show("Ramp Filter in Spatial Domain (rearranged for FFT-Shift)");
		
		// <your code> // TODO: Transform the filter into frequency domain (look for an appropriate method of Grid1DComplex)
		
		if(!filterShownOnce) {
			
			ramp.show("Ramp Filter in Frequency Domain");
			filterShownOnce = true;
		}

		Grid1DComplex projectionF = null;// TODO: Transform the input sinogram signal ...
		// <your code> // ... into the frequency domain (hint: similar to the last TODO)
		
		if (projectionF != null) {
			for(int p = 0; p < projectionF.getSize()[0]; p++){
				// <your code> // TODO: Multiply the transformed sinogram with the ramp filter (complex multiplication) 
			}	
		}

		// <your code> // TODO: transform back to get the filtered sinogram (i.e. invert the Fourier transform)
				
		// crop the image to its initial size
		Grid1D grid = new Grid1D(projection);
		if (projectionF != null) {
			grid = projectionF.getRealSubGrid(0, projection.getSize()[0]);
		}
		
		NumericPointwiseOperators.multiplyBy(grid, (float) deltaS); // integral discretization (filtering)
		
		return grid;
	}
	
	// TODO: implement the ram-lak filter in the spatial domain
	public void ramlak(Grid1DComplex filterGrid, int paddedSize, double deltaS) {

		final float constantFactor = -1.f / ((float) ( Math.PI * Math.PI * deltaS * deltaS));
		
		// <your code>  // TODO: set correct value in filterGrid for zero frequency
		
		for (int i = 1; i < paddedSize/2; ++i) { // the "positive wing" of the filter 
			
			if (false) {// TODO: condition -> only odd indices are nonzero
				// <your code> // TODO: use setAtIndex and the constant "constantFactor"
			}
		}
		
		// remark: the sorting of frequencies in the Fourier domain is 0,...,k,-k,...,1
		// to implement the "negative wing", a periodicity assumption is used and the values are added accordingly
		for (int i = paddedSize / 2; i < paddedSize; ++i) { 
			
			final int tmp = paddedSize - i; // now we go back from N/2 to 1
			if (false) {// TODO: condition -> only odd indices are nonzero
				// <your code> // TODO: use setAtIndex and the constant "constantFactor"
			}
		}
	}
	
	// TODO: implement the Shepp-Logan filter in the spatial domain
	public void sheppLogan(Grid1DComplex filterGrid, int paddedSize, double deltaS) {
		
		final float constantFactor = - 2.f / ((float) ( Math.PI * Math.PI * deltaS * deltaS));
		
		// <your code> // TODO: set correct value in filterGrid for zero frequency
		
		for (int i = 1; i < paddedSize/2; ++i){ // the "positive wing" of the filter
			// <your code> // TODO: use setAtIndex and the constant "constantFactor"
		}

		// remark: the sorting of frequencies in the Fourier domain is 0,...,k,-k,...,1
		// to implement the "negative wing", a periodicity assumption is used and the values are added accordingly
		for (int i = paddedSize / 2; i < paddedSize; ++i) {
			
			final float tmp = paddedSize - i; // now we go back from N/2 to 1
			// <your code> // TODO: use setAtIndex and the constant "constantFactor"
		}
	}

	/**
	 * The following content is read-only.
	 * Both methods
	 * 1. projectRayDriven(...) 
	 * 2. and backprojectPixelDriven(...) 
	 * are used above to 
	 * 1. generate the sinogram
	 * 2. and to reconstruct the phantom from it.
	 * You can try to understand it if you want to see how these operators can be implemented.
	 * Otherwise the exercise is finished here.
	 */
	
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
		
		final double samplingRate = 3.d; // # of samples per pixel

		// prepare output (sinogram)
		int maxThetaIndex = (int) (maxTheta / deltaTheta);
		int maxSIndex = (int) (maxS / deltaS);
		
		Grid2D sino = new Grid2D(new float[maxThetaIndex*maxSIndex], maxSIndex, maxThetaIndex); // rows: angle, columns: detector pixel
		sino.setSpacing(deltaS, deltaTheta);

		// set up image bounding box in WC
		Translation trans = new Translation(
				-(grid.getSize()[0] * grid.getSpacing()[0])/2, -(grid.getSize()[1] * grid.getSpacing()[1])/2, -1);
		Transform inverse = trans.inverse();
		//
		Box b = new Box((grid.getSize()[0] * grid.getSpacing()[0]), (grid.getSize()[1] * grid.getSpacing()[1]), 2);
		b.applyTransform(trans); // box now centered at world origin

		for(int e=0; e<maxThetaIndex; ++e){ // rotation angles
			
			// compute theta [rad] and angular functions.
			double theta = deltaTheta * e;
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);

			for (int i = 0; i < maxSIndex; ++i) { // detector cells
				
				// compute s, the distance from the detector edge in WC [mm]
				double s = deltaS * i - maxS / 2;
				
				// compute two points on the line through s and theta
				// We use PointND for Points in 3D space and SimpleVector for directions.
				PointND p1 = new PointND(s * cosTheta, s * sinTheta, .0d);
				PointND p2 = new PointND(-sinTheta + (s * cosTheta), cosTheta + (s * sinTheta), .0d);
				// set up line equation
				StraightLine line = new StraightLine(p1, p2);
				// compute intersections between bounding box and intersection line
				ArrayList<PointND> points = b.intersect(line);

				// only if we have intersections
				if (points.size() != 2){
					if(points.size() == 0) {
						line.getDirection().multiplyBy(-1.d);
						points = b.intersect(line);
					}
					if(points.size() == 0) // ray passes the image area
						continue;
				}

				PointND start = points.get(0); // [mm]
				PointND end = points.get(1);   // [mm]

				// get the normalized increment
				SimpleVector increment = new SimpleVector(end.getAbstractVector());
				increment.subtract(start.getAbstractVector());
				double distance = increment.normL2();
				increment.divideBy(distance * samplingRate);

				double sum = .0;
				
				start = inverse.transform(start); // use image coordinates
				
				// compute the integral along the line.
				for (double t = 0.0; t < distance * samplingRate; ++t) {
					
					PointND current = new PointND(start);
					current.getAbstractVector().add(increment.multipliedBy(t));

					double x = current.get(0) / grid.getSpacing()[0];
					double y = current.get(1) / grid.getSpacing()[1];

					if (grid.getSize()[0] <= x + 1 // ray outside right of image
							|| grid.getSize()[1] <= y + 1 // ray above image
							|| x < 0 // ray outside left of image
							|| y < 0) // ray below image
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
	 * Backprojection of the projections/sinogram
	 * The projections are created pixel driven.
	 * -> Rule of thumb: Always sample in the domain where you expect the output!
	 * Here, we want to reconstruct the volume, thus we sample in the reconstructed grid!
	 * 
	 * @param sino: the sinogram
	 * @param imageSizeX: x-dimension of reconstructed image
	 * @param imageSizeY: y-dimension of reconstructed image
	 * @param pxSzXMM: granularity in x
	 * @param pxSzYMM: granularity in y 	 
	 */
	
	public Grid2D backprojectPixelDriven(Grid2D sino, int imageSizeX, int imageSizeY, float pxSzXMM, float pxSzYMM) {
		
		int maxSIndex = sino.getSize()[0];
		double deltaS = sino.getSpacing()[0];
		int maxThetaIndex = sino.getSize()[1];
		double deltaTheta = sino.getSpacing()[1];

		double maxS = maxSIndex * deltaS; // detector length
		
		Grid2D grid = new Grid2D(imageSizeX, imageSizeY);
		grid.setSpacing(pxSzXMM, pxSzYMM);
		grid.setOrigin(-(grid.getSize()[0]*grid.getSpacing()[0])/2, -(grid.getSize()[1]*grid.getSpacing()[1])/2);
		
		// loop over the projection angles
		for (int i = 0; i < maxThetaIndex; i++) {
			
			// compute actual value for theta
			double theta = deltaTheta * i;
			
			// get detector direction vector
			double cosTheta = Math.cos(theta);
			double sinTheta = Math.sin(theta);
			SimpleVector dirDetector = new SimpleVector(cosTheta*grid.getSpacing()[0],sinTheta*grid.getSpacing()[1]);
			dirDetector.normalizeL2();
			
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
					if (subgrid.getSize()[0] <= s + 1 ||  s < 0)
						continue;
					
					// get interpolated value and add value to sinogram
					float val = InterpolationOperators.interpolateLinear(subgrid, s);
					grid.addAtIndex(x, y, val);
				}
			}
		}
		
		// apply correct scaling
		NumericPointwiseOperators.multiplyBy(grid, (float) deltaTheta); // integral discretization (theta)
		
		return grid;
	}
	
	/**
	* 
	* end of the exercise
	*/	
	
	public ExercisePB() {
		
		sheppLoganPhantom = new SheppLogan(phantomSize);
		sheppLoganPhantom.setSpacing(world_unit,world_unit);
	}

	public Grid2D parallelProjection(){
		
		Grid2D grid = projectRayDriven(sheppLoganPhantom, angularRange, angularStepLength, detectorLength, detectorSpacing);
		return grid;
	}
	
	public Grid2D filterSinogram(){
		
		Grid2D grid = new Grid2D(sinogram);
		
		for (int theta = 0; theta < sinogram.getSize()[1]; ++theta) {
			
			// Filter each line of the sinogram independently
			Grid1D tmp = rampFiltering(sinogram.getSubGrid(theta), filterType);
			
			for(int i = 0; i < tmp.getSize()[0]; i++) {
				grid.putPixelValue(i, theta, tmp.getAtIndex(i));
			}
		}

		return grid;
	}
	
	public Grid2D backprojectFilteredSinogram(){
			
		Grid2D reco = backprojectPixelDriven(filteredSinogram, recoSize, recoSize, recoPixelDim, recoPixelDim);
		
		return reco;
			
	}
	
	// getters for members
	// variables which are checked (DO NOT CHANGE!)
	public RampFilterType get_RampFilterType() {
		return filterType;
	}
	public SheppLogan get_sheppLoganPhantom() {
		return sheppLoganPhantom;
	}
	public int get_phantomSize() {
		return phantomSize;
	}
	public float get_world_unit() {
		return world_unit;
	}
	public int get_projectionNumber() {
		return projectionNumber;
	}
	public double get_angularRange() {
		return angularRange;
	}
	public double get_angularStepLength() {
		return angularStepLength;
	}
	public float get_detectorSize() {
		return detectorSize;
	}
	public float get_detectorSpacing() {
		return detectorSpacing;
	}
	public float get_detectorLength() {
		return detectorLength;
	}
	public int get_paddedSize() {
		return paddedSize;
	}
	public float get_recoPixelDim() {
		return recoPixelDim;
	}
	public int get_recoSize() {
		return recoSize;
	}
	public Grid2D get_sinogram() {
		return sinogram;
	}
	public Grid1DComplex get_ramp() {
		return ramp;
	}
	public Grid2D get_filteredSinogram() {
		return filteredSinogram;
	}
	public Grid2D get_recoImage() {
		return recoImage;
	}
}

