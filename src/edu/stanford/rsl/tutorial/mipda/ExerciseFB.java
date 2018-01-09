package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.filters.SheppLoganKernel;
import ij.IJ;
import ij.ImageJ;

/**
 * Short Scan for Fan Beam Reconstruction
 * Programming exercise for module "Fan Beam Reconstruction"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch, Marco Boegel, Anna Gebhard, Mena Abdelmalek
 *
 */

public class ExerciseFB {
	
	public enum RampFilterType {NONE, RAMLAK, SHEPPLOGAN};
	
	//TODO: choose the ramp filter (NONE, RAMLAK, SHEPPLOGAN)
	final RampFilterType filter = RampFilterType.RAMLAK;
	//TODO: choose the sinogram data (Sinogram0.tif, Sinogram1.tif, Sinogram2.tif)
	final String filename = "Sinogram0.tif";
	
	// system parameters
	final float focalLength = 600.f; // source to detector distance in [mm]
	//
	final int detectorPixels;// number of pixels on the detector
	final float detectorSpacing = 0.5f;// spacing on the detector in [mm] per pixel
	float detectorLength;// detector length in [mm]
	//
	final int numProjs;// number of acquired projections
	float betaIncrement;// increment of rotation angle beta between two projections in [rad]
	float maxBeta;// max rotation angle beta in [rad]
	float halfFanAngle;// half fan angle in [rad]
	
	// parameters for reconstruction
	int[] recoSize = new int[]{128, 128};
	float[] spacing = new float[]{1.0f, 1.0f};	
	
	// data containers
	Grid2D sino;// original sinogram
	Grid2D sino_cosW;// sinogram + cosine filter
	Grid2D sino_cosW_parkerW;// sinogram + cosine filter + Parker weights
	Grid2D sino_cosW_parkerW_filt;// sinogram + cosine filter + Parker weights + NONE/RAMLAK/SHEPPLOGAN
	Grid2D parker;// Parker weights
	Grid2D reconstruction;// reconstructed image
	
	public static void main(String arg[]) {
		
		ImageJ ij = new ImageJ();
		
		//TODO: Look for the method initParameters() and do the first TODO-s there!
		ExerciseFB fbp = new ExerciseFB();
		
		fbp.sino.show("Sinogram");
		
		System.out.println("Half fan angle [deg]: " + fbp.get_halfFanAngle()*180.0/Math.PI);
		System.out.println("Short-scan range [deg]: " + fbp.get_maxBeta()*180/Math.PI);

		fbp.doShortScanReconstruction(); //more TODO-s here!	
	}
	
	void initParameters() {

		detectorLength = detectorPixels*detectorSpacing;

		halfFanAngle = 1.0f; //TODO: determine the half fan angle such that the complete fan covers the detector exactly (hint: trigonometry) 
		maxBeta = 1.0f; //TODO: compute the short scan minimum range using the value of halfFanAngle
		betaIncrement = maxBeta/(float) numProjs;
	}

	void doShortScanReconstruction() {
		
		//assume a flat detector (equally-spaced case)
		
		//apply cosine weights
		sino_cosW = applyCosineWeights(sino); //TODO: Go to this method and implement the missing steps!
		sino_cosW.show("After Cosine Weighting");
		
		//apply Parker redundancy weights
		sino_cosW_parkerW = applyParkerWeights(sino_cosW); //TODO: Go to this method and implement the missing steps!
		parker.show("Parker Weights");
		sino_cosW_parkerW.show("After Parker Weighting");
		
		//apply ramp filter
		sino_cosW_parkerW_filt = applyRampFiltering(sino_cosW_parkerW);
		switch(filter) {
		case RAMLAK:			
			sino_cosW_parkerW_filt.show("After RamLak Filter");
			break;
		case SHEPPLOGAN:
			sino_cosW_parkerW_filt.show("After Shepp-Logan Filter");
			break;
		case NONE: //
		default: //
		}

		//perform backprojection to reconstruct the image
		Grid2D reconstruction = backproject(sino_cosW_parkerW_filt,recoSize,spacing); //TODO: Go to this method and implement the missing steps!
		reconstruction.show("Reconstructed Image");
	}

	/**
	 * Cosine weighting for 2D sinograms
	 * @param sino the sinogram
	 * @return the cosine weighted sinogram
	 */
	public Grid2D applyCosineWeights(Grid2D sino) {
		
		Grid2D result = new Grid2D(sino);
		
		//create 1D kernel (identical for each projection)
		Grid1D cosineKernel = new Grid1D(detectorPixels);
		for(int i=0; i < detectorPixels; ++i){
			
			float t = 0.f; //TODO: compute center of the current detector pixel (zero is center of detector)
			cosineKernel.setAtIndex(i, 0.0f); //TODO (hint: use t and focalLength)
		}
		
		//apply cosine weights to each projection
		for(int p = 0; p < numProjs; p++){
			NumericPointwiseOperators.multiplyBy(result.getSubGrid(p), cosineKernel);
		}
		
		return result;
	}
	
	/**
	 * Parker redundancy weights for a 2D sinogram
	 * @param sino the sinogram
	 * @return parker weighted sinogram
	 */
	public Grid2D applyParkerWeights(Grid2D sino) {
		
		Grid2D result = new Grid2D(sino);
		parker = new Grid2D(sino.getWidth(), sino.getHeight());
		
		float delta = halfFanAngle;
		float beta, gamma;

		// iterate over the detector elements
		for (int i = 0; i < detectorPixels; ++i) {
			
			// compute gamma of the current ray (detector element & flat detector)
			gamma = (float) Math.atan((i*detectorSpacing - detectorLength/2.f + 0.5f*detectorSpacing)/focalLength);
			
			// iterate over the projection angles
			for (int b = 0; b < numProjs; ++b) {
				
				beta = b*betaIncrement;
				
				if (beta > 2.f*Math.PI) {
					beta -= 2.f*Math.PI;
				}

				// implement the conditions as described in Parker's paper ( https://dx.doi.org/10.1118/1.595078 )
				if (beta <= 2*(delta - gamma)) {

					double tmp = 0.f; //TODO
					float val = 1.f; //TODO
					
					if (Double.isNaN(val)){
						continue;
					}
					
					parker.setAtIndex(i, b, val);
				}
				else if (beta <= Math.PI - 2.f*gamma) {
					parker.setAtIndex(i, b, 1);
				}
				else if (beta <= (Math.PI + 2.f*delta) + 1e-12) {
					
					double tmp = Math.PI/4.f * (Math.PI + 2.f*delta - beta)/(delta + gamma); // (error in the original paper)
					float val = 1.f; //TODO
					
					if (Double.isNaN(val)){
						continue;
					}
					
					parker.setAtIndex(i, b, val);
				}
			}
		}
		
		// correct for scaling due to varying angle
		NumericPointwiseOperators.multiplyBy(parker, (float)(maxBeta/(Math.PI)));
		
		//apply the Parker weights to the sinogram
		NumericPointwiseOperators.multiplyBy(result, parker);
		
		return result;
	}

	/**
	 * A pixel driven backprojection algorithm
	 * Cosine, redundancy weighting and ramp filters need to be applied separately beforehand!
	 * Remark: This implementation requires a detector whose center coincides with the center of rotation.
	 * @param sino the filtered sinogram
	 * @param recoSize the dimension of the output image
	 * @param spacing the spacing of the output image
	 * @return the reconstruction
	 */
	public Grid2D backproject(Grid2D sino, int[] recoSize, float[] spacing) {
		
		Grid2D result = new Grid2D(recoSize[0],recoSize[1]); // x and y flipped for ImageJ
		result.setSpacing(spacing[0],spacing[1]);
		
		//remark: float values are used here to make a transfer to a GPU implementation easier
		for(int p = 0; p < numProjs; p++) {
			
			//first, compute the rotation angle beta and pre-compute cos(beta), sin(beta)
			final float beta = (float) (p*betaIncrement);
			final float cosBeta = (float) Math.cos(beta);
			final float sinBeta = (float) Math.sin(beta);
			
			//compute rotated source point (here: center of rotation coincides with the origin, beta==0 <=> source in top position)
			final PointND source = new PointND(-focalLength*sinBeta, focalLength*cosBeta, 0.f);
			
			//compute the world coordinate point of the left detector edge (here: center of detector coincides with the origin)
			float offset = -detectorLength/2.f;
			final PointND detBorder = new PointND(0.f, 0.f, 0.f); //TODO
			
			//compute direction vector along the detector array (towards detector center)
			SimpleVector dirDet = null; //TODO
			if (dirDet == null)
				dirDet = new SimpleVector(3);
			dirDet.normalizeL2(); // ensure length == 1
			
			final StraightLine detLine = new StraightLine(detBorder, dirDet);
			detLine.setDirection(detLine.getDirection().normalizedL2()); // bugfix (to be sure)

			//pick current projection
			Grid1D currProj = sino.getSubGrid(p);
			
			float wx, wy;
			//pixel driven BP: iterate over all output image pixels
			for(int x = 0; x < recoSize[0]; x++) {
				
				//transform the image pixel coordinates to world coordinates
				wx = getWorldCoordinate(x, recoSize[0], spacing[0]); // TODO-s here!
				
				for(int y = 0; y < recoSize[1]; y++) {
					
					//transform the image pixel coordinates to world coordinates
					wy = getWorldCoordinate(y, recoSize[1], spacing[1]); // TODO-s here! (in case you missed it above)
					
					final PointND reconstructionPointWorld = new PointND(wx, wy, 0.f);

					//intersect the projection ray with the detector
					final StraightLine projectionLine = null; //TODO
					if (projectionLine != null)
						projectionLine.setDirection(projectionLine.getDirection().normalizedL2()); // bugfix
					final PointND detPixel = null; //TODO
					
					float valueTemp;
					if(detPixel != null) {
						
						//calculate position of the projected point on the 1D detector
						float t = (float) SimpleOperators.multiplyInnerProd(detPixel.getAbstractVector(), dirDet) - offset;
						
						//check if projection of this world point hits the detector
						if(t < 0 || t > detectorLength)
							continue;

						float value = InterpolationOperators.interpolateLinear(currProj, t/detectorSpacing - 0.5f);
					
						//apply distance weighting
						float dWeight = distanceWeight(reconstructionPointWorld, beta); // TODO-s here!
						valueTemp = (float) (value/(dWeight*dWeight));
					}
					else {
						
						//projected pixel lies outside of the detector
						valueTemp = 0.f;
					}
					
					result.addAtIndex(x, recoSize[1]-y-1, valueTemp); // flip to show top-down in ImageJ 
				}
			}
		}
		
		//adjust scale, required because of short scan
		float normalizationFactor = (float) (numProjs/Math.PI);
		NumericPointwiseOperators.divideBy(result, normalizationFactor);

		return result;
	}
	
	/**
	 * Transformation from image coordinates to world coordinates (single axis)
	 * @param imCoordinate image coordinate
	 * @param dim the dimension of the image
	 * @param spacing the spacing of the image
	 * @return world coordinate
	 */
	public float getWorldCoordinate(int imCoordinate, int dim, float spacing) {

		float wCoordinate = 0.f; //TODO
		
		return wCoordinate;
	}
	
	/**
	 * Compute distance weight (refer to the course slides)
	 * @param reconstructionPointWorld reconstruction point in world coordinates
	 * @param beta rotation angle beta
	 * @return distance weight
	 */
	public float distanceWeight(PointND reconstructionPointWorld, float beta) {
		
		//Compute distance weight
		float radius = 0.f; //TODO
		float phi = 0.f; //TODO
		float U = 0.f; //TODO
		
		return U;
	}

	
	/**
	 * 
	 * end of the exercise
	 */		
	
	public ExerciseFB() {
		
		//Load and visualize the projection image data
		String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/mipda/";
		
		sino = readImageFile(imageDataLoc + filename);
		
		// transpose so that each row is a projection
		sino = sino.getGridOperator().transpose(sino); // warning "Error in transpose" can be ignored (too strict tolerance settings)
		
		detectorPixels = sino.getWidth();
		numProjs = sino.getHeight();
		
		initParameters();

		sino.setSpacing(detectorSpacing, betaIncrement);
	}

	public Grid2D readImageFile(final String filename){
		return ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
	}
	
	private Grid2D applyRampFiltering(Grid2D sino) {
		
		Grid2D result = new Grid2D(sino);
		
		switch(filter) {

		case RAMLAK:
			
			RamLakKernel ramLak = new RamLakKernel(detectorPixels, detectorSpacing);
			for(int i = 0; i < numProjs; i++) {
				ramLak.applyToGrid(result.getSubGrid(i));
			}
			
			break;

		case SHEPPLOGAN:
			
			SheppLoganKernel sheppLogan = new SheppLoganKernel(detectorPixels, detectorSpacing);
			for(int i = 0; i < numProjs; i++) {
				sheppLogan.applyToGrid(result.getSubGrid(i));
			}
			
			break;

		case NONE: //
		default: //
		}
		
		return result;
	}
	
	// getters for members
	// variables which are checked (DO NOT CHANGE!)
	public RampFilterType get_filter() {
		return filter;
	}
	//
	public String get_filename() {
		return filename;
	}
	//
	public float get_focalLength() {
		return focalLength;
	}
	//
	public int get_detectorPixels() {
		return detectorPixels;
	}
	//
	public float get_detectorSpacing() {
		return detectorSpacing;
	}
	//	
	public float get_detectorLength() {
		return detectorLength;
	}
	//
	public int get_numProjs() {
		return numProjs;
	}
	//
	public float get_betaIncrement() {
		return betaIncrement;
	}
	//
	public float get_maxBeta() {
		return maxBeta;
	}
	//
	public float get_halfFanAngle() {
		return halfFanAngle;
	}
	//
	public int[] get_recoSize() {
		return recoSize;
	}
	//
	public float[] get_spacing() {
		return spacing;
	}
	//
	public Grid2D get_sino() {
		return sino;
	}
	//
	public Grid2D get_sino_cosW() {
		return sino_cosW;
	}
	//
	public Grid2D get_sino_cosW_parkerW() {
		return sino_cosW_parkerW;
	}
	//
	public Grid2D get_sino_cosW_parkerW_filt() {
		return sino_cosW_parkerW_filt;
	}
	//
	public Grid2D get_parker() {
		return parker;
	}
	//
	public Grid2D get_reconstruction() {
		return reconstruction;
	}
}
