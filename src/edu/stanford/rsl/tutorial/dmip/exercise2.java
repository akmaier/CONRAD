package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;

/**
 * Defect Pixel Undistortion
 * Exercise 2 of Diagnostic Medical Image Processing (DMIP)
 * @author Frank Schebesch
 *
 */

public class exercise2 {

	String filename; // do not edit, choose a number in the main method instead
	Grid2D originalImage; // read from filename when instantiating the object
	Grid2D distortedImage; // generated in the main method
	
	Grid2D xprime; // x-coordinates of point correspondences in the undistorted image
	Grid2D yprime; // y-coordinates of point correspondences in the undistorted image
	Grid2D x; // x-coordinates of point correspondences in the distorted image
	Grid2D y; // y-coordinates of point correspondences in the distorted image
	
	Grid2D undistortedImage; // use this member variable to write your output image to
	
	
	public void doImageUndistortion(){

		// check if image distortion routine has been run
		if (distortedImage == null || xprime == null || yprime == null || x == null || y == null){
			System.err.println("Error in doImageUndistortion() called before initialization of members.");
			return;
		}
			
		
		/** 1. Number of lattice points (this only works for symmetric images):
		 *  nx, ny feature points: usually provided by tracking point
		 *  correspondences in the phantom during the calibration step.
		 *  Here, we use the distorted and undistorted coordinates generated
		 *  in the course of the distortion simulation.
		 */ 
		
		// number of lattice points
		// define the number of lattice points
		int nx = //TODO
		int ny = //TODO
		
		int imageWidth = distortedImage.getWidth();
		int imageHeight = distortedImage.getHeight();
		
		// step size
		// calculate the step size of the lattice points 
		float fx = //TODO
		float fy = //TODO
		
		// Fill the distorted and undistorted lattice points 
		// with data from the given correspondences
		SimpleMatrix Xu = new SimpleMatrix(ny,nx); // matrix = number of rows (y) x number of columns (x)
		SimpleMatrix Yu = new SimpleMatrix(ny,nx);
		SimpleMatrix Xd = new SimpleMatrix(ny,nx);
		SimpleMatrix Yd = new SimpleMatrix(ny,nx);
		
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){

				//TODO: sample the distorted and undistorted grid points at the lattice points
			}
		}
		
		// Compute the distorted points: be aware of the fact, that the artificial deformation takes
		// place from the distorted coordinates to the undistorted ones!
		// The distortion was simulated by: X + distortion = Xd. So the entries in Xd map from the 
		// undistorted coordinates to the respective distorted coordinates.
		// Now the mapping of the distortion field has to be considered:
		// Distorted = Undistorted + Deformation, i.e. Xd = Xu + (Xu - Xd).
		Xd.multiplyBy(-1);
		Yd.multiplyBy(-1);
		Xd.add(Xu,Xu);
		Yd.add(Yu,Yu);
		
		
		/** 2. Polynomial of degree d
		 * Polynomial of degree d -> (d-1): extrema (e.g., d=5: 4 extrema)
		 * d=0: constant (horizontal line with y-intercept a_0 -> f(x)=a_0)
		 * d=1: oblique line with y-intercept a_0 & slope a_1 -> f(x)=a_0 + a_1 x
		 * d=2: parabola
		 * d>=2: continuous non-linear curve
		 * ...
		 * d = 10 -> NumKoeff: 66 -> but only 64 lattice points are known (nx, ny = 8)
		 */

		int degree = 5; // polynomial's degree: 2,...,10
		
		// number of Coefficients
		int numCoeff = //TODO
		
		// number of Correspondences
		int numCorresp = //TODO
		
		// Print out of the used parameters
		System.out.println("Polynom of degree: " + degree);
		System.out.println("Number of Coefficients: " + numCoeff);
		System.out.println("Number of Correspondences: " + numCorresp);
		
		
		/**
		 * 3. Create the matrix A
		 */

		SimpleMatrix A = new SimpleMatrix(numCorresp, numCoeff);
		A.zeros();
		
		// Realign the grid matrix into a vector for easier access in the next step
		SimpleVector XuVector = new SimpleVector(numCorresp);
		SimpleVector YuVector = new SimpleVector(numCorresp);
		SimpleVector XdVector = new SimpleVector(numCorresp);
		SimpleVector YdVector = new SimpleVector(numCorresp);
		
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){
				
				XuVector.setElementValue(i*ny + j, Xu.getElement(j, i));
				YuVector.setElementValue(i*ny + j, Yu.getElement(j, i));
				XdVector.setElementValue(i*ny + j, Xd.getElement(j, i));
				YdVector.setElementValue(i*ny + j, Yd.getElement(j, i));
			}
		}
		
		// Compute matrix A
		for(int r = 0; r < numCorresp; r++){
			
			int cc = 0;
			
			for(int k = 0; k <= degree; k++){
				for(int l = 0; l <= (degree-k); l++){
					
					// TODO
				}
			}
		}
		
		// Compute the pseudo-inverse of A with the help of the SVD (class: DecompositionSVD)
		// TODO
		
		// Compute the distortion coefficients
		// TODO
		
		
		/**
		 * 4. Compute the distorted grid points (xDist, yDist) which are used to sample the
		 * distorted image to get the undistorted image
		 * (x,y) is the position in the undistorted image
		 * (xDist,yDist) is the position in the distorted (observed) X-ray image.
		 */

		Grid2D xDist = new Grid2D(imageWidth,imageHeight);  
		Grid2D yDist = new Grid2D(imageWidth,imageHeight); 
		
		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				int cc = 0;
				
				for(int k = 0; k <= degree; k++){
					for(int l = 0; l <= degree - k; l++){
						
						// TODO
					}
				}
			}
		}
		
		undistortedImage = new Grid2D(imageWidth,imageHeight);
		
		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				// TODO
			}
		}
	}
	
	
	/** 
	 * main method
	 * Here you can choose the image, and set the distortion parameters
	 */
	public static void main(String[] args) {
		
		new ImageJ();
		
		int caseNo = 0; // choose 0,1, or 2 for different test images
		
		exercise2 exObj = new exercise2(caseNo);
		
		// generate distorted image
		float a = 0.6f; // a: spread among x-direction
		float b = 0.3f; // b: spread among y-direction
		float d = 12.0f; // d: d/2 value of distortion field at level set for (X/a)^2+(Y/b)^2=1
		exObj.distortedImage = exObj.generateDistortedImage(exObj.originalImage,a,b,d);
		
		exObj.doImageUndistortion();
		
		exObj.originalImage.show("Original Image");
//		exObj.distortionField.show("Artificial Distortion Field");
		exObj.distortedImage.show("Distorted Image");
		exObj.undistortedImage.show("Undistorted Image");
		
		Grid2D differenceImage = (Grid2D) NumericPointwiseOperators.subtractedBy(
				exObj.originalImage, exObj.undistortedImage);
		
		differenceImage.show("Difference Original vs. Undistorted");
		
		Grid2D distortedVSundistorted = (Grid2D) NumericPointwiseOperators.subtractedBy(
				exObj.undistortedImage, exObj.distortedImage);
		
		distortedVSundistorted.show("Difference Undistorted vs. Distorted");
	}

	
	/**
	 * standard constructor
	 */
	public exercise2(){
		
		this(0);
	}
	
	
	/**
	 * parameterized constructor for the exercise2 object
	 * @param caseNo: select one of three images to test your undistortion method
	 */
	public exercise2(int caseNo){
		
		switch (caseNo) {
		case 0:
			filename = "frame32.jpg";
			break;
		case 1:
			filename = "undistorted.jpg";
			break;
		case 2:
			filename = "frame90.jpg";
			break;
		default:
			filename = "frame32.jpg";
			break;
		}
				
		originalImage = ImageUtil.wrapImagePlus(IJ.openImage(getClass().getResource(filename).getPath())).getSubGrid(0);
		
		// Normalize intensity values to [0,1]
		float max = NumericPointwiseOperators.max(originalImage);
		float min = NumericPointwiseOperators.min(originalImage);
		
		for(int i = 0; i < originalImage.getWidth(); i++){
			for(int j = 0; j < originalImage.getHeight(); j++){
				
				float scaledValue = (originalImage.getAtIndex(i, j) - min) / (max-min);
				originalImage.setAtIndex(i, j, scaledValue);
			}
		}
	}

	
	/**
	 * method to generate a distorted image
	 * An artificial distortion field is generated. With this field, 
	 * a distorted image is generated from the original image.
	 * Remark: coordinates of the latest distortion simulated are saved in class member variables
	 * @param inputImage: undistorted image on which distortion is simulated
	 * @param a: elliptic extent of the distortion in x-direction
	 * @param b: elliptic extent of the distortion in y-direction
	 * @param d: strength of the distortion
	 * @return: distorted image
	 */
	private Grid2D generateDistortedImage(Grid2D inputImage, float a, float b, float d){

		Grid2D distortedImage = new Grid2D(inputImage);
		int imageWidth = distortedImage.getWidth();
		int imageHeight = distortedImage.getHeight();
		
		// Generate grids to sample coordinates X and Y of the original image
		Grid2D X = new Grid2D(imageWidth,imageHeight); // x-coordinates with respect to origin in image center, spacing = 1 [unit]
		Grid2D Y = new Grid2D(imageWidth,imageHeight); // y-coordinates with respect to origin in image center, spacing = 1 [unit]
		
		float halfWidth = imageWidth / 2.0f;
		float halfHeight = imageHeight / 2.0f;
		
		for(int i = 0; i < X.getWidth(); i++){
			for(int j = 0; j < X.getHeight(); j++){
				
				X.setAtIndex(i, j, i - halfWidth + 0.5f); // assign pixel centers in x-direction
				Y.setAtIndex(i, j, j - halfHeight + 0.5f); // assign pixel center in y-direction
			}
		}
		
		// Create an artificial elliptical distortion field R (additive field)
		Grid2D R = new Grid2D(imageWidth,imageHeight);

		for(int i = 0; i < R.getWidth(); i++){
			for(int j = 0; j < R.getHeight(); j++){
				
				// distortion gets stronger with the distance from the image center
				float r = d * (float)(Math.sqrt(
						Math.pow(X.getAtIndex(i, j)/(a*halfWidth),2)
						+ Math.pow(Y.getAtIndex(i, j)/(b*halfHeight),2)));
				R.setAtIndex(i, j, r);
			}
		}
		
		// Shifting the positive range to half positive/half negative range
		float maxR = NumericPointwiseOperators.max(R);
		NumericPointwiseOperators.subtractBy(R, maxR * 0.5f);

		// Create the distorted image coordinates:
		// distorted image coordinates = undistorted image points + artificial distortion field

		// distorted image coordinates Xd, Yd
		Grid2D Xd = new Grid2D(X);
		Grid2D Yd = new Grid2D(Y);
		
		NumericPointwiseOperators.addBy(Xd, R);
		NumericPointwiseOperators.addBy(Yd, R);
		
		// Create the distorted image:
		// re-sample the original input image at the distorted image points (Xd,Yd) 
		// to create artificial distortion

		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				float distortedValue = InterpolationOperators.interpolateLinear(
						inputImage, 
						Xd.getAtIndex(i, j) + halfWidth - 0.5f, // remember: coordinate origin was set to image center
						Yd.getAtIndex(i, j) + halfHeight - 0.5f);
				distortedImage.setAtIndex(i, j, distortedValue);
			}
		}

		xprime = X;
		yprime = Y;
		x = Xd;
		y = Yd;
		
		return distortedImage;
	}
}
