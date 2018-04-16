package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.utils.ImageUtil;
//import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.IJ;
import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
//import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;

/**
 * Geometric Undistortion
 * Programming exercise 1 for module "Defect Pixel Interpolation"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch, Ashwini Jadhav, Anna Gebhard, Mena Abdelmalek
 *
 */

public class ExerciseGeoU {

	final static int caseNo = 0; // choose 0, 1, or 2 for different test images

	final static float a = 0.6f; // a: spread among x-direction
	final static float b = 0.3f; // b: spread among y-direction
	final static float d = 5.0f; // d: d/2 value of distortion field at level set for (X/a)^2+(Y/b)^2=1
	
	String filename; // do not edit, choose a number in the main method instead
	Grid2D originalImage; // do not edit, read from filename when instantiating the object
	Grid2D distortedImage; // do not edit, generated in the main method
	Grid2D undistortedImage; // do not edit, member variable for the output image
	
	// number of lattice points
	final int nx = 0;//TODO: define the number of lattice point: nx (Positive number and less than 20)
	final int ny = 0;//TODO: define the number of lattice point: ny (Positive number and less than 20)
	
	float fx;
	float fy;
	SimpleMatrix Xu; 
	SimpleMatrix Yu;
	SimpleMatrix Xd;
	SimpleMatrix Yd;
	int numCoeff; 
	int numCorresp; 
	SimpleMatrix A;
	DecompositionSVD svd;
	SimpleMatrix A_pseudoinverse;
	SimpleVector u_vec;
	SimpleVector v_vec;
	SimpleVector XuVector;
	SimpleVector YuVector;
	SimpleVector XdVector;
	SimpleVector YdVector; 
	Grid2D xDist;  
	Grid2D yDist;
	Grid2D xprime; // x-coordinates of point correspondences in the undistorted image
	Grid2D yprime; // y-coordinates of point correspondences in the undistorted image
	Grid2D x; // x-coordinates of point correspondences in the distorted image
	Grid2D y; // y-coordinates of point correspondences in the distorted image
	
	
	/** 
	 * main method
	 * Here you can choose the image, and set the distortion parameters
	 */
	public static void main(String[] args) {
		
		new ImageJ();
		
		// generate distorted image
		ExerciseGeoU exObj = new ExerciseGeoU(caseNo);
		exObj.init(a,b,d);
		
		// TASK: go to the method doImageUndistortion(...) and complete it 
		exObj.undistortedImage = exObj.doImageUndistortion(exObj.distortedImage);
		
		exObj.originalImage.show("Original Image");
		exObj.distortedImage.show("Distorted Image");
		exObj.undistortedImage.show("Undistorted Image");
		
		Grid2D differenceImage = (Grid2D) NumericPointwiseOperators.subtractedBy(
				exObj.originalImage, exObj.undistortedImage);
		
		differenceImage.show("Difference Original vs. Undistorted");
		
		Grid2D distortedVSundistorted = (Grid2D) NumericPointwiseOperators.subtractedBy(
				exObj.undistortedImage, exObj.distortedImage);
		
		distortedVSundistorted.show("Difference Undistorted vs. Distorted");
	}
	
	public Grid2D doImageUndistortion(Grid2D distortedImage){
		
		Grid2D grid = new Grid2D(distortedImage);
		
		// check if image distortion routine has been run
		if (distortedImage == null || undistortedImage == null
				|| xprime == null || yprime == null
				|| x == null || y == null) {
			
			System.err.println("Error in doImageUndistortion(): called before initialization of members.");
			return grid;
		}
		
		
		getLatticePoints(distortedImage);// There are TODOs here 
		
		
		int degree = 5; // polynomial's degree: 2,...,10
		calcPolynomial(degree, Xd);// There are TODOs here
		
		
		
		computeMatrixA(degree, numCorresp, numCoeff);// There are TODOs here
		
		
		computeDistortionCoeff(A, XuVector, YuVector);// There are TODOs here
		
		
		grid = computeDistortedGrid(distortedImage, grid, degree);// There are TODOs here
		
		return grid;
	}
	
	
	
	/** 1. Number of lattice points (this only works for symmetric images):
	 *  nx, ny feature points: usually provided by tracking point
	 *  correspondences in the phantom during the calibration step.
	 *  Here, we use the distorted and undistorted coordinates generated
	 *  in the course of the distortion simulation.
	 */ 
	public void getLatticePoints (Grid2D distortedImage){
		
		
		int imageWidth = distortedImage.getWidth();
		int imageHeight = distortedImage.getHeight();
		
		// step size
		// calculate the step size of the lattice points: fx and fy 
		fx = 0; //TODO
		fy = 0; //TODO
		
		// fill the distorted and undistorted lattice points 
		// with data from the given correspondences
		// matrix = number of rows (y) x number of columns (x)
		Xu = new SimpleMatrix(ny,nx);
		Yu = new SimpleMatrix(ny,nx);
		Xd = new SimpleMatrix(ny,nx);
		Yd = new SimpleMatrix(ny,nx);
		
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){

				// sample the distorted and undistorted grid points at the lattice points
				//TODO: fill matrix Xu
				//TODO: fill matrix Yu
				//TODO: fill matrix Xd
				//TODO: fill matrix Yd
			}
		}
	}
	
	/** 2. Polynomial of degree d
	 * Polynomial of degree d -> (d-1): extrema (e.g., d=5: 4 extrema)
	 * d=0: constant (horizontal line with y-intercept a_0 -> f(x)=a_0)
	 * d=1: oblique line with y-intercept a_0 & slope a_1 -> f(x)=a_0 + a_1 x
	 * d=2: parabola
	 * d>=2: continuous non-linear curve
	 * ...
	 * d = 10 -> NumCoeff: 66 -> but only 64 lattice points are known (nx, ny = 8)
	 */
	
	public void calcPolynomial(int degree, SimpleMatrix Xd){
		
		// number of coefficients: numCoeff
		// (hint: this is NOT the total number of multiplications!)
		numCoeff = 0; //TODO
		
		// number of correspondences: numCorresp
		numCorresp =  0; //TODO
		
		// Printout of the used parameters
		System.out.println("Polynom of degree: " + degree);
		System.out.println("Number of Coefficients: " + numCoeff);
		System.out.println("Number of Correspondences: " + numCorresp);
		
	}
	
	/**
	 * 3. Create the matrix A
	 */
	
	public void computeMatrixA(int degree, int numCorresp, int numCoeff){
		
		A = new SimpleMatrix(numCorresp, numCoeff);
		A.zeros();
		
		// Realign the grid matrix into a vector for easier access in the next step
		XuVector = new SimpleVector(numCorresp);
		YuVector = new SimpleVector(numCorresp);
		XdVector = new SimpleVector(numCorresp);
		YdVector = new SimpleVector(numCorresp);
		
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){
				
				XuVector.setElementValue(i*ny + j, Xu.getElement(j, i));
				YuVector.setElementValue(i*ny + j, Yu.getElement(j, i));
				XdVector.setElementValue(i*ny + j, Xd.getElement(j, i));
				YdVector.setElementValue(i*ny + j, Yd.getElement(j, i));
			}
		}
		
		// Compute matrix A (coordinates from distorted image)
		for(int r = 0; r < numCorresp; r++){
			
			int cc = 0;
			
			for(int k = 0; k <= degree; k++){
				for(int l = 0; l <= (degree-k); l++){
					
					// TODO: fill matrix A
					cc++;
					
				}
			}
		}
	}
	
	public void computeDistortionCoeff(SimpleMatrix A, SimpleVector XuVector, SimpleVector YuVector){
		
		// Compute the pseudo-inverse of A with the help of the SVD (class: DecompositionSVD)
		svd = null; // TODO
	    A_pseudoinverse = null; // TODO
	  
		// Compute the distortion coefficients (solve for known corresponding undistorted points)
		u_vec = null;// TODO
		v_vec = null;// TODO
	}
	
	/**
	 * 4. Compute the distorted grid points (xDist, yDist) which are used to sample the
	 * distorted image to get the undistorted image
	 * (xprime,yprime) is the position in the undistorted image
	 * (xDist,yDist) is the position in the distorted (observed) X-ray image.
	 */
	public Grid2D computeDistortedGrid(Grid2D distortedImage, Grid2D grid_out, int degree){
		
		int imageWidth = distortedImage.getWidth();
		int imageHeight = distortedImage.getHeight();
		
		xDist = new Grid2D(imageWidth,imageHeight);  
		yDist = new Grid2D(imageWidth,imageHeight); 
		
		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				float val1, val2; //variables Val1 and Val2 are used for intermediate computation of xDist and yDist respectively
				
				int cc = 0;
				
				for(int k = 0; k <= degree; k++){
					for(int l = 0; l <= degree - k; l++){
						
						val1 = 0;// TODO
						val2 = 0;// TODO
						
						// TODO: fill xDist
						// TODO: fill yDist
						
						cc++;
					}
				}
			}
		}
		
		float val; // variable for intermediate computations of undistorted image
		
		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				// hint: consider the fact that the coordinate origin is in the center of the image
				val = 0;//TODO
				//TODO: fill grid_out
			}
		}
		
		return grid_out;
		
	}
	
	
	
	/**
	 * 
	 * end of the exercise
	 */	

	
	/**
	 * standard constructor
	 */
	public ExerciseGeoU(){
		
		this(0);
	}
		
	/**
	 * parameterized constructor for the ExerciseDPI object
	 * @param caseNo: select one of three images to test your undistortion method
	 */
	public ExerciseGeoU(int caseNo){
		
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
		
		String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/mipda/";
	
		originalImage = ImageUtil.wrapImagePlus(IJ.openImage(imageDataLoc + filename)).getSubGrid(0);
	}
	
	public void init(float a, float b, float d) {
		
		// Normalize intensity values to [0,1]
		float max = NumericPointwiseOperators.max(originalImage);
		float min = NumericPointwiseOperators.min(originalImage);
		
		for(int i = 0; i < originalImage.getWidth(); i++){
			for(int j = 0; j < originalImage.getHeight(); j++){
				
				float scaledValue = (originalImage.getAtIndex(i, j) - min) / (max-min);
				originalImage.setAtIndex(i, j, scaledValue);
			}
		}
		
		undistortedImage = new Grid2D(originalImage.getWidth(),originalImage.getHeight()); // initialization with zeros
		
		distortedImage = generateDistortedImage(originalImage,a,b,d);	
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
    
	public Grid2D generateDistortedImage(Grid2D inputImage, float a, float b, float d){

		Grid2D grid = new Grid2D(inputImage);
		int imageWidth = grid.getWidth();
		int imageHeight = grid.getHeight();
		
		// generate grids to sample coordinates X and Y of the original image
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
		
		// create an artificial elliptical distortion field R (additive field)
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
		
		// shifting the positive range to half positive/half negative range
		float maxR = NumericPointwiseOperators.max(R);
		NumericPointwiseOperators.subtractBy(R, maxR * 0.5f);

		// create the distorted image coordinates:
		// distorted image coordinates = undistorted image points + artificial distortion field

		// distorted image coordinates Xd, Yd
		Grid2D Xd = new Grid2D(X);
		Grid2D Yd = new Grid2D(Y);
		
		NumericPointwiseOperators.addBy(Xd, R);
		NumericPointwiseOperators.addBy(Yd, R);
		
		// create the distorted image:
		// re-sample the original input image at the distorted image points (Xd,Yd) 
		// to create artificial distortion
		for(int i = 0; i < imageWidth; i++){
			for(int j = 0; j < imageHeight; j++){
				
				float distortedValue = InterpolationOperators.interpolateLinear(
						inputImage,
						Xd.getAtIndex(i, j) + halfWidth - 0.5f, // remember: coordinate origin was set to image center
						Yd.getAtIndex(i, j) + halfHeight - 0.5f);
				grid.setAtIndex(i, j, distortedValue);
			}
		}

		xprime = X;
		yprime = Y;
		x = Xd;
		y = Yd;
		
		return grid;	
	}	
	
	//getters for members
	// variables which are checked (DO NOT CHANGE!)
	
	public static int get_caseNo() {
		return caseNo;
	}
	
	public Grid2D get_originalImage() {
		return originalImage;
	}
	
	public Grid2D get_distortedImage() {
		return distortedImage;
	}
	
	public Grid2D get_undistortedImage() {
		return undistortedImage;
	}
	
	public int get_nx() {
		return nx;
	}
	
	public int get_ny() {
		return ny;
	}
	
	public float get_fx() {
		return fx;
	}
	
	public float get_fy() {
		return fy;
	}
	
	public SimpleMatrix get_Xu() {
		return Xu;
	}
	
	public SimpleMatrix get_Yu() {
		return Yu;
	}
	
	public SimpleMatrix get_Xd() {
		return Xd;
	}
	
	public SimpleMatrix get_Yd() {
		return Yd;
	}
	
	public int get_numCoeff() {
		return numCoeff;
	}
	
	public int get_numCorresp() {
		return numCorresp;
	}
	
	public SimpleMatrix get_A() {
		return A;
	}
	
	public DecompositionSVD get_svd() {
		return svd;
	}
	
	public SimpleMatrix get_A_pseudoinverse() {
		return A_pseudoinverse;
	}
	
	public SimpleVector get_u_vec() {
		return u_vec;
	}
	
	public SimpleVector get_v_vec() {
		return v_vec;
	}
	
	public SimpleVector get_XuVector() {
		return XuVector;
	}
	
	public SimpleVector get_YuVector() {
		return YuVector;
	}
	
	public SimpleVector get_XdVector() {
		return XdVector;
	}
	
	public SimpleVector get_YdVector() {
		return YdVector;
	}
	
	public Grid2D get_xDist() {
		return xDist;
	}
	
	public Grid2D get_yDist() {
		return yDist;
	}
	
	public Grid2D get_xprime() {
		return xprime;
	}
	
	public Grid2D get_yprime() {
		return yprime;
	}
	
	public Grid2D get_x() {
		return x;
	}
	
	public Grid2D get_y() {
		return y;
	}
	
}
