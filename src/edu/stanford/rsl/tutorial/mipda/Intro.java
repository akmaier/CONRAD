package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleVector.VectorNormType;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.IJ;
import ij.ImageJ;
import ij.plugin.filter.Convolver;
import ij.process.FloatProcessor;


/**
 * Introduction to the CONRAD Framework
 * Programming exercise for module "Course Introduction"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Frank Schebesch, Marco Boegel, Ashwini Jadhav
 *
 */

public class Intro {
	
	// variables which are checked (DO NOT CHANGE!)
	SimpleVector v1;
	SimpleVector vRand;
	SimpleMatrix M;
	double Mdeterminant;
	SimpleMatrix Mtrans;
	SimpleMatrix Mcopy;
	int numRows;
	int numCols;
	String elementsOutput = "";
	SimpleMatrix Mones;
	SimpleMatrix Mzeros;
	SimpleMatrix Midentity;
	SimpleMatrix ResMat;
	SimpleVector resVec;
	SimpleVector subVector;
	SimpleMatrix MsquaredElem ;
	double matrixNormL1;
	double vecNormL2;
	SimpleVector v2;
	SimpleVector tempColVector;
	int plotLength = 500;
	double[] y = new double[plotLength];
	double[] x = new double [plotLength];
	int imageSizeX = 256, imageSizeY = 256;
	Grid2D image;
	Grid2D copy;
	String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/mipda/";
	String filename = imageDataLoc + "mr12.dcm"; 
	Grid2D mrImage;
	Convolver conv;
    boolean convolution;
    Grid2D  convolvedImage;
    String imageFileName;
    String outFileName;
    
    /**
     * You can change this to a location OUTSIDE of the Conrad workspace, where you want to save the result to.
     * If you leave it as it is, you find the result in your home directory.
     */
    String outputDataLoc = System.getProperty("user.home") + "/mipda/output/";
    
    
    /**
     * In the basicIntro routine you find out how matrix and vector computations
     * can be performed. Follow the code and fill in missing code indicated by TODOs.
     * There are either empty lines which need exactly one line of code,
     * or you have to initialize declared variables appropriately.
     */
	public void basicIntro() {
		
		//display text
		System.out.println("Creating a vector: v1 = [3.0; 2.0; 1.0]");
		
		//create column vector with the entries 3 2 and 1 as floats
		v1 = null; //TODO
		if (v1 != null)
			System.out.println("v1 = " + v1.toString());
		
		//create a randomly initialized vector with values between 0 and 1
		vRand = new SimpleVector(3);
		//TODO
		System.out.println("vRand = " + vRand.toString());
		
		//create a 3x3 matrix M with row vectors (1, 2, 3), (4, 5, 6) and (7, 8, 9)
		//hint: have a look at the method setColValue(int, SimpleVector)
		M = null;//TODO
		//TODO
		//TODO
		//TODO
		if (M != null)
			System.out.println("M = " + M.toString());
		
		//compute the determinant of M
		Mdeterminant = 0.0; //TODO
		System.out.println("Determinant of matrix m: " + Mdeterminant);
		
		//transpose M
		Mtrans = null;//TODO
		//copy matrix using copy constructor
		Mcopy = null;//TODO
		//transpose Mcopy in-place
		//TODO
		//get size
		numRows = 0;//TODO
		numCols = 0;//TODO
		
		//access and print the elements of M (the original matrix from above)
		System.out.println("M: ");
	
		for(int i = 0 ; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				double element = 0;//TODO
				
				elementsOutput = elementsOutput + element + " ";
			    
			}
			elementsOutput += "\n";
		}
		System.out.print(elementsOutput);
		
		//create a 3x3 matrix Mones of ones (all entries are 1)
		Mones = new SimpleMatrix(3,3);
		//TODO
		//create a 3x3 matrix Mzeros of zeros (all entries are 0)
		Mzeros = new SimpleMatrix(3,3);
		//TODO
		//create a 3x3 identity matrix
		Midentity = new SimpleMatrix(3,3);
		//TODO
		
		//matrix multiplication
		//compute the matrix product of Mtrans and M
		//hint: have a look at the class edu.stanford.rsl.conrad.numerics.SimpleOperators
		ResMat = null;//TODO
		if (ResMat != null)
			System.out.println("M^T * M = " + ResMat.toString());
		
		//matrix vector multiplication
		//compute the matrix vector product of M and v1
		// -> SimpleOperators
		resVec = null;//TODO
		if (resVec != null)
			System.out.println("M * v1 = " + resVec.toString());
		
		//extract the last column vector from matrix M
		SimpleVector colVector = null;
		if (M != null) {
			if (M.getCols() > 2) {
				colVector = M.getCol(2);
			}
		}
		
		//extract the top 1x2 subvector from the last column of matrix M 
		subVector = null;//TODO
		if (subVector != null)
			System.out.println("[m(0)(2); m(1)(2)] = " + subVector);
		
		//matrix elementwise multiplication
		//compute a matrix which has the squared value of the elements of M in each component
	    MsquaredElem = null;//TODO
	    if (MsquaredElem != null)
	    	System.out.println("M squared Elements: " + MsquaredElem.toString());
		
		//examples how to round vectors
	    // copy of the random vector vRand
		SimpleVector vRandCopy = new SimpleVector(vRand);
		System.out.println("vRand         = " + vRandCopy.toString());
		// bring down to a round figure
		vRandCopy.floor();
		System.out.println("vRand.floor() = " + vRandCopy.toString());
		// round up
		vRand.ceil();
		System.out.println("vRand.ceil()  = " + vRand.toString());
		
		//compute min and max values
		// for vectors
		if (v1 != null) {
			double minV1 = v1.min();
			double maxV1 = v1.max();
			System.out.println("Min(v1) = " + minV1 + " Max(v1) = " + maxV1);
		}
		// for matrices: iterate over row or column vectors
		if (M != null) {
			SimpleVector maxVec = new SimpleVector(M.getCols());
			for(int i = 0; i < M.getCols(); i++)
			{
				maxVec.setElementValue(i, M.getCol(i).max());
			}
			double maxM = 0; //TODO: compute total maximum
			System.out.println("Max(M) = " + maxM);
		}

		//norms
		matrixNormL1 = 0;//TODO Frobenius norm of M
		vecNormL2 = 0;//TODO L2 vector norm of colVector
		System.out.println("||M||_F = " + matrixNormL1);
		System.out.println("||colVec||_2 = " + vecNormL2);
		
		//get the normalized vector from colVector without overwriting
		v2 = null;//TODO
		
		if (colVector != null)
			tempColVector = new SimpleVector(colVector); //this copy is necessary for unit testing
		else
			tempColVector = new SimpleVector(2);
		
	    //normalize tempColVector vector in-place (overwrite it with normalized values)
		//TODO 
		if (v2 != null)
			System.out.println("Normalized colVector: " + v2.toString());
		System.out.println("||colVec||_2 = " + tempColVector.norm(VectorNormType.VEC_NORM_L2)); // if done correctly this yields 1	
	}
	
	
    /**
     * The signalIntro routine shows how a sine function can be plotted.
     * Follow the code and fill in missing code indicated by TODOs.
     * There are either empty lines which need exactly one line of code,
     * or you have to initialize declared variables appropriately.
     */	
	public void signalIntro()
	{
		//How is the sine function sin(2*PI*x) plotted using this framework?
		double stepSize = 0.01;
		//compute the image of sin(2*PI*x) using the given step size stepSize and starting at the origin
		for(int i = 0; i < y.length; i++)
		{
			y[i] = 0.0; //TODO
			
		}
		VisualizationUtil.createPlot(y).show();
		
		// now plot it with the specified x values
		// (such that it does not show the loop index but the actual x-values on the x-axis)
		for(int i = 0; i < x.length; i++)
		{
			x[i] = 0.0; //TODO
		}
		
		VisualizationUtil.createPlot(x, y, "sin(x)", "x", "y").show();			
	}
	
    /**
     * The gridIntro routine contains code for image manipulation with ImageJ.
     * First a simple image is created which shows a sphere.
     * Next, we load an image from the disk, implement an average filter, and save the result. 
     * Follow the code and fill in missing code indicated by TODOs.
     * There are either empty lines to be filled or you have to initialize declared variables appropriately.
     */	
	public void gridIntro(){
		
		//define the image size
		int imageSizeX = 256;
		int imageSizeY = 256;
	
		//define an image
		//hint: use the package edu.stanford.rsl.conrad.data.numeric.Grid2D
		image = null;//TODO
		
		//draw a filled circle
		int radius = 50;
		//set all pixels within the circle to 100
		int insideVal = 100;
		
		// fill 'image' with data that shows a sphere with radius 'radius' and intensity 'insideVal' 
		//TODO (multiple code lines, hint: two for loops)
		
		//show ImageJ GUI
		ImageJ ij = new ImageJ();
		
		//display image using the Grid2D class methods
		//TODO
		
		//copy an image
       	copy = null;//TODO
       	if (copy != null)
       		copy.show("Copied image of a circle");
		
		//load an image from file
	    // first use IJ to open the image
		// then wrap it using the class ImageUtil (static)
		// finally you need to extract the image with getSubGrid(0)
		mrImage = null;//TODO
		if (mrImage != null)
			mrImage.show("MR image");
		
		// learn how to compute the convolution of an image with a filter kernel
		conv = new Convolver();
		FloatProcessor ip = null;
		if (mrImage != null)
			ip = ImageUtil.wrapGrid2D(mrImage);
		
		//this defines a simple averaging 3x3 filter kernel
		int kw = 3;
		int kh = 3;
		float[] kernel = new float[kw*kh];
		
		for(int i = 0; i < kernel.length; i++)
		{	
			kernel[i] = 1.f / (kw*kh);
		}

		// trigger the convolution using ip, kernel, kw, kh
		convolution = false;//TODO
		if (ip != null) {
			convolvedImage = ImageUtil.wrapFloatProcessor(ip);
			convolvedImage.show("Convolved Image");
		}

		
		//write an image to disk & check the supported output formats
		imageFileName = "mr12out.tiff"; // TODO: choose a valid filename (just file+ending, no path)
		
		outFileName = outputDataLoc + imageFileName;
		if (mrImage != null) {
			//TODO: save the image using IJ and ImageUtil
		}			
	}
	
	/**
	 * 
	 * end of the exercise
	 */
	
	
	public static void main(String[] args) {
		
		(new Intro()).basicIntro();
		(new Intro()).signalIntro();
		(new Intro()).gridIntro();
	}
	
	// getters for members
	// variables which are checked (DO NOT CHANGE!)
	public SimpleVector get_v1() {
		return v1;
	}
	//
	public SimpleVector get_vRand() {
		return vRand;
	}
	//
	public SimpleMatrix get_M() {
		return M;
	}
	//
	public double get_Mdeterminant() {
		return Mdeterminant;
	}
	//
	public SimpleMatrix get_Mtrans() {
		return Mtrans;
	}
	//
	public SimpleMatrix get_Mcopy() {
		return Mcopy;
	}
	//
	public int get_numRows() {
		return numRows;
	}
	//
	public int get_numCols() {
		return numCols;
	}
	//
	public String get_elementsOutput() {
		return elementsOutput;
	}
	//
	public SimpleMatrix get_Mones() {
		return Mones;
	}
	//
	public SimpleMatrix get_Mzeros() {
		return Mzeros;
	}
	//
	public SimpleMatrix get_Midentity() {
		return Midentity;
	}
	//
	public SimpleMatrix get_ResMat() {
		return ResMat;
	}
	//
	public SimpleVector get_resVec() {
		return resVec;
	}
	//
	public SimpleVector get_subVector() {
		return subVector;
	}
	//
	public SimpleMatrix get_MsquaredElem() {
		return MsquaredElem;
	}
	//
	public double get_matrixNormL1() {
		return matrixNormL1;
	}
	//
	public double get_vecNormL2() {
		return vecNormL2;
	}
	//
	public SimpleVector get_v2() {
		return v2;
	}
	//
	public SimpleVector get_tempColVector() {
		return tempColVector;
	}
	//
	public int get_plotLength() {
		return plotLength;
	}
	//
	public double[] get_y() {
		return y;
	}
	//
	public double[] get_x() {
		return x;
	}
	//
	public int get_imageSizeX(){
		return imageSizeX;
	}
	//
	public int get_imageSizeY() {
		return imageSizeY;
	}
	//
	public Grid2D get_image() {
		return image;
	}
	//
	public Grid2D get_copy() {
		return copy;
	}
	//
	public String get_imageDataLoc() {
		return imageDataLoc;
	}
	//
	public String get_filename() {
		return filename;
	}
	//
	public Grid2D get_mrImage() {
		return mrImage;
	}
	//
	public Convolver get_conv() {
		return conv;
	}
	//
	public boolean get_convolution() {
		return convolution;
	}
	//
	public Grid2D get_convolvedImage() {
		return convolvedImage;
	}
	//
	public String get_imageFileName() {
		return imageFileName;
	}
	//
	public String get_outFileName() {
		return outFileName;
	}
}
