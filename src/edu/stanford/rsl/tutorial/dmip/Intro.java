package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
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
 * Exercise 0 of Diagnostic Medical Image Processing (DMIP)
 * @author Frank Schebesch, Marco Boegel
 *
 */

public class Intro {
	
	
	public static void gridIntro(){
				
		//Define the image size
		int imageSizeX = 256;
		int imageSizeY = 384;
	
		//Define an image
		//Hint: Import the package edu.stanford.rsl.conrad.data.numeric.Grid2D
		//TODO
	
		//Draw a circle
		int radius = 55;
		//Set all pixels within the circle to 100
		int insideVal = 100;
	
		//TODO
		//TODO
		//TODO
		
		//Show ImageJ GUI
		ImageJ ij = new ImageJ();
		//Display image
		//TODO
		
		//Copy an image into a new container
		//TODO
		copy.show("Copy of circle");
		
		
		//Load an image from file
		String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/dmip/";

		String filename = imageDataLoc + "mr12.dcm";
		//TODO. Hint: Use IJ and ImageUtil
		mrImage.show();
		
		//prepare convolution by creating the relevant objects
		//TODO
		//TODO
		
		//define the kernel. Try simple averaging 3x3 filter
		int kw = 3;
		int kh = 3;
		float[] kernel = new float[kw*kh];
		for(int i = 0; i < kernel.length; i++)
		{	
			kernel[i] = 1.f / (kw*kh);
		}
		
		// test for error and directly show convolved image
		//TODO
			
		
		//write an image to disk, check the supported output formats
		String outFilename ="G:/DMIP/exercises/2016/0/code/solution/mr12out.tif";
		//TODO
	}
	
	
	public static void signalIntro()
	{
		//How can I plot a sine function sin(x) 
		//which has its zeroes at multiples of 3?
		double stepSize = 0.02;
		int plotLength = 800;
		
		double[] y = new double[plotLength];
		
		for(int i = 0; i < y.length; i++)
		{
			//TODO
			y[i] = val;
			
		}
		
		VisualizationUtil.createPlot(y).show();
		
		// now plot it with the specified x values
		double[] x = new double [plotLength];
		for(int i = 0; i < x.length; i++)
		{
			//TODO
		}
		
		VisualizationUtil.createPlot(x, y, "sin(x)", "x", "y").show();		
		
	}
	
	public static void basicIntro()
	{
		//Display text
		System.out.println("Creating a vector: v1 = [3.0; 2.0; 1.0]");
		
		//create column vector
		//TODO
		System.out.println("v1 = " + v1.toString());
		
		//create a randomly initialized vector
		SimpleVector vRand = new SimpleVector(3);
		//TODO
		System.out.println("vRand = " + vRand.toString());
		
		//create matrix M 3x3  1 2 3; 4 5 6; 7 8 9
		//TODO
		System.out.println("M = " + M.toString());
		
		//determinant of M
		System.out.println("Determinant of matrix m: " + TODO );
		
		//transpose M
		//TODO
		//copy matrix
		//TODO
		//transpose M inplace
		//TODO
		
		//get size
		int numRows = M.getRows();
		int numCols = M.getCols();
		
		//access elements of M
		System.out.println("M: ");
		for(int i = 0 ; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				//TODO
				System.out.print(element + " ");
			}
			System.out.println();
		}
		
		//Create 3x3 Matrix of 1's
		SimpleMatrix Mones = new SimpleMatrix(3,3);
		//TODO
		//Create a 3x3 Matrix of 0's
		SimpleMatrix Mzeros = new SimpleMatrix(3,3);
		//TODO
		//Create a 3x3 Identity matrix
		SimpleMatrix Midentity = new SimpleMatrix(3,3);
		//TODO
		
		//Matrix multiplication
		//TODO
		System.out.println("M^T * M = " + ResMat.toString());
		

		//Matrix vector multiplication
		//TODO
		System.out.println("M * v1 = " + resVec.toString());
		
		
		//Extract the last column vector from matrix M
		SimpleVector colVector = M.getCol(2);
		//Extract the 1x2 subvector from the last column of matrix M
		//TODO
		System.out.println("[m(0)(2); m(1)(2)] = " + subVector);
		
		//Matrix elementwise multiplication
		//TODO
		System.out.println("M squared Elements: " + MsquaredElem.toString());
		
		//round vectors
		SimpleVector vRandCopy = new SimpleVector(vRand);
		System.out.println("vRand         = " + vRandCopy.toString());
		
		vRandCopy.floor();
		System.out.println("vRand.floor() = " + vRandCopy.toString());
		
		vRand.ceil();
		System.out.println("vRand.ceil()  = " + vRand.toString());
		
		//min, max, mean
		double minV1 = v1.min();
		double maxV1 = v1.max();
		System.out.println("Min(v1) = " + minV1 + " Max(v1) = " + maxV1);
		
		//for matrices: iterate over row or column vectors
		SimpleVector maxVec = new SimpleVector(M.getCols());
		for(int i = 0; i < M.getCols(); i++)
		{
			maxVec.setElementValue(i, M.getCol(i).max());
		}
		double maxM = maxVec.max();
		System.out.println("Max(M) = " + maxM);
		
		
		
		//Norms
		//TODO matrix L1
		//TODO vector L2
		System.out.println("||M||_F = " + matrixNormL1);
		System.out.println("||colVec||_2 = " + vecNormL2);
		
		//get normalized vector
		//TODO
		//normalize vector in-place
		//TODO
		System.out.println("Normalized colVector: " + colVector.toString());
		System.out.println("||colVec||_2 = " + colVector.norm(VectorNormType.VEC_NORM_L2));
	}

	public static void main(String arg[])
	{
		basicIntro();
		gridIntro();
		signalIntro();
	}
}
