package edu.stanford.rsl.tutorial.mipda;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.fitting.LinearFunction;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.MatrixNormType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.IJ;
import ij.ImageJ;

/**
 * Singular Value Decomposition
 * Exercise 1 of Diagnostic Medical Image Processing (DMIP)
 * @author Frank Schebesch, Marco Boegel
 *
 */
 
public class exercise1 {
	
	public static void invertSVD(SimpleMatrix A)
	{
		
					
		System.out.println("A = " + A.toString());
		
		//Compute the inverse of A without using inverse()				
		//TODO
		
		//Check output: re-compute A = U * S * V^T
		SimpleMatrix temp = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		SimpleMatrix A2 = SimpleOperators.multiplyMatrixProd(temp, svd.getV().transposed());
		System.out.println("U * S * V^T: " + A2.toString());
		
		//Moore-Penrose Pseudoinverse defined as V * Sinv * U^T
		//Compute the inverse of the singular matrix S
		SimpleMatrix Sinv = new SimpleMatrix( svd.getS().getRows(), svd.getS().getCols());
		Sinv.zeros();

		int size = Math.min(Sinv.getCols(), Sinv.getRows());
		SimpleVector SinvDiag = new SimpleVector( size);
		
		for(int i = 0; i < Sinv.getRows(); i++)
		{
			double val = 1.0 / svd.getS().getElement(i, i);
			SinvDiag.setElementValue(i, val);
		}
		
		Sinv.setDiagValue(SinvDiag);
		
		//Compare our implementation to svd.getreciprocalS()
		System.out.println("Sinv = " + Sinv.toString());
		System.out.println("Srec = " +svd.getreciprocalS().toString());
		
		
		SimpleMatrix tempInv = SimpleOperators.multiplyMatrixProd(svd.getV(), svd.getreciprocalS());
		SimpleMatrix Ainv = SimpleOperators.multiplyMatrixProd(tempInv, svd.getU().transposed());
		System.out.println("Ainv        = " + Ainv.toString());
		System.out.println("A.inverse() = " + A.inverse(InversionType.INVERT_SVD));
		
		double cond = svd.cond();
		System.out.println("Cond(A) = " + cond);
		
		//introduce a rank deficiency
		double eps = 10e-3;
		SimpleMatrix Slowrank = new SimpleMatrix(svd.getS());
		int sInd = //TODO
		for(int i = 0; i < sInd; i++)
		{
			double val = svd.getS().getElement(i, i);
			if(val> eps)
			{
				Slowrank.setElementValue(i, i, val);
			}
		}
		
		SimpleMatrix templowrank = SimpleOperators.multiplyMatrixProd(svd.getU(), Slowrank);
		SimpleMatrix Alowrank = //TODO
		System.out.println("A rank deficient = " + Alowrank.toString());
		
		
		
		//Show that a change in a vector b of only 0.1% can lead to 240% change in the result of A*x=b
		//Consider A as already defined
		//And vector b
		SimpleVector b = new SimpleVector(1.001, 0.999, 1.001);
		
		//solve for x
		SimpleVector x = SimpleOperators.multiply(Ainv, b);
		System.out.println(x.toString());
		
		//if we set vector b to the vector consisting of 1's, we imply a 0.1% change
		SimpleVector br = new SimpleVector(1,1,1);
		
		
		SimpleVector xr = SimpleOperators.multiply(Ainv, br);
		System.out.println(xr.toString());
		//We want only the difference caused by the change	
		SimpleVector xn = //TODO
		
		//Alternatively:
		//SimpleVector bn = SimpleOperators.subtract(br, b);
		//SimpleVector xn = SimpleOperators.multiply(Ainv, bn);
				
		// compute and show percentual change
		//TODO
	}
	
	public static void optimizationProblem1(SimpleMatrix A, int rankDeficiency)
	{
		System.out.println("Optimization Problem 1");
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Optimization problem I
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// Let us consider the following problem that appears in many image
		// processing and computer vision problems:
		// We computed a matrix A out of sensor data like an image. By theory
		// the matrix A must have the singular values s1, s2, . . . , sp, where
		// p = min(m, n). Of course, in practice A does not fulfill this constraint.
		// Problem: What is the matrix A0 that is closest to A (according to the
		// Frobenius norm) and has the required singular values?


		// Let us assume that by theoretical arguments the matrix A is required
		// to have a rank deficiency of one, and the two non-zero singular values
		// are identical. The matrix A0 that is closest to A according to the
		// Frobenius norm and fulfills the above requirements is:

		// Build mean of first two singular values to make them identical
		// Set third singular value to zero
		
		DecompositionSVD svd = new DecompositionSVD(A);
		int rank = svd.rank();
		
		int newRank = //TODO
		
		
		
		//Compute mean of remaining singular values
		double mean = 0;
		for(int i = 0; i < newRank; i++)
		{
			mean += svd.getS().getElement(i, i);
		}
		mean /= newRank;
		
		//Create new Singular matrix
		SimpleMatrix Slowrank = new SimpleMatrix(svd.getS().getRows(), svd.getS().getCols());
		Slowrank.zeros();
		//Fill in remaining singular values with the mean.
		for(int i = 0; i < newRank; i++)
		{
			Slowrank.setElementValue(i, i, mean);
		}
		
		//compute A0
		//TODO
		//TODO
		
		System.out.println("A0 = " + A0.toString());
		
		double normA = A.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		double normA0 = A0.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		
		System.out.println("||A||_F  = " + normA);
		System.out.println("||A0||_F = " + normA0);
		
		
	}
	
	
	public static void optimizationProblem2(SimpleMatrix b)
	{
		System.out.println("Optimization Problem 2");
		// Estimate the matrix A in R^{2,2} such that for the following vectors
		// the optimization problem gets solved:
		// sum_{i=1}^{4} b'_{i} A b_{i} and ||A||_{F} = 1
		// The objective function is linear in the components of A, thus the whole
		// sum can be rewritten in matrix notation:
		// Ma = 0
		// where the measurement matrix M is built from single elements of the
		// sum.
		// Given the four points we get four equations and therefore four rows in M:
		
		SimpleMatrix M = new SimpleMatrix( b.getCols(), 4);
		
		for(int i = 0; i < b.getCols(); i++)
		{	
			// i-th column of b contains a vector. First entry of M is x^2
			M.setElementValue(i, 0, Math.pow(b.getElement(0, i), 2.f));
			M.setElementValue(i, 1, b.getElement(0, i)*b.getElement(1, i));
			M.setElementValue(i, 2, M.getElement(i, 1));
			M.setElementValue(i, 3, Math.pow(b.getElement(1, i), 2.f));
		}
		
		// TASK: estimate the matrix A
		// SOLUTION: Compute vector that spans the nullspace of A; we can find it in
		// the last column vector of V.
		DecompositionSVD svd = new DecompositionSVD(M);
		int lastCol = svd.getV().getCols() -1;
		SimpleVector a = svd.getV().getCol(lastCol);
		
		//We need to reshape the vector back to the desired 2x2 matrix form
		SimpleMatrix A = new SimpleMatrix(2,2);
		A.setColValue(0, a.getSubVec(0, 2));
		A.setColValue(1, a.getSubVec(2, 2));
		
		//check if Frobenius norm is 1.0
		double normF = A.norm(MatrixNormType.MAT_NORM_FROBENIUS);
		System.out.println("||A||_F = " + normF);
		
		//check solution
		
		SimpleVector temp = SimpleOperators.multiply(M, a);
		double result = SimpleOperators.multiplyInnerProd(a, temp);
		System.out.println("Minimized error: " + result);
	}
	
	public static void optimizationProblem3(Grid2D image, int rank)
	{
		//%
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Optimization problem III
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// The SVD can be used to compute the image matrix of rank 1 that best
		// approximates an image. Figure 1 shows an example of an image I
		// and its rank-1-approximation I0 = u_{1} * s_{1} * v'_{1}.
		
		//In order to apply the svd, we first need to transfer our Grid2D image to a matrix
		SimpleMatrix I = new SimpleMatrix(image.getHeight(), image.getWidth());
		
		for(int i = 0; i < image.getHeight(); i++)
		{
			for(int j = 0; j < image.getWidth(); j++)
			{
				double val = image.getAtIndex(j, i);
				I.setElementValue(i, j, val);
			}
		}		
		
		DecompositionSVD svd = new DecompositionSVD(I);
		
		Grid3D imageRanks = new Grid3D(image.getWidth(), image.getHeight(), rank);
	
		// track error
		float[] error = new float[rank];
		NumericGridOperator op = new NumericGridOperator();
		
		//Create Rank k approximations
		for(int k = 0; k < rank; k++)
		{
			SimpleVector us = svd.getU().getCol(k).multipliedBy(svd.getSingularValues()[k]);
			SimpleMatrix Iapprox = SimpleOperators.multiplyOuterProd(us, svd.getV().getCol(k));
	
			//Transfer back to grid
			Grid2D imageRank = new Grid2D(image.getWidth(),image.getHeight());
			for(int i = 0; i < image.getHeight(); i++)
			{
				for(int j = 0; j < image.getWidth(); j++)
				{
					if(k == 0)
					{
						imageRank.setAtIndex(j, i, (float) Iapprox.getElement(i, j));
			
					}
					else
					{
						imageRank.setAtIndex(j, i, imageRanks.getAtIndex(j, i, k-1) + (float) Iapprox.getElement(i, j));
					}
					
					
				}
			}
			imageRanks.setSubGrid(k, imageRank);

			// evaluate RMSE for the k-th approximation
			//TODO
		}
		
		imageRanks.show("Stack of approximations");
		VisualizationUtil.createPlot("Rank-Error-Plot", error).show();
		
		//Direct estimation of rank K (using rank 1-matrices, cf. lecture) [alternative solution]
		SimpleMatrix usK = SimpleOperators.multiplyMatrixProd(svd.getU().getSubMatrix(0, 0, svd.getU().getRows(), rank), svd.getS().getSubMatrix(0, 0, rank, rank));
		SimpleMatrix IapproxK = SimpleOperators.multiplyMatrixProd(usK, svd.getV().getSubMatrix(0, 0, svd.getV().getRows(), rank).transposed());
		//Transfer back to grid
		Grid2D imageRankK = new Grid2D(image.getWidth(), image.getHeight());
		for(int i = 0; i < image.getHeight(); i++)
		{
			for(int j = 0; j < image.getWidth(); j++)
			{
					imageRankK.setAtIndex(j, i, (float) IapproxK.getElement(i, j));
			}
		}
		imageRankK.show("Direct output from rank 1-matrices");
		
	}

	public static void optimizationProblem4(double[] xCoords, double[] yCoords)
	{
		System.out.println("Optimization Problem 4");
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//Optimization problem IV
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// The Moore-Penrose pseudo-inverse is required to find the
		// solution to the following optimization problem:
		// Compute the regression line through the following 2-D points:
		// We have sample points (x_i,y_i) and want to fit a line model
		// y = mx + t (linear approximation) through the points
		// y = A*b
		// Note: y = m*x +t*1  Thus: A contains (x_i 1) in each row, and y contains the y_i coordinates
		// We need to solve for b = (m t)
		
		SimpleMatrix A = new SimpleMatrix(xCoords.length, 2);
		SimpleVector aCol = new SimpleVector(xCoords.length);
		SimpleVector y = new SimpleVector(xCoords.length);
		//TODO
		//TODO
		//TODO
		A.setColValue(0, aCol);
		SimpleVector aColOnes = new SimpleVector(xCoords.length);
		aColOnes.ones();
		A.setColValue(1, aColOnes);
		
		

		// Note: We need to use SVD matrix inversion because the matrix is not square
		// more samples (7) than unknowns (2) -> overdetermined system -> least-square
		// solution.
		SimpleMatrix Ainv = A.inverse(InversionType.INVERT_SVD);		
		SimpleVector b = SimpleOperators.multiply(Ainv, y);
	
		
		LinearFunction lFunc = new LinearFunction();
		lFunc.setM(b.getElement(0));
		lFunc.setT(b.getElement(1));
		VisualizationUtil.createScatterPlot(xCoords, yCoords, lFunc).show();
		
	}
	
	public static void main(String[] args) {
		
		ImageJ ij = new ImageJ();
		//Create matrix A 
		// 11 10  14
		// 12 11 -13
		// 14 13 -66
		SimpleMatrix A = new SimpleMatrix(3,3);
		A.setRowValue(0, new SimpleVector(11, 10,  14));
		A.setRowValue(1, new SimpleVector(12, 11, -13));
		A.setRowValue(2, new SimpleVector(14, 13, -66));
		
		invertSVD(A);
		optimizationProblem1(A, 1);

		// Data for problem 2 from lecture slides
		SimpleMatrix vectors = new SimpleMatrix(2,4);
		vectors.setColValue(0, new SimpleVector(1.f, 1.f));
		vectors.setColValue(1, new SimpleVector(-1.f, 2.f));
		vectors.setColValue(2, new SimpleVector(1.f, -3.f));
		vectors.setColValue(3, new SimpleVector(-1.f, -4.f));
		
		optimizationProblem2(vectors);
		
		//Load the head angiography image image
		String imageDataLoc = System.getProperty("user.dir") + "/data/" + "/dmip/";
		String filename = "mr_head_angio.jpg";
		Grid2D image = ImageUtil.wrapImagePlus(IJ.openImage(imageDataLoc + filename)).getSubGrid(0);
		image.show("mr_head_angio:original");
			
		int rank = 150;	
		optimizationProblem3(image, rank);
		
		//Data for problem 4
		double[] xCoords = new double[]{-3.f, -2.f, -1.f, 0.f, 1.5f, 2.f, 3.1f, 5.9f, 7.3f};
		double[] yCoords = new double[]{7.f, 8.f, 9.f, 3.3f, 2.f, -3.f, 4.f, -0.1, -0.5};
	
		optimizationProblem4(xCoords, yCoords);

	}

}
