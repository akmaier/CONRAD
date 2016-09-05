package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.gui.Plot;


/**
 * Exercise 7 of Diagnostic Medical Image Processing (DMIP)
 * @author Bastian Bier
 *
 */
public class Registration1 {
	
	/**
	 * Registration method
	 * Point Correspondences are used to calculate the rotation and translation
	 * Transformation from p to q
	 * @param p reference point cloud
	 * @param q point cloud to be registered
	 * @return translation and rotation for the registration (phi, t1, t2)
	 */
	private SimpleVector registrationUsingPointCorrespondences(SimpleMatrix p, SimpleMatrix q){
		
		int numPoints = p.getRows();
		
		// Build up measurement matrix m
		SimpleMatrix m = new SimpleMatrix(numPoints * 2, 4);
		
		for(int i = 0; i < numPoints * 2; i++)
		{
			if(i < numPoints)
			{
				// TODO
				// TODO
				// TODO
				// TODO
			}
			if(i >= numPoints)
			{
				// TODO
				// TODO
				// TODO
				// TODO
			}
		}
		
		// Build up solution vector b
		SimpleVector b = new SimpleVector(2 * numPoints);
		
		for(int i = 0; i < 2 * numPoints; i++)
		{
			if(i < numPoints)
			{
				// TODO
			}
			
			if(i >= numPoints)
			{
				// TODO
			}
		}
		
		// Calculate Pseudo Inverse of m
		SimpleMatrix m_inv;
		m_inv = m.inverse(InversionType.INVERT_SVD);
		
		// Calculate Parameters with the help of the Pseudo Inverse
		SimpleVector x;
		x = SimpleOperators.multiply(m_inv, b);
		
		double r1 = x.getElement(0);
		double r2 = x.getElement(1);
		double t1 = x.getElement(2);
		double t2 = x.getElement(3);
		
		
		// TODO: normalize r
		
		
		double phi = 0; // TODO
		
		// Write the result for the translation and the rotation into the result vector
		SimpleVector result = new SimpleVector(phi, t1, t2);
		
		return result;
	}
	
	
	private SimpleMatrix applyTransform(SimpleMatrix points, double phi, SimpleVector translation){
		
		SimpleMatrix r = new SimpleMatrix(2,2);
		// TODO: fill the rotation matrix
		
		// TODO
		// TODO
		// TODO
		// TODO

		SimpleMatrix transformedPoints = new SimpleMatrix(points.getRows(), points.getCols());
				
		for(int i = 0; i < transformedPoints.getRows(); i++)
		{
			// TODO: transform points
			
		}
		
		return transformedPoints;
	}
	
	public static void main(String[] args){
		
		// Define Point Cloud 
		SimpleMatrix p_k = new SimpleMatrix(4,2);
		SimpleMatrix q_k = new SimpleMatrix(4,2);
		
		p_k.setRowValue(0, new SimpleVector(1,2));
		p_k.setRowValue(1, new SimpleVector(3,6));
		p_k.setRowValue(2, new SimpleVector(4,6));
		p_k.setRowValue(3, new SimpleVector(3,4));
		
		q_k.setRowValue(0, new SimpleVector(-0.7, 2.1));
		q_k.setRowValue(1, new SimpleVector(-2.1, 6.4));
		q_k.setRowValue(2, new SimpleVector(-1.4, 7.1));
		q_k.setRowValue(3, new SimpleVector(-0.7, 4.9));
		
		// Plot Point Cloud
		Plot plot = new Plot("Regression Line", "X", "Y", Plot.DEFAULT_FLAGS);
		plot.setLimits(-10, 10, -10, 10);
		plot.addPoints(p_k.getCol(0).copyAsDoubleArray(), p_k.getCol(1).copyAsDoubleArray(), Plot.BOX);
		plot.addPoints(q_k.getCol(0).copyAsDoubleArray(), q_k.getCol(1).copyAsDoubleArray(), Plot.CIRCLE);
		
		// Calculate registration parameter
		Registration1 reg = new Registration1();
		SimpleVector parameter = reg.registrationUsingPointCorrespondences(p_k, q_k);
		
		// Rotation and translation
		double phi = parameter.getElement(0);
		SimpleVector translation = new SimpleVector(parameter.getElement(1), parameter.getElement(2));
		
		// Transform points
		SimpleMatrix transformedPoints = reg.applyTransform(q_k, phi, translation);
		
		// Add transformed point cloud to the plot
		plot.addPoints(transformedPoints.getCol(0).copyAsDoubleArray(), transformedPoints.getCol(1).copyAsDoubleArray(), Plot.CROSS);
		plot.show();
		
	}
	
	
}
