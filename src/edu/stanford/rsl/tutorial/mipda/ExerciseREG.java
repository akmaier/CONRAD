package edu.stanford.rsl.tutorial.mipda;

import java.awt.Color;

import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.gui.Plot;


/**
 * Registration of Point Correspondences
 * Programming exercise for chapter "Rigid Registration"
 * of the course "Medical Image Processing for Diagnostic Applications (MIPDA)"
 * @author Bastian Bier, Frank Schebesch, Anna Gebhard, Mena Abdelmalek, Ashwini Jadhav
 *
 */

public class ExerciseREG {

	SimpleMatrix p_k;
	SimpleMatrix q_k;
	Plot plot;

	public static void main(String[] args){
		
		ExerciseREG exREGobj = new ExerciseREG();
		
		// calculate registration parameters: rotation angle and translation vector
		RigidParameters parameter = exREGobj.registerPoints(exREGobj.p_k, exREGobj.q_k); //TODO: Go to this method and implement the missing steps!
		
		// transform points
		SimpleMatrix transformedPoints = exREGobj.applyTransform(exREGobj.q_k, parameter); //TODO: Go to this method and implement the missing steps!
		
		// add transformed point cloud into the plot
		exREGobj.plot.setColor(Color.red);
		exREGobj.plot.addPoints(transformedPoints.getCol(0).copyAsDoubleArray(), transformedPoints.getCol(1).copyAsDoubleArray(), Plot.CROSS);
		exREGobj.plot.show();
	}
	
	/**
	 * Registration method
	 * Point correspondences are used to calculate the optimal rotational and translational
	 * transformations from p to q
	 * @param p reference point cloud
	 * @param q point cloud to be registered
	 * @return parameter object, contains rotation angle and translation vector (phi, [t1, t2])
	 */	
	public RigidParameters registerPoints(SimpleMatrix p, SimpleMatrix q){
		
		int numPoints = p.getRows();
		
		// build up measurement matrix m for the complex numbers problem  
		SimpleMatrix m = buildMeasurementMatrix(q);// TODO-s here!
		
		// build up right hand side vector b for the complex numbers problem
		SimpleVector b = buildRHSVector(p);// TODO-s here!
		
		// just some check (ignore)
		assert(m.getRows() == 2*numPoints);
		
		// solve the problem
		SimpleMatrix m_inv = m.inverse(InversionType.INVERT_SVD);// calculate pseudo inverse of m
		SimpleVector x = SimpleOperators.multiply(m_inv, b);// solve for parameters
		
		double r1 = x.getElement(0);
		double r2 = x.getElement(1);
		double t1 = x.getElement(2);
		double t2 = x.getElement(3);
		
		//calculate the absolute value of r = r1+i*r2
		double abs_r = 0.d;//TODO
		//TODO: normalize r
		//TODO: normalize r
		
		//calculate the angle phi
		double phi = 0.d;//TODO
		
		// return both translation and rotation in a RigidParameters object
		return new RigidParameters(phi,new SimpleVector(t1,t2));
	}
	
	public SimpleMatrix buildMeasurementMatrix(SimpleMatrix q) {
		
		int numPoints = q.getRows();
		
		SimpleMatrix m = new SimpleMatrix(2*numPoints, 4);
		
		// build up measurement matrix m for the complex numbers problem  
		for(int i = 0; i < numPoints; i++) {
			
			//real part of the problem (even rows)
				//TODO
				//TODO
				//TODO
				//TODO
			//imaginary part of the problem (odd rows)
				//TODO
				//TODO
				//TODO
				//TODO
		}
		
		return m;
	}

	public SimpleVector buildRHSVector(SimpleMatrix p) {
		
		int numPoints = p.getRows();

		SimpleVector b = new SimpleVector(2*numPoints);// right hand side
		
		for(int i = 0; i < numPoints; i++) {

			//TODO: real part of the problem
			//TODO: imaginary part of the problem
		}
		
		return b;
	}
	
	/**
	 * Transformation method
	 * Computes the registered points by transforming the original points "points" using the parameters from "parameter"
	 * @param points point cloud to be registered
	 * @param parameter rigid transformation parameters
	 * @return transformedPoints registered point cloud
	 */	
	public SimpleMatrix applyTransform(SimpleMatrix points, RigidParameters parameter){
		
		SimpleMatrix r = new RotationMatrix2D(parameter.getAngle());
		SimpleMatrix transformedPoints = new SimpleMatrix(points.getRows(), points.getCols());
		
		// transform points (rotate and translate)
		for(int i = 0; i < transformedPoints.getRows(); i++){
			
			//TODO: rotate (you can use SimpleOperators)
			//TODO: translate (you can use SimpleOperators)
		}
		
		return transformedPoints;
	}
	
	/**
	 * 
	 * end of the exercise
	 */		
	
	public ExerciseREG(){
		
		// Define Point Cloud 
		p_k = new SimpleMatrix(4,2);
		q_k = new SimpleMatrix(4,2);
		
		p_k.setRowValue(0, new SimpleVector(1,2));
		p_k.setRowValue(1, new SimpleVector(3,6));
		p_k.setRowValue(2, new SimpleVector(4.5,5.5));
		p_k.setRowValue(3, new SimpleVector(3,3.5));
		
		q_k.setRowValue(0, new SimpleVector(-0.7, 2.1));
		q_k.setRowValue(1, new SimpleVector(-2.1, 6.4));
		q_k.setRowValue(2, new SimpleVector(-1.4, 7.1));
		q_k.setRowValue(3, new SimpleVector(-0.7, 4.9));
		
		// Plot Point Cloud
		plot = new Plot("Regression Line", "X", "Y", Plot.DEFAULT_FLAGS);
		plot.setLimits(-5, 5, 0, 10);
		plot.setSize(512, 512);
		plot.setScale(2.0f);
		plot.setLineWidth(1.5f);
		plot.addLegend("cloud {p_k}\ncloud {q_k}\nregistered {q_k}\nd");
		plot.addPoints(p_k.getCol(0).copyAsDoubleArray(), p_k.getCol(1).copyAsDoubleArray(), Plot.BOX);
		plot.setColor(Color.blue);
		plot.addPoints(q_k.getCol(0).copyAsDoubleArray(), q_k.getCol(1).copyAsDoubleArray(), Plot.CIRCLE);
		
	}
	
	public class RotationMatrix2D extends SimpleMatrix {
		
		private static final long serialVersionUID = -6939956675976343158L;
		
		public RotationMatrix2D() {
			this(0.0);
		}
		
		public RotationMatrix2D(double phi) {
			
			super(2,2);
			
			this.setElementValue(0, 0, Math.cos(phi));
			this.setElementValue(0, 1, - Math.sin(phi));
			this.setElementValue(1, 0, Math.sin(phi));
			this.setElementValue(1, 1, Math.cos(phi));
		}
	} 
	
	public class RigidParameters {
		
		private double phi;
		private SimpleVector translation;
		
		public RigidParameters(){
			
			phi = 0.0d;
			translation = new SimpleVector(2);
		}
		public RigidParameters(final double phi, final SimpleVector translation){
			
			this.phi = phi;
			this.translation = translation;
		}
		//
		public void setAngle(final double phi) {
			this.phi = phi;
		}
		public void setTranslation(final SimpleVector translation) {
			this.translation = translation;
		}
		//
		public double getAngle() {
			return phi;
		}
		public SimpleVector getTranslation() {
			return translation;
		}		
	}
	
	// getters for members
	// variables which are checked (DO NOT CHANGE!)
	public SimpleMatrix get_p_k() {
		return p_k;
	}
	//
	public SimpleMatrix get_q_k() {
		return p_k;
	}
	//
	public Plot get_plot() {
		return plot;
	}
}
