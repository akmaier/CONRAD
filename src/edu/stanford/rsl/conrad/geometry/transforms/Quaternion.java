package edu.stanford.rsl.conrad.geometry.transforms;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class Quaternion {
	private double scaler = 0;
	private SimpleVector vector;

	public Quaternion(double scaler, SimpleVector vector) {
		this.scaler = scaler;
		this.vector = vector;
	}
	
	public void sum(Quaternion q){
		scaler+=q.getScaler();
		vector.add(q.getVector());
	}
	
	public  Quaternion getSum(Quaternion q){
		double newscale = scaler+q.getScaler();
		SimpleVector v = SimpleOperators.add(vector,q.getVector());
		return new Quaternion(newscale, v);
	}
	
	public void multiplyBy(Quaternion q){
		double s1 = this.scaler;
		double s2 = q.getScaler();
		SimpleVector v2 = q.getVector();
		this.scaler = s1*s2 - SimpleOperators.multiplyInnerProd(vector, q.getVector());
		this.vector = SimpleOperators.add(v2.multipliedBy(s1),vector.multipliedBy(s2),General.crossProduct(vector, v2));
	}
	
	public Quaternion multipliedBy(Quaternion q){
		double s1 = this.scaler;
		double s2 = q.getScaler();
		SimpleVector v2 = q.getVector();
		double newscale = s1*s2 - SimpleOperators.multiplyInnerProd(vector, q.getVector());
		SimpleVector v = SimpleOperators.add(v2.multipliedBy(s1),vector.multipliedBy(s2),General.crossProduct(vector, v2));
		return new Quaternion(newscale, v);
	}
	
	public void invert(){
		double magnitude = magnitude();
		scaler/=magnitude;
		vector.negate();
		vector.multiplyBy(1/magnitude);
		
	}
	
	public Quaternion getInverse(){
		SimpleVector v = vector.negated();
		v.multiplyBy(1/magnitude());
		return new Quaternion(scaler/magnitude(), v);
	}
	
	public double magnitude(){
		double sqVecSum = 0;
		for(int i = 0; i < 3; i++){
			sqVecSum+= vector.getElement(i)*vector.getElement(i);
		}
		return Math.sqrt(scaler*scaler + sqVecSum);
	}
	
	public SimpleVector getVector() {
		return vector.clone();
	}

	public double getScaler() {
		return scaler;
	}

	public SimpleMatrix equivalentMatrix(){		
		double x = vector.getElement(0);
		double y = vector.getElement(1);
		double z = vector.getElement(2);
		
		double M11 = 1- 2*(y*y + z*z);
		double M12 = 2*(x*y - scaler*z);
		double M13 = 2*(x*z + scaler*y);
		double M21 = 2*(x*y + scaler*z);
		double M22 = 1- 2*(x*x + z*z);
		double M23 = 2*(y*z - scaler*x);
		double M31 = 2*(x*z - scaler*y);
		double M32 = 2*(y*z + scaler*x);
		double M33 = 1 - 2*(x*x + y*y);
		
		double [][] data = {{M11, M12, M13},{M21, M22, M23},{M31 ,M32, M33}};
		return new SimpleMatrix(data);
	}
	
	public Quaternion clone(){
		return new Quaternion(scaler, vector.clone());
	}
	
	public double distanceTo( Quaternion q ){
		Quaternion tmp = this.multipliedBy( q );
		return 2*Math.toDegrees( Math.acos( tmp.getScaler() ) );
	}
	
	public double length(){
		return Math.sqrt(scaler*scaler + vector.getElement(0)*vector.getElement(0) + vector.getElement(1)*vector.getElement(1) + vector.getElement(2)*vector.getElement(2));
	}
	
	/**
	 * Calculates a quaternion with a given rotation matrix.
	 * The direct way of creating linear equations from the relations given in function "equivalentMatrix()" is computationally intensive and can lead to "Gimbal lock".
	 * The solution can be determined in two steps:
	 * First, the maximum element of the quaternion has to be determined. The equation 
	 * 		R11 + R22 + R33 = (a*a + b*b - c*c - d*d) + (a*a - b*b + c*c - d*d) + (a*a - b*b - c*c + d*d)
	 * can be solved for every element of the quaternion, where "R11 + R22 + R33" is the trace of the rotation matrix.
	 * The maximum element is needed to avoid dividing by zero in the following steps. The result is not influenced by this choice. 
	 * Equations taken from: Prof. Dr.-Ing. Jörg Buchholz, 2010, Regelungstechnik und Flugregler, Munich, GRIN Verlag, http://www.grin.com/de/e-book/82818/regelungstechnik-und-flugregler
	 * @param R
	 * @return q
	 */
	public Quaternion equivalentQuaternion(SimpleMatrix R){
		Quaternion q;
		//find maximum element
		double scalarNew = Math.sqrt(R.getElement(1, 1) + R.getElement(2, 2) + R.getElement(3, 3) + 1) / 2;
		double vectorX = Math.sqrt(R.getElement(1, 1) - R.getElement(2, 2) - R.getElement(3, 3) + 1) / 2;
		double vectorY = Math.sqrt(-R.getElement(1, 1) + R.getElement(2, 2) - R.getElement(3, 3) + 1) / 2;
		double vectorZ = Math.sqrt(-R.getElement(1, 1) - R.getElement(2, 2) + R.getElement(3, 3) + 1) / 2;
		double maxEl = Math.max(Math.max(scalarNew, vectorX), Math.max(vectorY, vectorZ));
		//determine other elements with the found maximum
		if (maxEl == scalarNew){
			vectorX 	= (R.getElement(2, 3) - R.getElement(3, 2)) / (4 * maxEl);
			vectorY 	= (R.getElement(3, 1) - R.getElement(1, 3)) / (4 * maxEl);
			vectorZ 	= (R.getElement(1, 2) - R.getElement(2, 1)) / (4 * maxEl);
			SimpleVector vectorNew = new SimpleVector(vectorX, vectorY, vectorZ);
			q = new Quaternion(scalarNew, vectorNew);
		}else if (maxEl == vectorX){
			scalarNew 	= (R.getElement(2, 3) - R.getElement(3, 2)) / (4 * maxEl);
			vectorY 	= (R.getElement(1, 2) + R.getElement(2, 1)) / (4 * maxEl);
			vectorZ 	= (R.getElement(1, 3) + R.getElement(3, 1)) / (4 * maxEl);
			SimpleVector vectorNew = new SimpleVector(maxEl, vectorY, vectorZ);
			q = new Quaternion(scalarNew, vectorNew);
		}else if (maxEl == vectorY){
			scalarNew 	= (R.getElement(3, 1) - R.getElement(1, 3)) / (4 * maxEl);
			vectorX 	= (R.getElement(1, 2) + R.getElement(2, 1)) / (4 * maxEl);
			vectorZ 	= (R.getElement(2, 3) + R.getElement(3, 2)) / (4 * maxEl);
			SimpleVector vectorNew = new SimpleVector(vectorX, maxEl, vectorZ);
			q = new Quaternion(scalarNew, vectorNew);
		}else if (maxEl == vectorZ){
			scalarNew 	= (R.getElement(1, 2) - R.getElement(2, 1)) / (4 * maxEl);
			vectorX 	= (R.getElement(1, 3) + R.getElement(3, 1)) / (4 * maxEl);
			vectorY 	= (R.getElement(2, 3) + R.getElement(3, 2)) / (4 * maxEl);
			SimpleVector vectorNew = new SimpleVector(vectorX, vectorY, maxEl);
			q = new Quaternion(scalarNew, vectorNew);
		}else{
			q = new Quaternion(0, new SimpleVector());
		}
		return q;
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/