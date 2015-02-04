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
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/