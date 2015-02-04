package edu.stanford.rsl.conrad.geometry.shapes.simple;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


/**
 * Creates a Sphere.
 * @author Rotimi X Ojo
 */
public class Sphere extends QuadricSurface {

	private static final long serialVersionUID = 7869742887570580304L;
	private Axis principalAxis = new Axis(new SimpleVector(1,0,0));
	
	public Sphere(){
	}	
	
	public Sphere(double radius, PointND surfaceOrigin){		
		init(radius, surfaceOrigin);
	}
	
	public Sphere(double radius){		
		init(radius, new PointND(0,0,0));
	}
	
	public Sphere(Sphere s){
		super(s);
		principalAxis = (s.principalAxis != null) ? s.principalAxis.clone() : null; 
	}
	
	protected void init(double radius, PointND surfaceOrigin){
		SimpleMatrix mat = new SimpleMatrix(3,3);
		mat.identity();
		transform = new AffineTransform(mat, surfaceOrigin.getAbstractVector());
		double cons = 1/(radius*radius);
		double constArray[][] = {{cons,0,0},{0,cons,0},{0,0,cons}};
		super.constMatrix = new SimpleMatrix(constArray);
		super.constant = 1;
		
		this.min = new PointND(surfaceOrigin.get(0) - radius, surfaceOrigin.get(1) - radius, surfaceOrigin.get(2) - radius);
		this.max = new PointND(surfaceOrigin.get(0) + radius, surfaceOrigin.get(1) + radius, surfaceOrigin.get(2) + radius);
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Axis getPrincipalAxis() {
		return principalAxis;
	}
	
	@Override
	public void applyTransform(Transform t) {
		super.applyTransform(t);
		//this.min = t.transform(this.min);
		//this.max = t.transform(this.max);
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		
		PointND center = (new PointND((min.get(0)+max.get(0))/2, (min.get(1)+max.get(1))/2,(min.get(2)+max.get(2))/2));
		double radius = (max.get(0)-min.get(0))/2;
		
		
		int pointsOnCurve = (int) (accuracy * 50); 
		int nextCurve = pointsOnCurve + 1;
		double angle;
		double angleIncrement = (2*Math.PI)/pointsOnCurve;
		int slices = (int) (accuracy*40);
		if (slices%2 != 0) slices++; //uneven number of slices to include equator curve
		double curveDistance = (radius*2)/(slices);
		
		PointND[] curve = new PointND[(pointsOnCurve+1)*(slices + 1)]; //last value of each slice's curve equals first value. Total number of curves is number of slices + 1.


		/*
		 * Create all layers
		 */
		CompoundShape 	shape = new CompoundShape();
		double scalingCurve;
		for (int z = 0; z <= slices; z++) {
			scalingCurve = Math.sqrt(Math.pow(radius, 2) - Math.pow((radius-z*curveDistance), 2));
				angle = 0;
				for (int i = 0+z*nextCurve; i < pointsOnCurve+z*nextCurve; i++) {
					curve[i] = (new PointND(center.get(0)+scalingCurve*Math.cos(angle), center.get(1)+scalingCurve*Math.sin(angle), center.get(2)+radius-z*curveDistance));
					angle += angleIncrement;
				}
				curve[pointsOnCurve+z*nextCurve] = curve[0+z*nextCurve];
		}
		for (int z = 0; z < slices; z++) {
			for (int i = 0+z*nextCurve; i < pointsOnCurve+z*nextCurve; i++) {
				if (i > pointsOnCurve) shape.add(new Triangle(curve[i], curve[i+nextCurve], curve[i+1]));
				if (i < pointsOnCurve+(slices-2)*nextCurve) shape.add(new Triangle(curve[i+nextCurve], curve[i+1+nextCurve], curve[i+1]));
			}
		}
		return shape;
	}
	public double getRadius(){
		return (max.get(0)-min.get(0))/2;
	}
	
	public PointND getCenter(){
		return new PointND((min.get(0)+max.get(0))/2, (min.get(1)+max.get(1))/2,(min.get(2)+max.get(2))/2);
	}
	
	@Override
	public float[] getRasterPoints(int elementCountU, int elementCountV) {

		if (elementCountU < 2){
			System.out.println("Error! valuesU has to be higher than 1");
			return null;
		}

		PointND center = (new PointND((min.get(0)+max.get(0))/2, (min.get(1)+max.get(1))/2,(min.get(2)+max.get(2))/2));
		double radius = (max.get(0)-min.get(0))/2;
		
		
		int pointsOnCurve = elementCountU; 
		double angle;
		double angleIncrement = (2*Math.PI)/(pointsOnCurve-1);
		int numberOfCurves = elementCountV;
		
		double curveDistance = (radius*2)/(numberOfCurves-1);
		
		float[] curve = new float[(pointsOnCurve)*(numberOfCurves)*3];

		
		double scalingCurve;
		for (int i = 0; i < pointsOnCurve; i++) {
				//for (int j = 0+i*nextCurve; j < pointsOnCurve+i*nextCurve; j++) {
				for (int j = 0; j < numberOfCurves; j++){
					angle = i * angleIncrement;
					scalingCurve = Math.sqrt(Math.pow(radius, 2) - Math.pow((radius-j*curveDistance), 2));
					curve[(j*pointsOnCurve+i)*3] = (float) (center.get(0)+scalingCurve*Math.cos(angle));
					curve[(j*pointsOnCurve+i)*3+1] = (float) (center.get(1)+scalingCurve*Math.sin(angle));
					curve[(j*pointsOnCurve+i)*3+2] = (float) (center.get(2)-radius+j*curveDistance);
				}
		}

		return curve;
	}

	@Override
	public AbstractShape clone() {
		return new Sphere(this);
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/