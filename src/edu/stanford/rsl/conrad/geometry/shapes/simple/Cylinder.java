package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Creates a Cylinder.
 * @author Rotimi X Ojo
 */
public class Cylinder extends QuadricSurface {

	private static final long serialVersionUID = 6986546973890638795L;
	private Axis principalAxis = new Axis(new SimpleVector(0,0,1));
	public double dz, dx, dy;

	public Cylinder(){
	}

	/**
	 * Creates a new cylinder around (0,0,0) with radii dx, dy, and height dz.
	 * @param dx radius in x direction
	 * @param dy radius in y direction
	 * @param dz height
	 */
	public Cylinder(double dx, double dy, double dz){
		init(dx, dy, dz, null);
	}

	public Cylinder(Cylinder fc) {
		super(fc);
		principalAxis = (fc.principalAxis != null) ? fc.principalAxis.clone() : null;
		dx = fc.dx;
		dy = fc.dy;
		dz = fc.dz;
	}

	protected void init(double dx, double dy, double dz,Transform transform){
		if(transform == null){
			SimpleMatrix mat = new SimpleMatrix(3,3);
			mat.identity();
			transform = new AffineTransform(mat, new SimpleVector(3));
		}
		this.dz = dz;
		this.dx = dx;
		this.dy = dy;
		this.transform = transform;
		Plane3D top = new Plane3D(new PointND(0,0,dz/2), new SimpleVector(0,0,-1));
		Plane3D but = new Plane3D(new PointND(0,0,-dz/2), new SimpleVector(0,0,1));	
		addBoundingCondition(new HalfSpaceBoundingCondition(top));
		addBoundingCondition(new HalfSpaceBoundingCondition(but));
		double constArray[][] = {{1/(dx*dx),0,0},{0,1/(dy*dy),0},{0,0,0}};
		super.constMatrix = new SimpleMatrix(constArray);
		super.constant = 1;

		this.min = new PointND(-dx, -dy, -dz/2);
		//this.min = this.transform.transform(this.min);

		this.max = new PointND(dx, dy, dz/2);
		//this.max = this.transform.transform(this.max);
	}

	@Override
	public ArrayList<PointND> getHits (AbstractCurve other){
		ArrayList<PointND> results = super.getHits(other);
		SimpleVector dir = new SimpleVector(0, 0 ,1);
		StraightLine line = (StraightLine)other;
		SimpleVector lineOrigin = line.getPoint().getAbstractVector();
		SimpleVector direction = line.getDirection();
		//if (results.size() == 0){
			if (General.areColinear(direction, dir, CONRAD.FLOAT_EPSILON)){
				// This is done redundantly --> Just add a dummy variable that is out of bounds! 
				double first = Math.pow(lineOrigin.getElement(0) / dx,2);
				double second = Math.pow(lineOrigin.getElement(1) / dy,2);
				if (first + second < 1){
					results.clear();
					//results.add(new PointND(lineOrigin.getElement(0), lineOrigin.getElement(1), -dz/2));
					//results.add(new PointND(lineOrigin.getElement(0), lineOrigin.getElement(1), dz/2));
					results.add(new PointND(0, 0, dz));
				}
			}
		//}

		return results;
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public int getDimension() {
		return 3;
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
		PointND centerTop = (new PointND((min.get(0)+max.get(0))/2, (min.get(1)+max.get(1))/2, max.get(2)));
		PointND centerBottom = (new PointND((min.get(0)+max.get(0))/2, (min.get(1)+max.get(1))/2, min.get(2)));		
		
		int pointsOnCurve = (int) (accuracy * 60); 
		PointND[] upperCurve = new PointND[pointsOnCurve+1]; //last value equals first value
		PointND[] lowerCurve = new PointND[pointsOnCurve+1]; //last value equals first value
		
		double angleIncrement = (2*Math.PI)/pointsOnCurve;
		double angle;
		
		for (int i = 0; i < pointsOnCurve; i++) {
			angle = i*angleIncrement;
			upperCurve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), max.get(2)));
			lowerCurve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), min.get(2)));	
		}
		upperCurve[pointsOnCurve] = upperCurve[0];
		lowerCurve[pointsOnCurve] = lowerCurve[0];
		

		CompoundShape 	shape = new CompoundShape();
		for (int i = 0; i < pointsOnCurve; i++) {
			/*
			 * Create cylinder top and bottom plane
			 */
			shape.add(new Triangle(centerTop, upperCurve[i], upperCurve[i+1]));
			shape.add(new Triangle(centerBottom, lowerCurve[i+1], lowerCurve[i]));
			
			/*
			 * Create lateral surface
			 */
			shape.add(new Triangle(upperCurve[i], lowerCurve[i], upperCurve[i+1]));
			shape.add(new Triangle(lowerCurve[i], upperCurve[i+1], lowerCurve[i+1]));
		}

		
		return shape;
	}

	@Override
	public float[] getRasterPoints(int elementCountU, int elementCountV) {

		if (elementCountV < 2 || elementCountV < 3){
			System.out.println("Error! elementCounts too small.");
			return null;
		}
		
		int numberOfCurves = elementCountU;
		int pointsOnCurve = elementCountV;
		float[] curve = new float[numberOfCurves*pointsOnCurve*3]; 
		
		double angleIncrement = (2*Math.PI)/(pointsOnCurve-1);
		double angle;
		
		for (int j = 0; j < numberOfCurves; j++){
			for (int i = 0; i < pointsOnCurve; i++) {
				angle = i*angleIncrement;
				curve[(i*numberOfCurves+j)*3] = (float) (((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle));
				curve[(i*numberOfCurves+j)*3+1] = (float) (((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle));
				curve[(i*numberOfCurves+j)*3+2] = (float) (max.get(2)-   (max.get(2)-min.get(2)) *  ((float) j/(numberOfCurves-1)) );
				//curve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), max.get(2)));
				//lowerCurve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), min.get(2)));
			}
		}
		
		return curve;
	}

	@Override
	public AbstractShape clone() {
		return new Cylinder(this);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/