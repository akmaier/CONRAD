package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.bounds.HalfSpaceBoundingCondition;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Creates a Cone.
 */
public class Cone extends QuadricSurface {

	private static final long serialVersionUID = 6986546973890638795L;
	private Axis principalAxis = new Axis(new SimpleVector(0,0,1));
	
	public Cone() {
	}
	
	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public Cone(double dx, double dy, double dz){
		init(dx, dy, dz, null);
	}
	
	/**
	 * @param c
	 */
	public Cone(Cone c){
		super(c);
		principalAxis = (c.principalAxis != null) ? c.principalAxis.clone() : null; 
		min = c.min != null ? c.min.clone() : null;
		max = c.max != null ? c.max.clone() : null;
	}
	
	protected void init(double dx, double dy, double dz,Transform transform) {
		if(transform == null){
    		SimpleMatrix mat = new SimpleMatrix(3,3);
    		mat.identity();
    		transform = new AffineTransform(mat, new SimpleVector(3));
    	}
    	this.transform = transform;
		Plane3D topPlane = new Plane3D(new PointND(0,0,0), new SimpleVector(0,0,-1));
		Plane3D bottomPlane = new Plane3D(new PointND(0,0,-dz), new SimpleVector(0,0,1));
		addBoundingCondition(new HalfSpaceBoundingCondition(topPlane));
		addBoundingCondition(new HalfSpaceBoundingCondition(bottomPlane));
		
		double constArray[][] = {{dz/dx,0,0},{0,dz/dy,0},{0,0,-1}};
		super.constMatrix = new SimpleMatrix(constArray);
		super.constant = 0;
		
		this.min = new PointND(-dx, -dy, -dz);
		this.min = this.transform.transform(this.min);
		this.max = new PointND(dx, dy, 0);
		this.max = this.transform.transform(this.max);
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	@Override
	public int getDimension() {
		throw new UnsupportedOperationException();
	}

	public double interpolate(double factor){      // how to add some weight to make it non-uniform distribution along z axis???
		double first = min.get(2);
		double second = max.get(2);
		
		return (first*(1.0 - factor) + second*factor);
	}
	
	public float[] getRasterPoints(int elementCountU, int elementCountV) {

		if (elementCountV < 2 || elementCountV < 3){
			System.out.println("Error! elementCounts too small.");
			return null;
		}
		
		int numberOfCurves = elementCountU;
		int pointsOnCurve = elementCountV;
		float[] curve = new float[numberOfCurves*pointsOnCurve*3]; 
		
		double a = 0.5*(max.get(0) - min.get(0))/(max.get(2) - min.get(2));
		double b = 0.5*(max.get(1) - min.get(1))/(max.get(2) - min.get(2));
		
		double angleIncrement = (2*Math.PI)/(pointsOnCurve);
		double angle;
		double height;
		double factor;
		
		for (int j = 0; j < numberOfCurves; j++){
			for (int i = 0; i < pointsOnCurve; i++) {
				angle = i*angleIncrement;
				height = (double)(j*(max.get(2) - min.get(2))/(numberOfCurves - 1));
				factor = (double)(j/(numberOfCurves -1));
				
				float coorZ = (float) (min.get(2) + height);
				//float coorZ = (float) (interpolate(height));  // ?? the input of interpolate() should be factor, but error. why???
				
				curve[(i*numberOfCurves+j)*3] = (float) (((min.get(0) + max.get(0))/2) + a*coorZ*Math.cos(angle));
				curve[(i*numberOfCurves+j)*3+1] = (float) (((min.get(1) + max.get(1))/2) + b*coorZ*Math.sin(angle));
				curve[(i*numberOfCurves+j)*3+2] = coorZ;
			}
		}
		
		return curve;
	}
	
	@Override
	public Axis getPrincipalAxis() {
		return principalAxis;
	}

	@Override
	public void applyTransform(Transform t) {
		super.applyTransform(t);
		this.min = t.transform(this.min);
		this.max = t.transform(this.max);
	}
	@Override
	public AbstractShape tessellate(double accuracy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape clone() {
		return new Cone(this);
	}
	
	public ArrayList<PointND> getPointCloud(int u, int v){
		float[] rasterPoints = this.getRasterPoints(u, v);
		//PointND[] points = box.getRasterPoints(256);
		
	    ArrayList<PointND> pointsList= new ArrayList<PointND>();
		
	    for (int i= 0; i < rasterPoints.length/3; i++){

			PointND point = new PointND(rasterPoints[(i*3)],rasterPoints[(i*3)+1],rasterPoints[(i*3)+2]);
			//	PointND point = new PointND(points[i].get(0),points[i].get(1),points[i].get(2));

	    		pointsList.add(point);
	    	
		}
	    return pointsList;
	}
	
	public static void main(String [] args){
		Cone cone = new Cone (10,10,10);
		
		int u = 2;
		int v = 10;
		
		ArrayList<PointND> pointsList = new ArrayList<PointND>();
		pointsList = cone.getPointCloud(u, v);
		
		PointCloudViewer pcv = new PointCloudViewer("Cone Visualization", pointsList);
		pcv.setVisible(true);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo, Zijia Guo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/