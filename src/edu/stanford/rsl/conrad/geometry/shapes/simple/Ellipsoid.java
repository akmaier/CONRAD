package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * Creates an Ellipsoid
 */
public class Ellipsoid extends QuadricSurface {
	
	public static final long serialVersionUID = 7869742887570580304L;
	private Axis principalAxis = new Axis(new SimpleVector(1,0,0));
	public double dz, dx, dy;
	
	public Ellipsoid(){
		
	}
	
	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public Ellipsoid(double dx, double dy, double dz){
		init(dx, dy, dz, null);
	}
	
	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 * @param transform
	 */
	public Ellipsoid(double dx, double dy, double dz,Transform transform){
		init(dx, dy, dz, transform);
	}
	
	/**
	 * @param e
	 */
	public Ellipsoid(Ellipsoid e){
		super(e);
		principalAxis = (e.principalAxis != null) ? e.principalAxis.clone() : null; 
		dx = e.dx;
		dy = e.dy;
		dz = e.dz;
	}

    protected void init(double dx, double dy, double dz,Transform transform){
    	if(transform == null){
    		SimpleMatrix mat = new SimpleMatrix(3,3);
    		mat.identity();
    		transform = new AffineTransform(mat, new SimpleVector(3));
    	}
    	this.dx = dx;
    	this.dy = dy;
    	this.dz = dz;
    	this.transform = transform;
		double constArray[][] = {{1/(dx*dx),0,0},{0,1/(dy*dy),0},{0,0,1/(dz*dz)}};
		super.constMatrix = new SimpleMatrix(constArray);
		super.constant = 1;
		
		this.min = new PointND(-dx, -dy, -dz);
		this.min = this.transform.transform(this.min);
		
		this.max = new PointND(dx, dy, dz);
		this.max = this.transform.transform(this.max);
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
	
	public float[] getRasterPoints(int elementCountU, int elementCountV) {

		if (elementCountV < 2 || elementCountV < 3){
			System.out.println("Error! elementCounts too small.");
			return null;
		}
		
		int numberOfCurves = elementCountU;
		int pointsOnCurve = elementCountV;
		float[] curve = new float[numberOfCurves*pointsOnCurve*3]; 
		
		double angleIncrement1 = (2*Math.PI)/(pointsOnCurve);
		double angleIncrement2 = Math.PI/(numberOfCurves-1);
		double angle1;
		double angle2;
		
		for (int j = 0; j < numberOfCurves; j++){
			for (int i = 0; i < pointsOnCurve; i++) {
				angle1 = i*angleIncrement1;
				angle2 = j*angleIncrement2;
				
				curve[(i*numberOfCurves+j)*3] = (float) (((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.sin(angle2)*Math.cos(angle1));
				curve[(i*numberOfCurves+j)*3+1] = (float) (((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle2)*Math.sin(angle1));
				curve[(i*numberOfCurves+j)*3+2] = (float) (((min.get(2)+ max.get(2))/2)+0.5*(max.get(2)-min.get(2))*Math.cos(angle2));
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
		return new Ellipsoid(this);
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
		Ellipsoid ellipsoid = new Ellipsoid (1,2,2);
		
		float[] rasterPoints = ellipsoid.getRasterPoints(4,4 );
		//PointND[] points = ellipsoid.getRasterPoints(256);
		
		ArrayList<PointND> pointsList= new ArrayList<PointND>();
		
		for (int i= 0; i < rasterPoints.length/3; i++){
			PointND point = new PointND(rasterPoints[(i*3)],rasterPoints[(i*3)+1],rasterPoints[(i*3)+2]);
			pointsList.add(point);
		}
		PointCloudViewer pcv = new PointCloudViewer("Ellipsoid Visualization", pointsList);
		pcv.setVisible(true);
	}
}
/*
 * Copyright (C) 2010-2014 Zijia Guo, Andreas Maier, Rotimi X Ojo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/