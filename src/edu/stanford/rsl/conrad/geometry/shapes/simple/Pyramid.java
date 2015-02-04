package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Creates a Pyramid
 */
public class Pyramid extends SimpleSurface {


	/**
	 * 
	 */
	private static final long serialVersionUID = -3684428666704786034L;
	private Axis principalAxis = new Axis(new SimpleVector(0,0,1));
	
	public Pyramid() {
		// TODO Auto-generated constructor stub
	}
	
	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public Pyramid(double dx, double dy, double dz) {
		this();
		init(dx, dy, dz, null);
	}
	
	/**
	 * @param dx
	 * @param dy
	 * @param dz
	 * @param transform
	 */
	public Pyramid(double dx, double dy, double dz,AffineTransform transform){
		this();
		init(dx, dy, dz, transform);
	}

	/**
	 * @param p
	 */
	public Pyramid(Pyramid p) {
		super(p);
		// TODO Auto-generated constructor stub
		principalAxis = (p.principalAxis != null) ? p.principalAxis.clone() : null; 
	}
	
	public void init(double dx, double dy, double dz, AffineTransform transform){
		if(transform == null){
    		SimpleMatrix mat = new SimpleMatrix(3,3);
    		mat.identity();
    		this.transform = new AffineTransform(mat, new SimpleVector(3));
    	}
		this.transform = new Translation(0, 0, 0);
		this.min = new PointND(-dx, -dy, -dz);
		//this.min = this.transform.transform(this.min);
		this.max = new PointND(dx, dy, 0);
		//this.max = this.transform.transform(this.max); 
	}

	@Override
	public boolean isBounded() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public Axis getPrincipalAxis() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape clone() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public PointND interpolatePoint(double factor, PointND[] point){
		
		PointND first;
		PointND second;
		if (factor < 0.25){
			first = point[0];
			second = point[1];
		}else if (factor < 0.5){
			first = point[1];
			second = point[2];
			factor -= 0.25;
		}else if (factor < 0.75){
			first = point[2];
			second = point[3];
			factor -= 0.5;
		}else {
			first = point[3];
			second = point[0];
			factor -= 0.75;
		}
		return interpolation(first, second, 4.0*factor);
	}
	
	public PointND interpolation(PointND first, PointND second, double factor){
		SimpleVector resultVector = first.getAbstractVector().multipliedBy(1.0 - factor);
		resultVector.add(second.getAbstractVector().multipliedBy(factor)); 
		return new PointND(resultVector);
	}
	
	public float[] getRasterPoints(int elementCountU, int elementCountV){
		if (elementCountV < 2 || elementCountV < 3){
			System.out.println("Error! elementCounts too small.");
			return null;
		}
		
		PointND[] point = new PointND[4];

		
		int numberOfCurves = elementCountU;
		int pointsOnCurve = elementCountV;
		float[] curve = new float[numberOfCurves*pointsOnCurve*3];
		
		double range = (max.get(2) - min.get(2));
		double a = 0.5*(max.get(0) - min.get(0))/ range;
		double b = 0.5*(max.get(1) - min.get(1))/ range;
		double height;
		
		for (int j = 0; j < numberOfCurves; j++){
			//double factor = ((double)j/(numberOfCurves - 1));
			height = j*(max.get(2) - min.get(2))/(numberOfCurves - 1);
		
			//double coorZ = (double)min.get(2) + factor;

			double coorZ = (min.get(2) + height)/range;
			
			for (int i = 0; i < pointsOnCurve; i++){
				double v = ((double)i/(pointsOnCurve));
				
				point[0] = transform.transform(new PointND(min.get(0)*a*Math.abs(coorZ), min.get(1)*b*Math.abs(coorZ), coorZ*range));
				point[1] = transform.transform(new PointND(max.get(0)*a*Math.abs(coorZ), min.get(1)*b*Math.abs(coorZ), coorZ*range));
				point[2] = transform.transform(new PointND(max.get(0)*a*Math.abs(coorZ), max.get(1)*b*Math.abs(coorZ), coorZ*range));
				point[3] = transform.transform(new PointND(min.get(0)*a*Math.abs(coorZ), max.get(1)*b*Math.abs(coorZ), coorZ*range));
				
				PointND rasterPointV = interpolatePoint(v, point);
				
				curve[(i*numberOfCurves+j)*3] = (float)rasterPointV.get(0);
				curve[(i*numberOfCurves+j)*3+1] = (float)rasterPointV.get(1);
				curve[(i*numberOfCurves+j)*3+2] = (float)(coorZ*range);
			}
		}
		
		return curve;
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
		Pyramid pyramid = new Pyramid(10,10,10);
	    if (pyramid != null) {
		
		float[] rasterPoints = pyramid.getRasterPoints(3, 16);
		
	    ArrayList<PointND> pointsList= new ArrayList<PointND>();
	    

		
		    for (int i= 0; i < rasterPoints.length/3; i++){
	
				PointND point = new PointND(rasterPoints[(i*3)],rasterPoints[(i*3)+1],rasterPoints[(i*3)+2]);
				//	PointND point = new PointND(points[i].get(0),points[i].get(1),points[i].get(2));
	
		    		pointsList.add(point);
		    	
			}
			PointCloudViewer pcv = new PointCloudViewer("Pyramid Visualization", pointsList);
			pcv.setVisible(true);
	    }
	}

	@Override
	public ArrayList<PointND> getHits(AbstractCurve other) {
		// TODO Auto-generated method stub
		//throw new RuntimeException("Not implemented yet");
		//return null;
		StraightLine buff = (StraightLine) other;
		StraightLine line = new StraightLine(buff.getPoint().clone(), buff.getDirection().clone());
		line.applyTransform(transform.inverse());		
		ArrayList<PointND> hitsOnShape = General.intersectRayWithCuboid(line, min, max);
		return getCorrectedHits(hitsOnShape);
	}

	@Override
	public boolean isMember(PointND point, boolean pointTransformed) {
		// TODO Auto-generated method stub
		//throw new RuntimeException("Not implemented yet");
		//return false;
		// Now, only the bounding box is checked.
		for (int j=0; j < point.getDimension(); j++){
			double coord = point.get(j);
			if (coord > max.get(j) + CONRAD.SMALL_VALUE || coord < min.get(j) - CONRAD.SMALL_VALUE) {
				return false;
			}
		}
		return true;
	}

	@Override
	public PointND evaluate(double u, double v) {
		// TODO Auto-generated method stub
		throw new RuntimeException("Not implemented yet");
		//return null;
	}

	@Override
	public int getDimension() {
		return 3;
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		// TODO Auto-generated method stub
		//getRasterPoints((int)Math.sqrt(number), (int)Math.sqrt(number));
		throw new RuntimeException("Not implemented yet");
	}
	
	
}

/*
 * Copyright (C) 2010-2014 Zijia Guo, Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
