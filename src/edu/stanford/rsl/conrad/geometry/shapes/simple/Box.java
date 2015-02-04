package edu.stanford.rsl.conrad.geometry.shapes.simple;

import java.util.ArrayList;

import edu.stanford.rsl.apps.gui.opengl.PointCloudViewer;
import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.Axis;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * Creates a Box.
 */
public class Box extends SimpleSurface {


	public static final long serialVersionUID = 1760483762488429473L;
	private Axis principalAxis = new Axis(new SimpleVector(1,0,0));
	//protected PointND lowerCorner;
	public PointND lowerCorner;    // invisible in OpenCLBox, the suggestion is to change to 'public'
	//protected PointND upperCorner;
	public PointND upperCorner;
	
	public Box(){
		SimpleMatrix mat = new SimpleMatrix(3,3);	
		mat.identity();
		transform = new AffineTransform(mat, new SimpleVector(0,0,0));
	}
	
	/**
	 * Generates a box of size dx*dy*dz at between (0,0,0) and (dx,dy,dz)
	 * @param dx
	 * @param dy
	 * @param dz
	 */
	public Box(double dx, double dy, double dz){
		this();
		init((new PointND(0,0,0)),dx, dy, dz);
	}

	public Box(Box b){
		super(b);
		principalAxis = b.principalAxis!=null ? b.principalAxis.clone() : null;
		lowerCorner = b.lowerCorner != null ? b.lowerCorner.clone() : null;
		upperCorner = b.upperCorner != null ? b.upperCorner.clone() : null;
	}
	
	protected void init(PointND origin, double dx, double dy,	double dz) {
		min = origin;
		lowerCorner = min;
		max = new PointND(origin.get(0)+ dx,origin.get(1)+dy,origin.get(2)+dz);
		upperCorner = max;
		generateBoundingPlanes();		
	}
	
	@Override
	public boolean isMember(PointND point, boolean pointTransformed) {
		// TODO: Correct for new boundary conditions
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
	public PointND evaluate(PointND u) {
		return null;
	}

	@Override
	public int getDimension() {
		return 3;
	}

	@Override
	public int getInternalDimension() {
		return 0;
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		StraightLine buff = (StraightLine) other;
		StraightLine line = new StraightLine(buff.getPoint().clone(), buff.getDirection().clone());
		line.applyTransform(transform.inverse());		
		ArrayList<PointND> hitsOnShape = General.intersectRayWithCuboid(line, lowerCorner, upperCorner);
		return getCorrectedHits(hitsOnShape);	
	}

	@Override
	public boolean isBounded() {
		return true;
	}


	@Override
	public PointND[] getRasterPoints(int number) {
		PointND one = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		PointND two = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		PointND three = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		PointND four = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		PointND five = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		PointND six = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		PointND seven = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		PointND eight = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		int subNumber = number /12;
		Edge e01 = new Edge(one, two);
		Edge e02 = new Edge(one, three);
		Edge e03 = new Edge(one, five);
		Edge e04 = new Edge(two, six);
		Edge e05 = new Edge(two, four);
		Edge e06 = new Edge(three, four);
		Edge e07 = new Edge(seven, three);
		Edge e08 = new Edge(four, eight);
		Edge e09 = new Edge(five, six);
		Edge e10 = new Edge(five, seven);
		Edge e11 = new Edge(six, eight);
		Edge e12 = new Edge(seven, eight);
		PointND [] points01 = e01.getRasterPoints(subNumber);
		PointND [] points02 = e02.getRasterPoints(subNumber);
		PointND [] points03 = e03.getRasterPoints(subNumber);
		PointND [] points04 = e04.getRasterPoints(subNumber);
		PointND [] points05 = e05.getRasterPoints(subNumber);
		PointND [] points06 = e06.getRasterPoints(subNumber);
		PointND [] points07 = e07.getRasterPoints(subNumber);
		PointND [] points08 = e08.getRasterPoints(subNumber);
		PointND [] points09 = e09.getRasterPoints(subNumber);
		PointND [] points10 = e10.getRasterPoints(subNumber);
		PointND [] points11 = e11.getRasterPoints(subNumber);
		PointND [] points12 = e12.getRasterPoints(subNumber);
		PointND [] points = new PointND[12 * subNumber];
		int offset = 0;
		System.arraycopy(points01, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points02, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points03, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points04, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points05, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points06, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points07, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points08, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points09, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points10, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points11, 0, points, offset, subNumber);
		offset += subNumber;
		System.arraycopy(points12, 0, points, offset, subNumber);
		return points;
	}

	@Override
	public ArrayList<PointND> getHits(AbstractCurve other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Axis getPrincipalAxis() {
		return principalAxis ;
	}

	@Override
	public PointND evaluate(double u, double v) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AbstractShape tessellate(double accuracy) {
		PointND one = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		PointND two = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		PointND three = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		PointND four = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		PointND five = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		PointND six = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		PointND seven = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		PointND eight = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		CompoundShape shape = new CompoundShape();
		shape.add(new Triangle(one, two, four));
		shape.add(new Triangle(one, three, four));
		
		shape.add(new Triangle(one, two, six));
		shape.add(new Triangle(one, five, six));
		
		shape.add(new Triangle(two, four, eight));
		shape.add(new Triangle(two, six, eight));
		
		shape.add(new Triangle(one, three, seven));
		shape.add(new Triangle(one, five, seven));
		
		shape.add(new Triangle(six, five, seven));
		shape.add(new Triangle(six, seven, eight));
		
		shape.add(new Triangle(three, four, eight));
		shape.add(new Triangle(three, seven, eight));
		
		return shape;
	}


	public float[] getRasterPointsOld(int elementCountU, int elementCountV) {
		//if (elementCountU % 8 != 0 || elementCountV % 8 != 0){
		//	System.out.println("Error! valuesU and valuesV have to be a multiple of 4");
		//	return null;
		//}
		int pointsOnCurve = elementCountU;
		int numberOfCurves = elementCountV;
		int extraValuesPerEdge = (elementCountU-4)/4;
		
		PointND[] points = new PointND[8];
		
		points[0] = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		points[1] = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		points[2] = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		points[3] = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		
		points[4] = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		points[5] = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		points[6] = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		points[7] = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		
		float[] curve = new float[elementCountU*elementCountV*3];
		
	/*	for (int i = 0; i < pointsOnCurve; i++) {
			for (int j = 0; j < numberOfCurves; j++){
				if (i%(extraValuesPerEdge+1) == 0) { //regular corner point
					curve[(j*pointsOnCurve+i)*3] = (float) points[i%(extraValuesPerEdge)].get(0);
					curve[(j*pointsOnCurve+i)*3+1] = (float) points[i%4].get(1);
				}
				curve[(j*pointsOnCurve+i)*3] = (float) points[i%4].get(0);
				curve[(j*pointsOnCurve+i)*3+1] = (float) points[i%4].get(1);
				//curve[(j*pointsOnCurve+i)*3+2] = (float) (points[i%4].get(2)-points[];
				//curve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), max.get(2)));
				//lowerCurve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), min.get(2)));
				
			}
		} */
		int k = 0;
		
		for (int i = 0; i < pointsOnCurve; i++) {
			for (int j = 0; j < numberOfCurves; j++){
				k = (4*i)/pointsOnCurve;
				
				curve[(i*numberOfCurves+j)*3] = (float) points[k%4].get(0);
				curve[(i*numberOfCurves+j)*3+1] = (float) points[k%4].get(1);
				curve[(i*numberOfCurves+j)*3+2] = (float) (points[k%4].get(2) - (points[k%4].get(2)-points[k%4+4].get(2)) * ((float) j/(numberOfCurves-1)));
				
				//curve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), max.get(2)));
				//lowerCurve[i] = (new PointND(((min.get(0)+ max.get(0))/2)+0.5*(max.get(0)-min.get(0))*Math.cos(angle), ((min.get(1)+ max.get(1))/2)+0.5*(max.get(1)-min.get(1))*Math.sin(angle), min.get(2)));
			}
		}
		return curve;
	}
	
	public PointND interpolateU(double u, PointND [] points, int offset){
		PointND first;
		PointND second;
		if (u <= 0.25){
			first = points[0+offset];
			second = points[1+offset];
		} else if (u <= 0.5){
			u -= 0.25;
			first = points[1+offset];
			second = points[2+offset];
		} else if (u <= 0.75){
			u -= 0.5;
			first = points[2+offset];
			second = points[3+offset];
		} else {
			u -= 0.75;
			first = points[3+offset];
			second = points[0+offset];
		}
		return interpolate(first, second, u*4.0);
	}
	
	public PointND interpolate(PointND first, PointND second, double factor){
		SimpleVector resultVector = first.getAbstractVector().multipliedBy(1.0-factor);
		resultVector.add(second.getAbstractVector().multipliedBy(factor)); 
		return new PointND(resultVector);
	}
	
	@Override
	public float[] getRasterPoints(int elementCountU, int elementCountV) {
		
		PointND[] points = new PointND[8];
		
		points[0] = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		points[1] = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), upperCorner.get(2)));
		points[2] = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		points[3] = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), upperCorner.get(2)));
		
		points[4] = transform.transform(new PointND(lowerCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		points[5] = transform.transform(new PointND(upperCorner.get(0), lowerCorner.get(1), lowerCorner.get(2)));
		points[6] = transform.transform(new PointND(upperCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		points[7] = transform.transform(new PointND(lowerCorner.get(0), upperCorner.get(1), lowerCorner.get(2)));
		
		float[] curve = new float[elementCountU*elementCountV*3];
		
		
		for (int j = 0; j < elementCountV; j++){
			double v = ((double)j)/(elementCountV-1);
			for (int i = 0; i < elementCountU; i++) {
				double u = ((double)i)/(elementCountU);
				
				PointND top = interpolateU(u, points, 0);
				PointND bottom = interpolateU(u, points, 4);
				PointND gridPoint = interpolate(bottom, top, v);
				
				curve[(j*elementCountU+i)*3] = (float) gridPoint.get(0);
				curve[(j*elementCountU+i)*3+1] = (float) gridPoint.get(1);
				curve[(j*elementCountU+i)*3+2] = (float) gridPoint.get(2);
			}
		} 
		return curve;
	}
	
	/*public float[] getRasterPointsNew(int elementCountU, int elementCountV){
		
		int pointsOnCurve = elementCountU;
		int numberOfCurves = elementCountV;
		int extraValuesPerEdge = (elementCountU-4)/4;
		float[] curve = new float[elementCountU*elementCountV*3];
		
		for (int j = 0; j < numberOfCurves; j++) {
			for (int i = 0; i < pointsOnCurve; i++){
				for (int k = 0; k < (int)pointsOnCurve/4; k++){ 
				curve[(j*pointsOnCurve+i+k)*3] = (float) (lowerCorner.get(0) + i*(upperCorner.get(0)-lowerCorner.get(0))/(pointsOnCurve-1));
				}
				curve[(j*pointsOnCurve+i+k)*3+1] = (float) (lowerCorner.get(1) + i*(upperCorner.get(1)-lowerCorner.get(1))/(pointsOnCurve-1));
				curve[(j*pointsOnCurve+i+k)*3+2] = (float) (lowerCorner.get(2) + j*(upperCorner.get(2)-lowerCorner.get(2))/(numberOfCurves-1));
			}
		}
		return curve;
	}*/

	@Override
	public AbstractShape clone() {
		return new Box(this);
	}

	/**
	 * @return the lowerCorner
	 */
	public PointND getLowerCorner() {
		return lowerCorner;
	}

	/**
	 * @param lowerCorner the lowerCorner to set
	 */
	public void setLowerCorner(PointND lowerCorner) {
		this.lowerCorner = lowerCorner;
	}

	/**
	 * @return the upperCorner
	 */
	public PointND getUpperCorner() {
		return upperCorner;
	}

	/**
	 * @param upperCorner the upperCorner to set
	 */
	public void setUpperCorner(PointND upperCorner) {
		this.upperCorner = upperCorner;
	}

	/**
	 * @param principalAxis the principalAxis to set
	 */
	public void setPrincipalAxis(Axis principalAxis) {
		this.principalAxis = principalAxis;
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
	
/*	public static void main(String [] args){
		Box box = new Box (1,1,1);
		
		ArrayList<PointND> pointsList=box.getPointCloud(40, 10);
		
	    PointCloudViewer pcv = new PointCloudViewer("Box Visualization", pointsList);
		pcv.setVisible(true);
	}*/

}

/*
 * Copyright (C) 2010-2014 Andreas Maier, Rotimi X Ojo, Zijia Guo 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

