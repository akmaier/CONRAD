package edu.stanford.rsl.conrad.geometry.splines;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class BSpline extends AbstractCurve {
	
	boolean IS_CLAMPED = true;
	private double lowerBound = 0;
	private double delta = 1;

	/**
	 * 
	 */
	public static final float BSPLINECOLLECTION = 0;
	public static final float SPLINE1DTO3D = 1;
	public static final float SPLINE2DTO3D = 2;
	public static final float SPLINE3DTO3D = 3;
	private static Map<String, Float> idMap = Collections.synchronizedMap(new HashMap<String, Float>());
	
	protected static final long serialVersionUID = -1510774879340694844L;
	
	protected int degree;
	private double [] knots = null;
	private ArrayList<PointND> controlPoints;
	protected int dimension;
	
	public BSpline (ArrayList<PointND> controlPoints, double ... uVector){
		this.setControlPoints(controlPoints);
		setKnots(uVector);
		degree = ((getKnots().length - controlPoints.size()));
		checkIfClamped();
	}
	
	public BSpline (ArrayList<PointND> controlPoints, int degree, double ... uVector){
		this.setControlPoints(controlPoints);
		setKnots(uVector);
		this.degree = degree;
		checkIfClamped();
	}
	
	public BSpline(BSpline bs){
		super(bs);
		degree = bs.degree;
		dimension = bs.dimension;
		if (bs.knots != null){
			knots = new double[bs.knots.length];
			System.arraycopy(bs.knots, 0, knots, 0, bs.knots.length);
		}
		else
			knots = null;
		
		if (bs.controlPoints != null){
			Iterator<PointND> it = bs.controlPoints.iterator();
			controlPoints = new ArrayList<PointND>();
			while (it.hasNext()) {
				PointND p = it.next();
				controlPoints.add((p!=null) ? p.clone() : null);
			}
		}
		else{
			controlPoints = null;
		}
		checkIfClamped();
	}
	
	private void checkIfClamped(){
		for(int i = 0; i < degree; i++){
			if( (knots[i] != 0) && (knots[knots.length-1-i] != 1) ){
				this.IS_CLAMPED = false;
				lowerBound =   knots[degree];
				delta = (knots[knots.length-degree]-lowerBound);
				break;
			}				
		}
	}
	
	/**
	Non-recursive implementation of the N-function.
	*/
	protected double N(double internalCoordinate, int index) {
		//double [] knot = getKnots();
		double d = 0;
		int [] coefficientArrayA = new int[2 * degree];
		int [] coefficientArrayC = new int[2 * degree];
		int degreeMinus2 = degree - 2;
		for (int j = 0; j < degree; j++) {
			double firstKnot = knots[index+j];
			double secondKnot = knots[index+j+1];
			if (internalCoordinate >= firstKnot && internalCoordinate <= secondKnot && firstKnot != secondKnot) {
				// set required coefficients to 0
				for (int k = degree - j - 1; k >= 0; k--){
					coefficientArrayA[k] = 0;
				}
				if (j > 0) {
					// set required coefficients to their index
					for (int k = 0; k < j; k++){
						coefficientArrayC[k] = k;
					}
					coefficientArrayC[j] = Integer.MAX_VALUE;
				}
				else {
					coefficientArrayC[0] = degreeMinus2;
					coefficientArrayC[1] = degree;
				}
				int z = 0;
				while (true) {
					if (coefficientArrayC[z] < coefficientArrayC[z+1] - 1) {
						double e = 1.0;
						int bCounter = 0;
						int y = degreeMinus2 - j;
						int p = j - 1;

						for (int m = degreeMinus2, n = degree; m >= 0; m--, n--) {
							if (p >= 0 && coefficientArrayC[p] == m) {
								int w = index + bCounter;
								double kd = knots[w+n];
								e *= (kd - internalCoordinate) / (kd - knots[w+1]);
								bCounter++;
								p--;
							}
							else {
								int w = index + coefficientArrayA[y];
								double kw = knots[w];
								e *= (internalCoordinate - kw) / (knots[w+n-1] - kw);
								y--;
							}
						}
						// this code updates the a-counter
						if (j > 0) {
							int g = 0;
							boolean reset = false;

							while (true) {
								coefficientArrayA[g]++;

								if (coefficientArrayA[g] > j) {
									g++;
									reset = true;
								}
								else {
									if (reset) {
										for (int h = g - 1; h >= 0; h--)
											coefficientArrayA[h] = coefficientArrayA[g];
									}
									break;
								}
							}
						}

						d += e;

						// this code updates the bit-counter
						coefficientArrayC[z]++;
						if (coefficientArrayC[z] > degreeMinus2) break;

						for (int k = 0; k < z; k++)
							coefficientArrayC[k] = k;
						z = 0;
					}
					else {
						z++;
					}
				}
				break; // required to prevent spikes
			}
		}

		return d;
	}
	
	public double [] evalFast(double t) {
		if(!IS_CLAMPED){
			t = lowerBound + t*delta;
		}
		double [] p = new double [getControlPoints().get(0).getDimension()];
		int context = degree+1;
		int index = Arrays.binarySearch(getKnots(), t);
		int numPts = getControlPoints().size();
		if (index < 0) index *= -1; 
		int start = index - context;
		int end = index + context;
		if (start < 0) start = 0;
		if (end > numPts) end = numPts;
		//System.out.print("t = " + t +" ");
		for (int i = start; i < end; i++) {
			double w = N(t, i);
			//double w = N(t, i, degree);
			double[] loc = getControlPoints().get(i).getCoordinates();
			for (int j = 0; j < loc.length; j++){
				p[j] += (loc[j] * w); //pt[i][j] * w);
				//if (j==0) System.out.print("weight " + w + " ");
			}
		}
		//System.out.println(p [0] + " " + p[1]);
		return p;
	}
	
	/**
	 * Constructor for a BSpline using ArbitraryPoints and a weight vector as SimpleVector
	 * @param controlPoints the control points
	 * @param knotVector the weight vector
	 */
	public BSpline (ArrayList<PointND> controlPoints, SimpleVector knotVector){
		this(controlPoints, new PointND(knotVector).getCoordinates());
	}
	
	@Override
	public PointND evaluate(double u) {
		double [] point = evalFast(u);
		return new PointND(point);
	}
	
	

	@Override
	public int getDimension() {
		return dimension;
	}
	
	/**
	 * Computes N iteratively.
	 * @param u the point on the curve
	 * @param i the control point
	 * @return the weight
	 */
	public double getWeight(double u, int i){
		return N(u, i);
	}

	/**
	 * Computes the bounding box for the curve.
	 */
	public void computeBounds(){
		min = getControlPoints().get(0);
		max = getControlPoints().get(0);
		for (int i = 1; i < getControlPoints().size(); i++){
			min.updateIfLower(getControlPoints().get(i));
			max.updateIfHigher(getControlPoints().get(i));
		}
	}
	
	/**
	 * Returns the i-th control point
	 * @param i the index
	 * @return the control point
	 */
	public PointND getControlPoint(int i){
		return getControlPoints().get(i);
	}
	
	/**
	 * Returns the i-th knot vector entry.
	 * 
	 * @param i
	 * @return the knot vector entry
	 */
	public double getKnotVectorEntry(int i){
		return getKnots()[i];
	}
	
	/**
	 * Returns the degree of the spline.
	 * @return the degree
	 */
	public int getDegree(){
		return degree;
	}
	
	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		throw new RuntimeException("intersection between BSplines and other curves are not yet implemented");
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	public double distance(double [] planeCoefficients, double u) {
		return distanceFast(planeCoefficients, u);
	}
	
	public double distance(double [] planeCoefficients, int i) {
		return distanceFast(planeCoefficients, i);
	}
	
	public double distanceFast(double[] plane, double u) {
		int numPts = getControlPoints().size();
		//System.out.println("GroupSize: " + numPts);
		int context = degree + 2;
		int index = Arrays.binarySearch(getKnots(), u);
		if (index < 0) index *= -1; 
		//System.out.println(gi.getGroupSize() + " " + index + " " + t);
		int start = index - context;
		int end = index + context;
		if (start < 0) start = 0;
		if (end > numPts) end = numPts;
		double revan = 0;
		for (int i = start; i < end; i++) {
			double w = N(u, i);
			//double w = N(t, i, degree);
			double[] loc = getControlPoints().get(i).getCoordinates();
			revan += (loc[0] * plane[0] +
					loc[1] * plane[1] +
					loc[2] * plane[2] +
					plane[3]) * w;
		}
		return revan;
	}

	@Override
	public void applyTransform(Transform t) {
		ArrayList<PointND> newPoints = new ArrayList<PointND>();
		for (int i = 0; i < getControlPoints().size(); i++){
			newPoints.add(t.transform(getControlPoints().get(i)));
		}
		setControlPoints(newPoints);
	}

	@Override
	public PointND[] getRasterPoints(int number) {
		PointND [] array = new PointND[number];
		for (double i = 0; i <number ; i++){
			array[(int)(i)] = evaluate(i/(number-1));
		}
		return array;
	}

	public static float getID(String title) {
		Float result = idMap.get(title);
		if (result == null){
			result = new Float(idMap.keySet().size());
			idMap.put(title, result);
		}
		return result;
	}

	/**
	 * Rewrites the BSpline into a float representation:
	 * <pre>
	 * type
	 * total size
	 * degree
	 * # of entries in knot vector
	 * # of control points
	 * knovector entries (1 float value)
	 * control points (3 float values: x,y,z)
	 * </pre>
	 * @return the binary represenation for use with openCL
	 */
	public float[] getBinaryRepresentation() {
		int totalsize = this.getKnots().length + (this.getControlPoints().size()*3) + 5;
		float [] binary = new float [totalsize];
		int index = 0;
		binary[index] = SPLINE1DTO3D;
		index ++;
		binary[index] = totalsize;
		index ++;
		binary[index] = this.degree;
		index ++;
		binary[index] = this.getKnots().length;
		index ++;
		binary[index] = this.getControlPoints().size();
		index ++;
		for (int i = 0; i < getKnots().length; i++){
			binary[index] = (float)getKnots()[i];
			index++;
		}
		for (int i = 0; i < getControlPoints().size(); i++){
			binary[index] = (float) getControlPoints().get(i).get(0);
			index++;
			binary[index] = (float) getControlPoints().get(i).get(1);
			index++;
			binary[index] = (float) getControlPoints().get(i).get(2);
			index++;
		}
		return binary;
	}
	
	/**
	 * Calculates the derivative of a spline
	 * @return
	 */
	public BSpline getDerivative(){
		int dim = controlPoints.get(0).getDimension();
		// implementation following James D Emery: "B-Splines And Divided Differences"
		double[] knots = this.knots;
		
		ArrayList<PointND> q = new ArrayList<PointND>();
		
		for(int i = 0; i < this.controlPoints.size()+1; i++){
			double factor = degree-1;
			factor /= (knots[i+degree-1]-knots[i]);
			
			SimpleVector pHigh = (i==controlPoints.size()) ? 
					new PointND(new double[dim]).getAbstractVector() :
						new SimpleVector(controlPoints.get(i).getAbstractVector());
			SimpleVector pLow = (i==0) ? new PointND(new double[dim]).getAbstractVector() :
					controlPoints.get(i-1).getAbstractVector();
			pHigh.subtract(pLow);
			q.add(new PointND(pHigh.multipliedBy(factor).copyAsDoubleArray()));
		}
		q.remove(q.size()-1);
		q.remove(0);
		knots = new double[this.knots.length-2];
		for(int i = 0; i < knots.length; i++){
			knots[i] = (IS_CLAMPED) ? this.knots[i+1]:(i/(knots.length-1));
		}
		
		return new BSpline(q,knots);
	}

	public void setKnots(double [] knots) {
		this.knots = knots;
	}

	public double [] getKnots() {
		return knots;
	}

	/**
	 * @param controlPoints the controlPoints to set
	 */
	public void setControlPoints(ArrayList<PointND> controlPoints) {
		this.controlPoints = controlPoints;
	}

	/**
	 * @return the controlPoints
	 */
	public ArrayList<PointND> getControlPoints() {
		return controlPoints;
	}

	@Override
	public AbstractShape clone() {
		return new BSpline(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/