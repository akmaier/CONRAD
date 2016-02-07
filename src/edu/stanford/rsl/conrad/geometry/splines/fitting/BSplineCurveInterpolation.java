/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
package edu.stanford.rsl.conrad.geometry.splines.fitting;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.BSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.Solvers;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * Class to fit a set of data points with an arbitrary degree B-spline curve using global interpolation.
 * See L Piegl. "The NURBS book". 2012, page 364
 * @author XinyunLi
 *
 */
public class BSplineCurveInterpolation {
	
	public static final int CLAMPED = 0;
	public static final int OPEN = 1;
	public static final int CLOSED = 2;
	
	public static final int UNIFORM = 0;
	public static final int CHORD = 1;
	public static final int CENTRIPETAL = 2;
	
	private ArrayList<PointND> dataPoints;	
	protected int dimension;
	protected int degree;//degree
	protected int n; //number of data points
	private double[] parameters;
	
	/** 
	 * Creates a new instance of BSplineCurveInterpolation
	 * @param dataPoints a list of points to be fitted to a BSpline curve
	 * @param degree the degree of the BSpline curve, not order!
	 */
	public BSplineCurveInterpolation(ArrayList<PointND> dataPoints, int degree){
		this.dataPoints = dataPoints;
		this.degree = degree;
		this.dimension =  dataPoints.get(0).getDimension();
		this.n = dataPoints.size();	
	}
	
	/**
	 * Fit the given data points to a BSline curve of the degree given in the Constructor
	 * User can choose different parameterization methods and endpoint conditions to form the BSpline curve
	 * @param parametrization three parameterization options : uniform, chord length and centripetal, see the predefined constants of the three types
	 * @param endpointCondition three endpoint Conditions: clamped(clamped knots), open(open knots), closed(closed curve with open knots)
	 * @return the fitted BSpline curve
	 */
	public BSpline applyInterpolation(int parametrization, int endpointCondition){
		//compute the parameter value
		double[] parameters = buildParameterVector(parametrization, endpointCondition);
		//compute the knot vector
		double[] knots = buildKnotVector(parameters, endpointCondition);	
		// set up the linear equation system
		SimpleMatrix A = buildCoefficientMatrix(knots, parameters, endpointCondition);		
		SimpleVector[] b = buildDataVector(knots);	
		// solve for control points
		ArrayList<PointND> controlPoints = solveLinearEquations(A, b, endpointCondition);	
		return new BSpline(controlPoints, knots);		
	}
	
	/**
	 * Interpolation based on given parameters and knots
	 * @param parameters interpolation parameters corresponding to the data points
	 * @param knots knot vector of the BSpline curve
	 * @param endpointCondition see applyInterpolation(int, int)
	 * @return the fitted BSpline curve
	 */
	public BSpline applyInterpolation(double[] parameters, double[] knots,  int endpointCondition){
		SimpleMatrix A = buildCoefficientMatrix(knots, parameters, endpointCondition);
		SimpleVector[] b = buildDataVector(knots);	
		ArrayList<PointND> controlPoints = solveLinearEquations(A, b, endpointCondition);
		return new BSpline(controlPoints, degree, knots);			
	}
	
	protected double[] buildParameterVector(int parametrization, int endpointCondition){	
		parameters = new double[n];
		//closed curve
		if(endpointCondition == 2){
			//very special case, the parameter corresponds to the first data point are both 0 and 1
			//So actually we have n+1 parameter, but we only take the first n for solving the linear equations
			if (parametrization == 0){
				for (int i = 1; i < n - 1; i++)
					parameters[i] = (double)i / n;
			} else if (parametrization == 1){
				for (int i = 1; i < n; i++){
					double dis = dataPoints.get(i).euclideanDistance(dataPoints.get(i - 1));
					parameters[i] = parameters[i - 1] + dis;
				}
				double total = parameters[n - 1] + dataPoints.get(0).euclideanDistance(dataPoints.get(n - 1));
				for(int i = 1; i < n; i++)
					parameters[i] /= total;
			} else if (parametrization == 2){
				for (int i = 1; i < n; i++){
					double dis = dataPoints.get(i).euclideanDistance(dataPoints.get(i - 1));
					parameters[i] = parameters[i - 1] + Math.sqrt(dis);
				}
				double total = parameters[n - 1] + dataPoints.get(0).euclideanDistance(dataPoints.get(n - 1));
				for (int i = 1; i < n; i++)
					parameters[i] /= total;			
			}
		} else {
			//same for open and clamped knots
			//equally spaced
			if (parametrization == 0){
				for (int i = 1; i < n - 1; i++)
					parameters[i] = (double)i / (n - 1);
			} else if (parametrization == 1) {
				for (int i = 1; i < n; i++) {
					double dis = dataPoints.get(i).euclideanDistance(dataPoints.get(i - 1));
					parameters[i] = parameters[i - 1] + dis;
				}
				for (int i = 1; i < n - 1; i++)
					parameters[i] /= parameters[n - 1];
			} else if (parametrization == 2) {
				for (int i = 1; i < n; i++) {
					double dis = dataPoints.get(i).euclideanDistance(dataPoints.get(i - 1));
					parameters[i] = parameters[i - 1] + Math.sqrt(dis);
				}
				for (int i = 1; i < n - 1; i++)
					parameters[i] /= parameters[n - 1];			
			}
			parameters[n - 1] = 1d; 
		}
		return parameters;
	}
	
	protected double[] buildKnotVector(double[] parameters, int endpointCondition){	
		if (endpointCondition == 1) {
			//averaging over the neighboring parameters in the middle while make two ends open
			int k = n + degree + 1; //number of knots
			double[] uKnots = new double[k];
			uKnots[n] = 1;
			for (int j = 1; j <= n - degree - 1; j++) {
				double sum = 0;
				for (int i = j; i < j + degree; i++)
					sum += parameters[i];
				uKnots[j + degree] =  sum / degree;
			}
			for (int i = 0; i < degree; i++) {
				uKnots[degree - i - 1] = uKnots[degree - i] - (uKnots[n - i] - uKnots[n - i - 1]);
				uKnots[n + i + 1] = uKnots[n + i] + (uKnots[degree + i + 1] - uKnots[degree + i]);						
			}
			return uKnots;
		} else if (endpointCondition == 2) {
			//For simplicity we use uniform knots here, may cause singularity in the coefficient matrix
			int k = n + degree * 2 + 1; //number of knots
			double[] knots = new double[k];
			for (int j = 0; j < k; j++)
				knots[j] = ((double)(j - degree)) / n;
			return knots;
		} else {
			//averaging over the neighboring parameters in the middle while make both ends clamped
			int k = n + degree + 1; //number of knots
			double[] uKnots = new double[k];
			// compute the knot vector
			for (int i = 0; i <= degree; i++) {
				uKnots[i] = 0;
				uKnots[k-i-1] = 1;
			}		
			for (int j = 1; j <= n - degree - 1; j++) {
				double sum = 0;
				for (int i = j; i < j + degree; i++)
					sum += parameters[i];
				uKnots[j + degree] =  sum / degree;
			}
			return uKnots;		
		}
	}

	protected SimpleMatrix buildCoefficientMatrix(double[] knots, double[] parameters, int endpointCondition ){
		if(endpointCondition == 2){
			double[][] A = new double[n][n];
			for (int i = 0; i < n; i ++){
				int index = Arrays.binarySearch(knots, parameters[i]);
				if (index < 0) index = -1 * index - 2; 
				if (index < degree) index = degree;
				if (index >= n + degree) index = n + degree -1;
				double[] Ai = BasisFuns(knots, parameters[i], index);
				if (index >= n){
					System.arraycopy(Ai, 0, A[i], index - degree, degree + n -index);
					System.arraycopy(Ai, degree + n -index, A[i], 0, index - n + 1);
				} else
					System.arraycopy(Ai, 0, A[i], index - degree, Ai.length );
			}
			return new SimpleMatrix(A);
		} else {
			double[][] A = new double[n][n];
			for (int i = 0; i < n; i ++){
				int index = Arrays.binarySearch(knots, parameters[i]);
				if (index < 0) index = -1 * index - 2; 
				if (index < degree) index = degree;
				if (index >= n) index = n-1;
				double[] Ai = BasisFuns(knots, parameters[i], index);
				System.arraycopy(Ai, 0, A[i], index - degree, Ai.length );
			}
			return new SimpleMatrix(A);
		}		
	}
	
	protected SimpleVector[] buildDataVector(double[] knots){
		SimpleVector[] b = new SimpleVector[dimension];
		double[][] dataPointsArray = new double[dimension][n];
 		for (int i = 0; i < dimension; i++){
			for (int j = 0; j < n; j++)
				dataPointsArray[i][j] = dataPoints.get(j).get(i);
		}
		for (int i =0; i < dimension; i++){
			b[i] = new SimpleVector(dataPointsArray[i]);
		}		
		return b;
		
	}
	
	protected ArrayList<PointND> solveLinearEquations(SimpleMatrix A, SimpleVector[] b, int endpointCondition){
		SimpleMatrix X = new SimpleMatrix(n, dimension);	
		for (int i = 0; i < dimension; i ++){
            SimpleVector xi =  Solvers.solveLinearSysytemOfEquations(A,  b[i]);
            X.setColValue(i, xi);
		}
		ArrayList<PointND> controlPoints = new ArrayList<PointND>();
		for (int i = 0; i < n; i ++){    
			controlPoints.add(new PointND(X.getRow(i)));
		}
		if(endpointCondition == 2){
			for (int i = 0; i < degree; i ++){    
				controlPoints.add(new PointND(X.getRow(i)));
			}
		}	
		return controlPoints;		
	}
	
	protected double[] BasisFuns(double[] knots, double internalCoordinate, int index) {
		//Nurbs Book Algorithms A2.2 
		//Compute the nonvanishing basis functions
		double[] N = new double[degree + 1];
		double[] left = new double[degree + 1];
		double[] right = new double[degree + 1];
		N[0] = 1.0d;
		for (int j = 1; j <= degree; j++){
			left[j] = internalCoordinate - knots[index + 1 - j];
			right[j] = knots[index + j] - internalCoordinate;
			double saved = 0.0d;
			for (int r = 0; r < j; r++){
				double temp = N[r] / (right[r+1] + left[j - r]);
				N[r] = saved + right[r + 1] * temp;
				saved  = left[j - r]  * temp;
			}
			N[j] = saved;
		}
		return N;
	}

	public static void main(String[] args) {
		
		//create a list of data points
		ArrayList<PointND> list = new ArrayList<PointND>();
		list.add(new PointND (-2, 5));
		list.add(new PointND (-5, 0));
		list.add(new PointND (-1, -3));
		list.add(new PointND (2, -3));
		list.add(new PointND (5, 2));
		list.add(new PointND (1, 3));
		
		//B-Spline curve interpolation
		int degree = 3;
		BSplineCurveInterpolation test = new BSplineCurveInterpolation(list, degree);
		BSpline spline_clamped = test.applyInterpolation(UNIFORM, CLAMPED);
		BSpline spline_open = test.applyInterpolation(CHORD, OPEN);
		BSpline spline_closed = test.applyInterpolation(CHORD, CLOSED);
	    
		//Visualization
		VisualizationUtil.createSplinePlot(spline_clamped).show();
		VisualizationUtil.createSplinePlot(spline_open).show();
		VisualizationUtil.createSplinePlot(spline_closed).show();
	}

}