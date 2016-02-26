/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */package edu.stanford.rsl.conrad.geometry.splines.fitting;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.apps.gui.opengl.BSplineVolumeRenderer;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSpline;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Class to fit a grid of data points with an arbitrary degree tensor product B-Spline surface.
 * @author Xinyun Li
 *
 */
public class BSplineSurfaceInterpolation {
	
	public static final int UNIFORM = 0;
	public static final int CHORD = 1;
	public static final int CENTRIPETAL = 2;
	
	private ArrayList<PointND> dataPoints;	
	protected int dimension;
	protected int degreeU;//degree
	protected int degreeV;
	protected int m; //number of columns along u direction
	protected int n; //number of rows along v direction
	protected double[] uParameters;
	protected double[] vParameters;
	
	/**
	 * Creates a new instance of BSplineSurfaceInterpolation
	 * @param dataPoints  a list of points to be fitted to a BSpline surface, 
	 * actually is reshaped from a grid of points row by row into one list
	 * @param degreeU degree in the u direction
	 * @param degreeV degree in the v direction
	 * @param m number of columns of the grid along u direction
	 * @param n number of rows of the grid along v direction
	 */
	public BSplineSurfaceInterpolation(ArrayList<PointND> dataPoints, int degreeU, int degreeV, int m, int n){
		this.dataPoints = dataPoints;
		this.degreeU = degreeU;
		this.degreeV = degreeV;
		this.dimension =  dataPoints.get(0).getDimension();
		this.m = m;
		this.n = n;
	}
	
	/**
	 * core method, apply the interpolation, which fits the given data grid to a B-Spline surface of the degree given in the Constructor
	 * Smooth close is not supported, in order to achieve this, duplicate the first point in the end to get a clamped close
	 * @param parametrization three parameterization options : uniform, chord length and centripetal
	 * @return the fitted B-Spline surface
	 */
	public SurfaceBSpline applyInterpolation(int parametrization){
		//only support clamped knots 
		int endpointCondition = 0;
		uParameters = buildUParameterVector(parametrization, endpointCondition);
		vParameters = buildVParameterVector(parametrization, endpointCondition);
		double[] uKnots = buildKnotVector(uParameters, degreeU, endpointCondition);
		System.out.println("uKnot:" + Arrays.toString(uKnots));
		double[] vKnots = buildKnotVector(vParameters, degreeV, endpointCondition);
		System.out.println("vKnot:" + Arrays.toString(vKnots));
		ArrayList<PointND> uControlPoints = new ArrayList<PointND>();
		for(int l = 0; l < m; l ++){
			ArrayList<PointND> uPoints = new ArrayList<PointND>();
			for(int k = 0; k < n; k ++)
				uPoints.add(dataPoints.get(k * m + l));
			//System.out.println("Udata"+uPoints);
			BSplineCurveInterpolation uSplineInterpolation = new BSplineCurveInterpolation(uPoints, degreeU);
			ArrayList<PointND> temp = uSplineInterpolation.applyInterpolation(uParameters, uKnots, endpointCondition).getControlPoints();
			//System.out.println("UCP_inter"+temp);
			uControlPoints.addAll(temp);		
		}
		//uCP should be saved as columns, but as they were saved as arraylist ,they should be read via columns
		ArrayList<PointND> ControlPoints = new ArrayList<PointND>();
		for(int k = 0; k < n; k ++){			
			ArrayList<PointND> vPoints = new ArrayList<PointND>();
			for(int l = 0; l < m; l ++)
				vPoints.add(uControlPoints.get(l * n + k));
		//	System.out.println("VCP_inter"+vPoints);
			BSplineCurveInterpolation vSplineInterpolation = new BSplineCurveInterpolation(vPoints, degreeV);
			ArrayList<PointND> temp = vSplineInterpolation.applyInterpolation(vParameters, vKnots, endpointCondition).getControlPoints();
			//System.out.println("VCP_final"+temp);
			ControlPoints.addAll(temp);
		}	
		return new SurfaceBSpline(ControlPoints, uKnots, vKnots);		
	}

	protected double[] buildUParameterVector(int parametrization, int endpointCondition){
		double[] uParameters = new double[n];
		uParameters[n - 1] = 1d;
		int num = m;
		for(int l = 0; l < m; l++){
			double total = 0d;
			double[] cds = new double[n];
			for (int k = 1; k < n; k ++){
				cds[k] = dataPoints.get(k * m + l).euclideanDistance(dataPoints.get((k - 1) * m + l));
				total += cds[k];
			}
			if (total == 0) num--;
			else {
				double d = 0d;
				for(int k = 1; k < n - 1; k ++){
					d += cds[k];
					uParameters[k] += d / total;
				}
			}
		}
		if(num == 0) System.out.println("error");
		for(int k = 1; k < n - 1; k ++) uParameters[k] /= num;		
		return uParameters;
	}
	
	protected double[] buildVParameterVector(int parametrization, int endpointCondition){
		double[] vParameters = new double[m];
		vParameters[m - 1] = 1d;
		int num = n;
		for (int k = 0; k < n; k++){
			double total = 0d;
			double[] cds = new double[m];
			for (int l = 1; l < m; l ++){
				cds[l] = dataPoints.get(k * m + l).euclideanDistance(dataPoints.get(k * m + l - 1));
				total += cds[l];
			}
			if (total == 0) num--;
			else{
				double d = 0d;
				for (int l = 1; l < m - 1; l ++){
					d += cds[l];
					vParameters[l] += d / total;
				}
			}
		}
		if(num == 0) System.out.println("error");
		for(int l = 1; l < m - 1; l ++) vParameters[l] /= num;
 		return vParameters;
	}
	
	protected double[] buildKnotVector(double[] parameters, int degree, int endpointCondition){
		int num = parameters.length;
		if (endpointCondition == 1){
			int k = num + degree + 1; //number of knots
			double[] uKnots = new double[k];
			uKnots[num] = 1;
			for (int j = 1; j <= num - degree - 1; j++){
				double sum = 0;
				for (int i = j; i < j + degree; i++)
					sum += parameters[i];
				uKnots[j + degree] =  sum / degree;
			}
			for(int i = 0; i < degree; i++){
				uKnots[degree - i - 1] = uKnots[degree - i] - (uKnots[num - i] - uKnots[num - i - 1]);
				uKnots[num + i + 1] = uKnots[num + i] + (uKnots[degree + i + 1] - uKnots[degree + i]);
			}
			return uKnots;
		} else if (endpointCondition == 2){
			//uniform knots
			int k = num + degree * 2 + 1; //number of knots
			double[] knots = new double[k];
			for(int j = 0; j < k; j++)
				knots[j] = ((double)(j - degree)) / num;
			return knots;		
		} else {
			int k = num + degree + 1; //number of knots
			double[] Knots = new double[k];
			// compute the knot vector
			for(int i = 0; i <= degree; i++){
				Knots[i] = 0;
				Knots[k-i-1] = 1;
			}
			for(int j = 1; j <= num - degree - 1; j++){
				double sum = 0;
				for(int i = j; i < j + degree; i++)
					sum += parameters[i];
				Knots[j + degree] =  sum / degree;
			}
			return Knots;
		}		
	}
	
	public double[] getUParameters() {return uParameters;}
	public double[] getVParameters() {return vParameters;}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Configuration.loadConfiguration();
		//generate a semi-sphere samples
		ArrayList<PointND> list = new ArrayList<PointND>();
		int r = 1;
		list.add(new PointND (0, r, 1));
		list.add(new PointND (0.707 * r, 0.707 * r, 1));
		list.add(new PointND (r, 0, 1));
		list.add(new PointND (0.707 * r, -0.707 * r, 1));
		list.add(new PointND (0, -r, 1));
		list.add(new PointND (-0.707 * r, -0.707 * r, 1));
		list.add(new PointND (-r, 0, 1));
		list.add(new PointND (-0.707 * r, 0.707 * r, 1));
		list.add(new PointND (0, r, 1));
	
		r =2;
		list.add(new PointND (0, r, 2));
		list.add(new PointND (0.707 * r, 0.707 * r, 2));
		list.add(new PointND (r, 0, 2));
		list.add(new PointND (0.707 * r, -0.707 * r, 2));
		list.add(new PointND (0, -r, 2));
		list.add(new PointND (-0.707 * r, -0.707 * r, 2));
		list.add(new PointND (-r, 0, 2));
		list.add(new PointND (-0.707 * r, 0.707 * r,2));
		list.add(new PointND (0, r, 2));
		
		r =3;
		list.add(new PointND (0, r, 3));
		list.add(new PointND (0.707 * r, 0.707 * r, 3));
		list.add(new PointND (r, 0, 3));
		list.add(new PointND (0.707 * r, -0.707 * r, 3));
		list.add(new PointND (0, -r, 3));
		list.add(new PointND (-0.707 * r, -0.707 * r, 3));
		list.add(new PointND (-r, 0, 3));
		list.add(new PointND (-0.707 * r, 0.707 * r, 3));
		list.add(new PointND (0, r, 3));
		
		r =5;
		list.add(new PointND (0, r, 5));
		list.add(new PointND (0.707 * r, 0.707 * r, 5));
		list.add(new PointND (r, 0, 5));
		list.add(new PointND (0.707 * r, -0.707 * r, 5));
		list.add(new PointND (0, -r, 5));
		list.add(new PointND (-0.707 * r, -0.707 * r, 5));
		list.add(new PointND (-r, 0, 5));
		list.add(new PointND (-0.707 * r, 0.707 * r, 5));
		list.add(new PointND (0, r, 5));
	
		//B-Spline surface interpolation
		int degreeU = 2, degreeV = 2;
		int m = 9, n = 4;
		BSplineSurfaceInterpolation test = new BSplineSurfaceInterpolation(list, degreeU, degreeV, m, n);
		SurfaceBSpline Sbsp = test.applyInterpolation(1);
		for(int k = 0; k < test.uParameters.length; k++) {
			for(int j = 0; j < test.vParameters.length; j++) {
				PointND res = Sbsp.evaluate(test.uParameters[k], test.vParameters[j]);
				System.out.println("RES at ("+test.uParameters[k]+ ", " + test.vParameters[j] + ") is "+res);			
			}		
		}
		
		//Visulization
		BSplineVolumeRenderer test1 = new BSplineVolumeRenderer(Sbsp);
	}
	

}
