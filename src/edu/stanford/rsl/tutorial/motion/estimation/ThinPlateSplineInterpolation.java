package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;

import java.util.ArrayList;

import Jama.Matrix;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

/**
 * This class provides methods to interpolate scattered data of any dimension
 * using a thin plate spline. The kernel is correct for 3-D. Other dimensions
 * will require a different kernel, e.g. r*r*ln(r) for 2-D
 * 
 * @author Marco Boegel
 * 
 */
public class ThinPlateSplineInterpolation {
	/**
	 * Input control points
	 */
	private ArrayList<PointND> gridPoints;
	/**
	 * Corresponding values to the gridPoints
	 */
	private ArrayList<PointND> values;
	/**
	 * Dimension of the points
	 */
	private int dim;
	/**
	 * TPS Interpolation coefficients
	 */
	private SimpleMatrix coefficients;
	/**
	 * Polynomial Ax+b
	 */
	private SimpleMatrix A;

	/**
	 * Polynomial Ax+b
	 */
	private SimpleVector b;
	private boolean debug = false;

	public float[] getAsFloatPoints() {
		float[] pts = new float[gridPoints.size()*dim];
		for(int i = 0; i < gridPoints.size(); i++) {
			for(int k = 0 ; k < dim; k++) {
				pts[i*dim +k] = (float) gridPoints.get(i).get(k);
			}
		}
		return pts;
	}

	public float[] getAsFloatA() {
		float[] Af = new float[A.getRows()*A.getCols()];

		for(int i = 0; i < A.getCols(); i++) {
			for(int j = 0; j < A.getRows(); j++) {
				Af[i*A.getRows()+j] = (float) A.getElement(j, i);
			}
		}
		return Af;
	}

	public float[] getAsFloatB() {
		float [] Bf = new float[b.getLen()];

		for(int i = 0; i < b.getLen(); i++) {
			Bf[i] = (float) b.getElement(i);
		}
		return Bf;
	}

	public float[] getAsFloatCoeffs() {
		float [] coeff = new float[coefficients.getCols()*coefficients.getRows()];
		for(int i = 0; i < coefficients.getCols(); i++) {
			for(int j = 0; j < coefficients.getRows(); j++) {
				coeff[i*coefficients.getRows()+j] = (float) coefficients.getElement(j, i);
			}
		}
		return coeff;
	}
	public ThinPlateSplineInterpolation(int dimension,
			ArrayList<PointND> points, ArrayList<PointND> values) {
		this.gridPoints = points;
		this.values = values;
		this.dim = dimension;
		estimateCoefficients();
	}

	/**
	 * This method accepts a new pointgrid for interpolation, and the
	 * corresponding values. It automatically re-estimates the interpolation
	 * coefficients
	 * 
	 * @param points
	 *            Interpolation grid
	 * @param values
	 *            corresponding values for the points
	 * @param dimension
	 *            Dimension
	 */
	public void setNewPointsAndRecalibrate(ArrayList<PointND> points,
			ArrayList<PointND> values, int dimension) {
		this.gridPoints = points;
		this.values = values;
		this.dim = dimension;
		estimateCoefficients();

	}

	/**
	 * Estimates the coefficients for the augmented TPS interpolation (including
	 * polynomial)
	 */
	private void estimateCoefficients() {

		long start = 0;
		if (debug)
			start = System.nanoTime();

		int n = gridPoints.size();
		int sizeL = dim * n + dim * dim + dim;

		A = new SimpleMatrix(dim, dim);
		b = new SimpleVector(dim);
		Matrix LJama = new Matrix(sizeL,sizeL);
		Matrix rhsJama = new Matrix(sizeL, 1);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < values.get(i).getDimension(); j++) {
				rhsJama.set(i*dim+j, 0, values.get(i).getAbstractVector().getElement(j));
			}
			for (int j = i+1; j < n; j++) {
				// symmetric matrix -- compute only upper triangle and write to both locations
				double val = kernel(gridPoints.get(i), gridPoints.get(j)); 
				for (int k = 0; k < dim; k++) {
					int currI = i * dim + k;
					int currJ = j * dim + k;
					LJama.set(currI, currJ, val);
					LJama.set(currJ, currI, val);
				}
			}
		}

		int offset = n*dim;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim; j++) {
				double val = gridPoints.get(i).get(j);
				for (int k = 0; k < dim; k++) {
					// Symmetric matrix!
					LJama.set(dim * i + k, offset + dim * j + k, val);
					LJama.set(offset + dim * j + k, dim * i + k, val);
				}
			}
			for (int k = 0; k < dim; k++) {
				// Symmetric matrix!
				LJama.set(dim * i + k, offset + dim * dim + k, 1.0);
				LJama.set(offset + dim * dim + k, dim * i + k, 1.0);
			}
		}
		if (debug){
			System.out.println("Time for reshaping : " + (System.nanoTime()-start)/1e6);
			start = System.nanoTime();
		}
		Jama.LUDecomposition cd = new Jama.LUDecomposition(LJama);
		Matrix parametersJama = cd.solve(rhsJama);
		if(debug)
			System.out.println("Time for inversion JAMA: " + (System.nanoTime()-start)/1e6);

		coefficients = new SimpleMatrix(dim, n);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < dim; j++) {
				coefficients.setElementValue(j, i, parametersJama.get(i * dim + j, 0));
			}
		}
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				A.setElementValue(j, i, parametersJama.get(dim * n + i * dim + j, 0));
			}
			b.setElementValue(i, parametersJama.get(dim * n + dim * dim + i, 0));
		}
	}

	/**
	 * Interpolation kernel. Implements only euclidean distance
	 * 
	 * @param p1
	 *            Point 1
	 * @param p2
	 *            Point 2
	 * @return euclidean distance between two points
	 */
	private double kernel(PointND p1, PointND p2) {
		return p1.euclideanDistance(p2);
	}

	/**
	 * Lifts the kernel to the required dimension for interpolation
	 * 
	 * @param p1
	 * @param p2
	 * @return
	 */
	private SimpleMatrix G(PointND p1, PointND p2) {
		SimpleMatrix G = new SimpleMatrix(dim, dim);
		G.identity();
		double val = kernel(p1, p2);
		G.multiplyBy(val);
		return G;
	}

	/**
	 * Interpolation function
	 * 
	 * @param pt
	 *            point at which the value should be interpolated
	 * @return value
	 */
	public SimpleVector interpolate(PointND pt) {

		SimpleVector res = new SimpleVector(pt.getDimension());
		res.zeros();
		for (int i = 0; i < coefficients.getCols(); i++) {
			SimpleVector Gc = SimpleOperators.multiply(
					G(pt, gridPoints.get(i)), coefficients.getCol(i));

			res = SimpleOperators.add(res, Gc);
		}
		SimpleVector Ax = SimpleOperators.multiply(A, pt.getAbstractVector());
		res = SimpleOperators.add(res, Ax, b);
		return res;
	}
	
	public ArrayList<PointND> getGridPoints() {
		return gridPoints;
	}

	public void setGridPoints(ArrayList<PointND> gridPoints) {
		this.gridPoints = gridPoints;
	}

	public ArrayList<PointND> getValues() {
		return values;
	}

	public void setValues(ArrayList<PointND> values) {
		this.values = values;
	}

	public SimpleMatrix getCoefficients() {
		return coefficients;
	}

	public void setCoefficients(SimpleMatrix coefficients) {
		this.coefficients = coefficients;
	}

	public SimpleMatrix getA() {
		return A;
	}

	public void setA(SimpleMatrix a) {
		A = a;
	}

	public SimpleVector getB() {
		return b;
	}

	public void setB(SimpleVector b) {
		this.b = b;
	}

	public static void main(String args[]) {
		new ImageJ();
		ArrayList<PointND> points = new ArrayList<PointND>();
		points.add(new PointND(30, 30));
		points.add(new PointND(60, 60));
		points.add(new PointND(90, 90));
		points.add(new PointND(120, 140));
		points.add(new PointND(150, 90));
		points.add(new PointND(180, 60));
		points.add(new PointND(210, 30));
		ArrayList<PointND> values = new ArrayList<PointND>();
		values.add(new PointND(100));
		values.add(new PointND(200));
		values.add(new PointND(300));
		values.add(new PointND(400));
		values.add(new PointND(300));
		values.add(new PointND(200));
		values.add(new PointND(100));

		for (int i = 0; i < 100; i++) {
			points.add(new PointND(i * 5, 400));
			values.add(new PointND(0, 0));
		}
		// for(int i = 0; i < 100;i++) {
		// points.add(new PointND(i*5,499));
		// values.add(new PointND(0,0));
		// }

		ThinPlateSplineInterpolation tps = new ThinPlateSplineInterpolation(2,
				points, values);

		tps.interpolate(new PointND(30, 210));
		Grid2D grid = new Grid2D(500, 500);

		for (int i = 0; i < 500; i++) {
			for (int j = 0; j < 500; j++) {
				grid.setAtIndex(i, j, (float) tps
						.interpolate(new PointND(i, j)).getElement(0));
			}
		}
		grid.show();


	}
}
/*
 * Copyright (C) 2010-2014 Marco B�gel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */