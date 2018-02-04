package edu.stanford.rsl.tutorial.motion.estimation;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.splines.UniformCubicBSpline;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class EstimateCubic2DSpline {

	private static final int degree = 3;

	private double[] uKnots;
	private ArrayList<PointND> gridPoints;
	private double[] uPara;
	private ArrayList<PointND> cPoints;

	/**
	 * Constructor
	 * 
	 * @param gridPoints
	 *            ArrayList of the points that need to be fitted
	 */
	public EstimateCubic2DSpline(ArrayList<PointND> gridPoints) {

		this.gridPoints = gridPoints;
	}

	/**
	 * Computes uniform parameter vector
	 * 
	 * @param length
	 *            length of array
	 * @return parameter vector
	 */
	private static double[] computeParamUniform(int length) {
		int N = length - 1;
		double delta = 1.0 / N;
		double[] knots = new double[length];
		for (int i = 0; i < length; i++) {
			knots[i] = i * delta;
		}

		return knots;
	}

	/**
	 * Computes a uniform knot vector
	 * 
	 * @param ctrlPoints
	 *            the number of control points the spline will have
	 * @return knot vector
	 */
	private static double[] computeKnotsUniform(int ctrlPoints) {
		int length = ctrlPoints - (degree - 1);
		double[] knots = new double[length + 2 * degree];
		for (int i = 0; i <= degree; i++) {
			knots[i] = 0;
			knots[knots.length - 1 - i] = 1;
		}
		double denom = length - 1;
		for (int i = degree + 1; i < degree + length; i++) {
			knots[i] = (i - degree) / (denom);
		}
		return knots;
	}

	/**
	 * This method provides the public interface to fit a spline
	 * 
	 * @param ctrlPoints
	 *            number of control points the spline should have
	 * @return uniform cubic B-Spline
	 */
	public UniformCubicBSpline estimateUniformCubic(int ctrlPoints) {
		// this.uPara = uPara;
		return estimateUniformCubicInternal(gridPoints, ctrlPoints);
	}

	/**
	 * This method computes the weights based on the b-spline basis function
	 * 
	 * @param internalCoordinate
	 * @param index
	 * @return weight
	 */
	protected double N(double internalCoordinate, int index) {

		double d = 0;
		int[] coefficientArrayA = new int[2 * degree];
		int[] coefficientArrayC = new int[2 * degree];
		int degreeMinus2 = degree - 2;
		for (int j = 0; j < degree; j++) {
			double firstKnot = uKnots[index + j];
			double secondKnot = uKnots[index + j + 1];
			if (internalCoordinate >= firstKnot && internalCoordinate <= secondKnot && firstKnot != secondKnot) {
				// set required coefficients to 0
				for (int k = degree - j - 1; k >= 0; k--) {
					coefficientArrayA[k] = 0;
				}
				if (j > 0) {
					// set required coefficients to their index
					for (int k = 0; k < j; k++) {
						coefficientArrayC[k] = k;
					}
					coefficientArrayC[j] = Integer.MAX_VALUE;
				} else {
					coefficientArrayC[0] = degreeMinus2;
					coefficientArrayC[1] = degree;
				}
				int z = 0;
				while (true) {
					if (coefficientArrayC[z] < coefficientArrayC[z + 1] - 1) {
						double e = 1.0;
						int bCounter = 0;
						int y = degreeMinus2 - j;
						int p = j - 1;

						for (int m = degreeMinus2, n = degree; m >= 0; m--, n--) {
							if (p >= 0 && coefficientArrayC[p] == m) {
								int w = index + bCounter;
								double kd = uKnots[w + n];
								e *= (kd - internalCoordinate) / (kd - uKnots[w + 1]);
								bCounter++;
								p--;
							} else {
								int w = index + coefficientArrayA[y];
								double kw = uKnots[w];
								e *= (internalCoordinate - kw) / (uKnots[w + n - 1] - kw);
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
								} else {
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
						if (coefficientArrayC[z] > degreeMinus2)
							break;

						for (int k = 0; k < z; k++)
							coefficientArrayC[k] = k;
						z = 0;
					} else {
						z++;
					}
				}
				break; // required to prevent spikes
			}
		}

		return d;
	}

	/**
	 * Getter for the knot vector
	 * 
	 * @return knot vector
	 */
	public double[] getKnots() {
		return uKnots;
	}

	/**
	 * This method estimates a uniform cubic B-Spline for the provided sample
	 * points Implemented based on
	 * http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES
	 * /INT-APP/CURVE-APP-global.html
	 * 
	 * @param points
	 *            sample points
	 * @param ctrlPoints
	 *            number of control points the spline should have
	 * @return uniform cubic B-Spline
	 */
	private UniformCubicBSpline estimateUniformCubicInternal(ArrayList<PointND> points, int ctrlPoints) {

		uPara = computeParamUniform(points.size());
		int h = ctrlPoints - 1;
		int n = uPara.length - 1;

		uKnots = computeKnotsUniform(ctrlPoints);

		SimpleMatrix N = new SimpleMatrix(n - 1, h - 1);
		SimpleMatrix Qk = new SimpleMatrix(n - 1, 3);
		SimpleMatrix Q = new SimpleMatrix(h - 1, 3);

		PointND P0 = points.get(0);
		PointND PH = points.get(n);

		for (int k = 1; k < n; k++) {
			SimpleVector val = points.get(k).getAbstractVector();
			SimpleVector s1 = points.get(0).getAbstractVector().multipliedBy(N(uPara[k], 0));
			val.subtract(s1);
			SimpleVector s2 = points.get(n).getAbstractVector().multipliedBy(N(uPara[k], h));
			val.subtract(s2);
			Qk.setRowValue(k - 1, val);
		}

		for (int i = 1; i < h; i++) {
			SimpleVector rowI = new SimpleVector(0, 0, 0);
			for (int k = 1; k < n; k++) {
				rowI.add(Qk.getRow(k - 1).multipliedBy(N(uPara[k], i)));
			}
			Q.setRowValue(i - 1, rowI);
		}

		for (int k = 1; k < n; k++) {
			for (int i = 1; i < h; i++) {
				N.setElementValue(k - 1, i - 1, N(uPara[k], i));
			}
		}
		SimpleMatrix NT = N.transposed();
		SimpleMatrix M = SimpleOperators.multiplyMatrixProd(NT, N);
		SimpleMatrix inversebFuncts = M.inverse(SimpleMatrix.InversionType.INVERT_SVD);
		SimpleMatrix parameters = SimpleOperators.multiplyMatrixProd(inversebFuncts, Q);

		cPoints = new ArrayList<PointND>();
		cPoints.add(P0);
		for (int i = 0; i < ctrlPoints - 2; i++) {
			PointND p = new PointND(parameters.getElement(i, 0), parameters.getElement(i, 1),
					parameters.getElement(i, 2));
			cPoints.add(p);
		}
		cPoints.add(PH);
		UniformCubicBSpline res = new UniformCubicBSpline(cPoints, uKnots);

		return res;
	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/