package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class RadialBasisFunctionInterpolation {

	private ArrayList<PointND> gridPoints;
	private double[] values;
	private int c;
	private SimpleVector coefficients;

	public static enum RBF {
		MQ, TPS, GAUSS
	};

	private RBF type;

	public RadialBasisFunctionInterpolation(RBF type, int c, ArrayList<PointND> points, double[] values) {
		this.gridPoints = points;
		this.values = values;
		this.c = c;
		this.type = type;
		estimateCoefficients();
	}

	private void estimateCoefficients() {

		int n = gridPoints.size();
		SimpleMatrix A = new SimpleMatrix(n, n);
		SimpleVector rhs = new SimpleVector(values.length);

		if (type == RBF.MQ) {
			for (int j = 0; j < n; j++) {
				rhs.setElementValue(j, values[j]);
				for (int k = 0; k < n; k++) {
					double val = Math.sqrt(c * c + Math.pow(gridPoints.get(j).euclideanDistance(gridPoints.get(k)), 2));
					A.setElementValue(j, k, val);
				}
			}
		} else if (type == RBF.TPS) {
			for (int j = 0; j < n; j++) {
				rhs.setElementValue(j, values[j]);
				for (int k = 0; k < n; k++) {
					double r = gridPoints.get(j).euclideanDistance(gridPoints.get(k));
					double val = r > 0 ? r * r + Math.log(r + 0.000001) : 0;
					A.setElementValue(j, k, val);
				}
			}
		} else if (type == RBF.GAUSS) {
			double sigmasquare = 100;
			for (int j = 0; j < n; j++) {
				rhs.setElementValue(j, values[j]);
				for (int k = 0; k < n; k++) {
					double r = gridPoints.get(j).euclideanDistance(gridPoints.get(k));
					double val = Math.exp((-r * r) / (sigmasquare));
					A.setElementValue(j, k, val);
				}
			}
		}
		SimpleMatrix Ainv = A.inverse(SimpleMatrix.InversionType.INVERT_SVD);
		coefficients = SimpleOperators.multiply(Ainv, rhs);

	}

	public double interpolate(PointND pt) {
		double val = 0;
		if (type == RBF.MQ) {
			for (int i = 0; i < coefficients.getLen(); i++) {
				val += coefficients.getElement(i)
						* Math.sqrt(c * c + Math.pow(pt.euclideanDistance(gridPoints.get(i)), 2));
			}
		} else if (type == RBF.TPS) {
			for (int i = 0; i < coefficients.getLen(); i++) {
				double r = pt.euclideanDistance(gridPoints.get(i));
				double tps = r > 0 ? r * r + Math.log(r + 0.000001) : 0;
				val += coefficients.getElement(i) * (tps);
			}
		} else if (type == RBF.GAUSS) {
			double sigmasquare = 100;
			for (int i = 0; i < coefficients.getLen(); i++) {
				double r = pt.euclideanDistance(gridPoints.get(i));
				val += coefficients.getElement(i) * Math.exp((-r * r) / sigmasquare);
			}
		}
		return val;
	}

	public static void main(String args[]) {
		new ImageJ();
		ArrayList<PointND> points = new ArrayList<PointND>();
		points.add(new PointND(30, 30));
		points.add(new PointND(60, 60));
		points.add(new PointND(90, 90));
		points.add(new PointND(120, 120));
		points.add(new PointND(90, 150));
		points.add(new PointND(60, 180));
		points.add(new PointND(30, 210));
		double[] values = new double[7 + 500];
		values[0] = 100;
		values[1] = 200;
		values[2] = 300;
		values[3] = 400;
		values[4] = 300;
		values[5] = 200;
		values[6] = 100;

		for (int i = 0; i < 250; i++) {
			points.add(new PointND(i * 2, 400));
			values[i + 7] = 0;
		}
		for (int i = 0; i < 250; i++) {
			points.add(new PointND(i * 2, 499));
			values[i + 7 + 250] = 0;
		}

		RadialBasisFunctionInterpolation rad = new RadialBasisFunctionInterpolation(RBF.MQ, 1, points, values);

		Grid2D grid = new Grid2D(500, 500);

		for (int i = 0; i < 500; i++) {
			for (int j = 0; j < 500; j++) {
				grid.setAtIndex(i, j, (float) rad.interpolate(new PointND(i, j)));
			}
		}
		grid.show();

		RadialBasisFunctionInterpolation rad2 = new RadialBasisFunctionInterpolation(RBF.TPS, 2, points, values);

		Grid2D grid2 = new Grid2D(500, 500);

		for (int i = 0; i < 500; i++) {
			for (int j = 0; j < 500; j++) {
				grid2.setAtIndex(i, j, (float) rad2.interpolate(new PointND(i, j)));
			}
		}
		grid2.show();

	}
}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/