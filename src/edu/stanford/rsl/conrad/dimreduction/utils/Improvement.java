package edu.stanford.rsl.conrad.dimreduction.utils;


import java.util.ArrayList;

import edu.stanford.rsl.conrad.dimreduction.DimensionalityReduction;
import edu.stanford.rsl.conrad.dimreduction.LagrangianOptimization;
import edu.stanford.rsl.conrad.dimreduction.PCA;
import edu.stanford.rsl.conrad.dimreduction.objfunctions.SammonObjectiveFunction;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;

/*
 * Copyright (C) 2013-14  Susanne Westphal
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
public class Improvement {

	private double[][] reducedDistances;
	private double[][] rejectedDistanceMatrix;

	// number of outliers
	private int numOutliers = 3;

	// reduces the dimension so that only big principal components exist.
	private boolean PCAOptimizer = false;
	// maximum dimension of the reconstruction
	private int maxDim = 12;

	// Rejection of the outliers
	private boolean rejectOutliers = false;

	// maximum value of each coordinate in the result (if one point runs
	// away...)
	private boolean withRestriction = false;
	// if all coordinates have the same upper and lower bound (otherwise change
	// it in Lagrangian Optimization)
	private double upperBound = 2;
	private double lowerBound = -2;

	/**
	 * in this improvement, first a high dimensional(maxDim) reconstruction of
	 * the points is computed by using the Sammon Objective Function, then a
	 * principal component analysis is computed without reduction of the
	 * dimensions, then the average of the principal components is computed and
	 * the ones that are smaller rejected. The distance matrix of the lower
	 * dimensional points is computed and with it and the chosen Lagrangian the
	 * low-dimensional representation of the data is computed.
	 * 
	 * @param is
	 *            boolean whether this improvement is used
	 */
	public void setPCAOptimizer(boolean is) {
		PCAOptimizer = is;
		if (rejectOutliers) {
			System.err.println("Only one improvement!");
			rejectOutliers = false;
		}
	}

	/**
	 * returns whether the PCA improvement is used
	 * 
	 * @return whether the PCA improvement is used
	 */
	public boolean getPCAOptimizer() {
		return PCAOptimizer;
	}

	/**
	 * in this improvement the points with the highest distance to the origin in
	 * the high dimensional space are rejected. Therefore first a reconstruction
	 * of the distance matrix in a high dimensional (maxDim) space is computed,
	 * then the outliers are rejected and then the chosen Lagrangian is used to
	 * compute the low dimensional mapping
	 * 
	 * @param is
	 *            boolean whether this improvement is used
	 */
	public void setRejectOutliers(boolean is) {
		if (PCAOptimizer) {
			System.err.println("only one improvement!!");
			PCAOptimizer = false;
		}
		rejectOutliers = is;
	}

	/**
	 * returns whether the reject outliers improvement is used
	 * 
	 * @return whether the reject outliers improvement is used
	 */
	public boolean getRejectOutliers() {
		return rejectOutliers;
	}

	/**
	 * Here you can set the dimension of the high dimensional reconstruction
	 * from the Sammon Transformation
	 * 
	 * @param dim
	 *            dimension of the high dimensional space
	 */
	public void setMaxDim(int dim) {
		maxDim = dim;
	}

	/**
	 * first the point are reconstructed in a high dimensional space, by using
	 * the Sammon Objective Function. This is the dimension of the high
	 * dimensional space.
	 * 
	 * @return returns the dimension of the high dimensional reconstruction
	 */
	public int getMaxDim() {
		return maxDim;
	}

	/**
	 * sets the number of outliers
	 * 
	 * @param num
	 *            number of outliers
	 */
	public void setNumOutliers(int num) {
		this.numOutliers = num;
	}

	/**
	 * returns the number of outliers in the rejectOutlier improvement
	 * 
	 * @return the number of outliers in the rejectOutlier improvement
	 */
	public int getNumOutliers() {
		return numOutliers;
	}

	/**
	 * sets the lower bound of the restriction Improvement
	 * default setting is -2.0, because the inner-point distances are scaled to a range of 0 to 4
	 * @param lowerBound
	 */
	public void setLowerBound(double lowerBound) {
		this.lowerBound = lowerBound;
	}

	/**
	 * returns the lower bound of the restricted coordinates
	 * 
	 * @return the lower bound of the restricted coordinates
	 */
	public double getLowerBound() {
		return lowerBound;
	}
	/**
	 * sets the upper bound of the restriction Improvement
	 * default setting is 2.0, because the inner-point distances are scaled to a range of 0 to 4
	 * @param upperBound
	 */
	public void setUpperBound(double upperBound) {
		this.upperBound = upperBound;
	}

	/**
	 * returns the upper bound of the restricted coordinates
	 * 
	 * @return the upper bound of the restricted coordinates
	 */
	public double getUpperBound() {
		return upperBound;
	}

	/**
	 * in a standard optimization the coordinates cannot become bigger or
	 * smaller than the lower or upper bound if the restriction is set (the inner-point distances are
	 * scaled to a range of 0 to 4)
	 * 
	 * @param is
	 *            whether the restriction improvement should be used
	 */
	public void setRestriction(boolean is) {
		this.withRestriction = is;
	}

	/**
	 * returns whether restrictions are used
	 * 
	 * @return whether restrictions are used
	 */
	public boolean getRestriction() {
		return withRestriction;
	}

	/**
	 * Chooses the right improvement
	 * 
	 * @param optimization
	 *            actual dimensionality Reduction
	 * @param distances
	 *            actual distancematrix
	 * @return the coordinates of the result
	 * @throws Exception
	 */
	public double[] improve(DimensionalityReduction optimization,
			double[][] distances) throws Exception {
		if (rejectOutliers) {
			return this.rejectPoints(optimization, distances);
		} else if (PCAOptimizer) {
			return this.pcaImprovement(optimization, distances);
		} else {
			return null;
		}
	}

	/**
	 * Dimension reduction by a PCA before the optimization. (1) reconstruction
	 * of the points with the SOF (2) PCA without dimension reduction (3)
	 * average of the principal components (4) PCA so that only the dimensions
	 * with a bigger size than the average exist (5) computation of the new
	 * distance matrix (6) reconstruction of the points with a Objective
	 * function
	 * 
	 * @param optimization
	 *            actual Optimization
	 * @param distances
	 *            original distance Matrix
	 * @return the coordinates of the reconstructed points
	 * @throws Exception
	 */
	public double[] pcaImprovement(DimensionalityReduction optimization,
			double[][] distances) throws Exception {

		int tarDim = optimization.getTargetDimension();
		optimization.setTargetDimension(maxDim);

		LagrangianOptimization lagrangian = new LagrangianOptimization(
				distances, maxDim, optimization, new SammonObjectiveFunction());
		double[] randomInitialization = lagrangian.randomInitialization();
		System.out.println("Sammon Transformation to maxDim running...");
		double[] resultSammon = lagrangian.optimize(randomInitialization);

		if (optimization.getComputeError()) {
			Error error = new Error();
			error.computeError(HelperClass.wrapArrayToPoints(resultSammon,
					distances.length), "Sammon high dimensional ", distances);
		}
		System.out.println("PCA without dimension reduction.");
		PCA pca = new PCA();
		pca.setTargetDimension(maxDim);
		pca.computePCA(HelperClass.wrapDoubleToPointND(resultSammon,
				distances.length));
		double[] principalComponents = pca.getPrincipalComponents();

		double sum = 0.0;
		for (int i = 0; i < principalComponents.length; ++i) {
			sum += principalComponents[i];
		}
		double average = sum / principalComponents.length;
		System.out.println("Average pc: " + average);
		int reduceDimension = 0;
		while (principalComponents[reduceDimension] > average) {
			reduceDimension++;
		}
		System.out.println("number of dimensions of PCA: " + reduceDimension);
		System.out.println("PCA with dimension reduction.");
		PCA pcaReduceDim = new PCA();
		pcaReduceDim.setTargetDimension(reduceDimension);
		ArrayList<PointND> solPCA = pcaReduceDim.computePCA(HelperClass
				.wrapDoubleToPointND(resultSammon, distances.length));
		if (optimization.getComputeError()) {
			Error error = new Error();
			error.computeError(
					HelperClass.wrapArrayToPoints(
							HelperClass.wrapArrayListToDouble(solPCA),
							distances.length),
					"PCA with dimensionality reduction ", distances);
		}
		ArrayList<PointND> reducedDimPoints = pcaReduceDim.getPointsInDim();
		HelperClass.showPoints(HelperClass.wrapArrayListToDouble(solPCA),
				solPCA.size(), "PCA...");

		optimization.setTargetDimension(tarDim);

		System.out
				.println("Build distance matrix from dimension reduced coordinates.");
		reducedDistances = HelperClass.buildDistanceMatrix(HelperClass
				.wrapListToArray(reducedDimPoints));
		LagrangianOptimization lagrangianFunction = new LagrangianOptimization(
				reducedDistances, optimization.getTargetDimension(),
				optimization, (PointCloudViewableOptimizableFunction) optimization.getTargetFunction());
		// double[] randomIni2 = lagrangianFunction.randomInitialization(
		// distances.length, optimization.getTargetDimension());
		//
		System.out.println("Startinitialization is PCA to target dimension.");
		PCA pcaReduceDimTargetDim = new PCA();
		pcaReduceDimTargetDim.setTargetDimension(tarDim);
		ArrayList<PointND> solPCATargetDim = pcaReduceDimTargetDim
				.computePCA(HelperClass.wrapDoubleToPointND(resultSammon,
						distances.length));
		double[] ini = HelperClass.wrapArrayListToDouble(solPCATargetDim);

		System.out.println("Optimize with chosen target function...");
		double[] resultOpt = lagrangianFunction.optimize(ini);
		HelperClass.showPoints(resultOpt, distances.length, "improved??!?!??!");
		System.out.println("Finished with PCA Optimizer.");
		return resultOpt;
	}

	/**
	 * returns the reduced distance matrix 
	 * 
	 * @return the reduced distance matrix computed by the PCAImprovement
	 */
	public double[][] getReducedDistances() {
		return reducedDistances;
	}

	/**
	 * in this improvement the points with the highest distance to the origin in
	 * the high dimensional space are rejected. Therefore first a reconstruction
	 * of the distance matrix in a high dimensional (maxDim) space is computed,
	 * then the outliers are rejected and then the chosen Lagrangian is used to
	 * compute the low dimensional mapping
	 * 
	 * @param optimization
	 *            actual optimization
	 * @param distances
	 *            original distance matrix
	 * @return the coordintates of the reconstructed points
	 * @throws Exception
	 */
	public double[] rejectPoints(DimensionalityReduction optimization,
			double[][] distances) throws Exception {
		if (numOutliers < distances.length) {

			int tarDim = optimization.getTargetDimension();
			optimization.setTargetDimension(maxDim);

			LagrangianOptimization lagrangian = new LagrangianOptimization(
					distances, maxDim, optimization,
					new SammonObjectiveFunction());
			double[] randomInitialization = lagrangian.randomInitialization();
			System.out.println("Sammon Transformation to maxDim running...");
			double[] resultSammon = lagrangian.optimize(randomInitialization);
			System.out
					.println("Remove the numPoints points with the largest distance to the origin.");
			PointND sum = new PointND(new double[maxDim]);
			for (int i = 0; i < distances.length; ++i) {
				for (int j = 0; j < maxDim; ++j) {
					sum.set(j, sum.get(j) + resultSammon[i * maxDim + j]);
				}
			}

			ArrayList<PointND> sammonHDim = new ArrayList<PointND>();
			PointND point = new PointND(new double[maxDim]);
			for (int i = 0; i < distances.length; ++i) {
				for (int j = 0; j < maxDim; ++j) {
					point.set(j, resultSammon[i * maxDim + j] - sum.get(j)
							/ distances.length);
				}
				sammonHDim.add(point.clone());
			}

			PointND origin = new PointND(new double[maxDim]);

			double euklideanDistance = 0.0;
			int indexMax = 0;
			for (int i = 0; i < numOutliers; ++i) {
				euklideanDistance = 0.0;
				for (int j = 0; j < sammonHDim.size(); ++j) {
					if (euklideanDistance < sammonHDim.get(j)
							.euclideanDistance(origin)) {
						euklideanDistance = sammonHDim.get(j)
								.euclideanDistance(origin);
						indexMax = j;
					}
				}

				sammonHDim.remove(indexMax);
			}
			HelperClass.showPoints(
					HelperClass.wrapArrayListToDouble(sammonHDim),
					sammonHDim.size(), "only non rejected points");
			rejectedDistanceMatrix = HelperClass
					.buildDistanceMatrix(HelperClass
							.wrapListToArray(sammonHDim));
			System.out.println(rejectedDistanceMatrix.length
					+ " = number of points left");

			optimization.setTargetDimension(tarDim);
			LagrangianOptimization newlagrangian = new LagrangianOptimization(
					rejectedDistanceMatrix, tarDim, optimization,
					(PointCloudViewableOptimizableFunction) optimization.getTargetFunction());
			double[] solreject = newlagrangian.optimize(newlagrangian
					.randomInitialization());

			if (optimization.getComputeError()) {
				Error error = new Error();
				System.out.println("Error, without the outliers: "
						+ error.computeError(HelperClass.wrapArrayToPoints(
								solreject, rejectedDistanceMatrix.length),
								"rejection of points", rejectedDistanceMatrix));
			}
			return solreject;

		} else {
			throw new MyException("You cannot delete more points than exist!");
		}

	}

	/**
	 * 
	 * @return the distance matrix computed without the outliers in the method
	 *         rejectPoints
	 */
	public double[][] getDistances() {
		return rejectedDistanceMatrix;
	}

}
