/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels;

import ij.ImageJ;
import ij.gui.Plot;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels.KernelFunction;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.kernels.PolynomialKernel;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;

/**
 * This class performs Kernel Principal Component Analysis (KPCA) on a data-set. The data-set can be composed of scalar values or 
 * of any dimension, e.g. vertices of a surface mesh representing shape. The columns of the data-set are treated as random 
 * variables, hence one sample needs to be stored in one column only, no matter what dimensionality. 
 * The implementation here calculates the Eigen-Analysis of the covariance matrix in feature space using a singular value decomposition. The 
 * Eigen-Values and -Vectors will be accessible through the class members.
 * The implementation assumes, that the dataset has been subject to Generalized Procrustes Alignment. If an implementation of GPA 
 * other than the one provided here is used, modifications to PCA (e.g. re-scaling and consensus subtraction) might not be necessary.
 * Sch�lkopf, Bernhard, Alexander Smola, and Klaus-Robert M�ller. "Nonlinear component analysis as a kernel eigenvalue problem." Neural computation 10.5 (1998): 1299-1319.
 * @author Mathias Unberath
 *
 */
public class KPCA extends PCA {

	/**
	 * Number of principal modes the data-sets are projected onto.
	 */
	public int numProjections = 2;
	/**
	 * The principal components stored for each data-set.
	 */
	public SimpleMatrix scores;
	/**
	 * The kernel to be used in KPCA.
	 */
	public KernelFunction kernel;
	/**
	 * Uncentered K-matrix corresponding to Sch�lkopf et al.
	 */
	private SimpleMatrix featureMatrix;

	//==========================================================================================
	// METHODS
	//==========================================================================================

	public SimpleMatrix getFeatureMatrix() {
		assert (featureMatrix != null) : new Exception("Run KPCA first!");
		return featureMatrix;
	}

	public void setFeatureMatrix(SimpleMatrix feat) {
		this.featureMatrix = feat;
	}

	/**
	 * Constructs an empty KPCA object. Variables need to be initialized before analysis can be performed.
	 * TODO implement init method for this constructor
	 */
	public KPCA() {
		this.variationThreshold = 0.95;
		this.kernel = new PolynomialKernel();
	}

	/**
	 * Constructs a KPCA object and initializes the data array and count variables.
	 * @param data The data array to be analyzed.
	 */
	public KPCA(DataMatrix data) {
		super(data);
		this.variationThreshold = 0.95;
		this.kernel = new PolynomialKernel();
	}

	/**
	 * Allows setting of a kernel function from outside of the class. Could be handled with GUI.
	 * @param kernel
	 */
	public void setKernel(KernelFunction kernel) {
		this.kernel = kernel;
	}

	/**
	 * Constructs a KPCA object and initializes the data array.
	 * Due to the lacking information about scaling factors and consensus object, this constructor is not to be 
	 * used for statistical shape model generation after generalized procrustes analysis.
	 * @param data The data array to be analyzed.
	 * @param dim The dimension of the data points.
	 */
	public KPCA(SimpleMatrix data, int dim) {
		super(data, dim);
		this.variationThreshold = 0.95;
		this.kernel = new PolynomialKernel();
	}

	@Override
	public void run() {
		assert (data != null) : new Exception("Initialize data array fist.");
		System.out.println("Starting principal component analysis on " + numSamples + " data-sets.");

		SimpleMatrix k = computeCenteredKMatrix();

		DecompositionSVD svd = new DecompositionSVD(k);

		plot(svd.getSingularValues());

		int threshold = getPrincipalModesOfVariation(svd.getSingularValues());
		double[] ev = new double[threshold];
		SimpleMatrix vec = new SimpleMatrix(numPoints, threshold);
		for (int i = 0; i < threshold; i++) {
			ev[i] = svd.getSingularValues()[i];
			vec.setColValue(i, svd.getU().getCol(i));
		}

		try {
			this.eigenVectors = normalizeColumns(vec, ev);
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.eigenValues = ev;
	}

	/**
	 * This method calculates the principal components, i.e. projections onto the eigenvectors, in feature space for all training data-sets.
	 * The procedure can take some time, as normalization in feature space has to be conducted, too. 
	 * Resulting principal components are stored in the scores class member.
	 */
	public void projectTrainingSets() {
		assert (this.eigenValues != null) : new Exception("Run KPCA first.");

		SimpleMatrix k1mp = allElementsEqualMatrix(numSamples, 1, 1 / (float) numSamples);
		SimpleMatrix k1m = allElementsEqualMatrix(numSamples, numSamples, 1 / (float) numSamples);
		SimpleMatrix k1mpK = SimpleOperators.multiplyMatrixProd(k1mp, featureMatrix);
		SimpleMatrix k1mpKk1m = SimpleOperators.multiplyMatrixProd(k1mp,
				SimpleOperators.multiplyMatrixProd(featureMatrix, k1m));

		SimpleMatrix scores = new SimpleMatrix(numProjections, numSamples);
		for (int i = 0; i < numSamples; i++) {
			// calculate k matrix for this set and center the matrix
			SimpleMatrix kMat = new SimpleMatrix(1, numSamples);
			for (int k = 0; k < numSamples; k++) {
				kMat.setElementValue(0, k, kernel.evaluateKernel(data.getCol(i), data.getCol(k)));
			}
			SimpleMatrix kk1m = SimpleOperators.multiplyMatrixProd(kMat, k1m);
			kMat.subtract(k1mpK);
			kMat.subtract(kk1m);
			kMat.add(k1mpKk1m);

			for (int j = 0; j < numProjections; j++) {
				SimpleVector alpha = eigenVectors.getCol(j);
				SimpleVector res = SimpleOperators.multiply(kMat, alpha);

				scores.setElementValue(j, i, res.getElement(0));
			}
		}
		this.scores = scores;
	}

	/**
	 * This method calculates the principal components, i.e. projections onto the eigenvectors, in feature space for one input data-set 
	 * using all training data-sets.
	 * The procedure can take some time, as normalization in feature space has to be conducted, too. 
	 * Resulting principal components are returned as double array.
	 * @param shapeMat The shape to be projected as stored in a mesh-class object
	 * @return The principal components.
	 */
	public double[] projectDataSet(SimpleMatrix shapeMat) {
		assert (this.eigenValues != null) : new Exception("Run KPCA first.");
		assert (shapeMat.getCols() * shapeMat.getRows() == numPoints) : new IllegalArgumentException(
				"Input shape does not correspond to training-shapes.");

		SimpleVector shape = toSimpleVector(alignWithConsensus(centerShape(shapeMat)));
		shape.subtract(toSimpleVector(data.consensus));

		SimpleMatrix k1mp = allElementsEqualMatrix(numSamples, 1, 1 / (float) numSamples);
		SimpleMatrix k1m = allElementsEqualMatrix(numSamples, numSamples, 1 / (float) numSamples);
		SimpleMatrix k1mpK = SimpleOperators.multiplyMatrixProd(k1mp, featureMatrix);
		SimpleMatrix k1mpKk1m = SimpleOperators.multiplyMatrixProd(k1mp,
				SimpleOperators.multiplyMatrixProd(featureMatrix, k1m));

		SimpleVector scores = new SimpleVector(numProjections);
		// calculate k matrix for this set and center the matrix
		SimpleMatrix kMat = new SimpleMatrix(1, numSamples);
		for (int k = 0; k < numSamples; k++) {
			kMat.setElementValue(0, k, kernel.evaluateKernel(shape, data.getCol(k)));
		}
		SimpleMatrix kk1m = SimpleOperators.multiplyMatrixProd(kMat, k1m);
		kMat.subtract(k1mpK);
		kMat.subtract(kk1m);
		kMat.add(k1mpKk1m);

		for (int j = 0; j < numProjections; j++) {
			SimpleVector alpha = eigenVectors.getCol(j);
			SimpleVector res = SimpleOperators.multiply(kMat, alpha);

			scores.setElementValue(j, res.getElement(0));
		}
		return scores.copyAsDoubleArray();
	}

	/**
	 * This method computes the centered K matrix using the kernel method as described in Sch�lkopf et al.
	 * @return The centered K matrix
	 */
	private SimpleMatrix computeCenteredKMatrix() {
		SimpleMatrix k = new SimpleMatrix(numSamples, numSamples);

		System.out.println("Calculating K-Matrix. This can take a while.");
		for (int i = 0; i < numSamples; i++) {
			for (int j = 0; j < numSamples; j++) {
				k.setElementValue(i, j, kernel.evaluateKernel(data.getCol(i), data.getCol(j)));
			}
		}
		this.featureMatrix = k;
		System.out.println("Centering K-Matrix.");
		SimpleMatrix oneOverM = allElementsEqualMatrix(numSamples, numSamples, 1 / (float) numSamples);

		// calculate centered K matrix following Schölkopf et al.
		// K' = K - 1_m*K - K*1_m + 1_m*K*1_m
		SimpleMatrix k1m = SimpleOperators.multiplyMatrixProd(k, oneOverM);
		SimpleMatrix kprime = k.clone();
		kprime.subtract(SimpleOperators.multiplyMatrixProd(oneOverM, k));
		kprime.subtract(k1m);
		kprime.add(SimpleOperators.multiplyMatrixProd(oneOverM, k1m));

		return kprime;
	}

	/**
	 * Constructs a SimpleMatrix that has the same value in each entry.
	 * @param rows Number of rows
	 * @param cols Number of cols
	 * @param val Value for each element
	 * @return SimpleMatrix 
	 */
	private SimpleMatrix allElementsEqualMatrix(int rows, int cols, float val) {
		SimpleMatrix m = new SimpleMatrix(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m.setElementValue(i, j, val);
			}
		}
		return m;
	}

	@Override
	public SimpleMatrix applyWeight(float[] weights) {
		throw new UnsupportedOperationException(
				"Method not applicable as variational modes are defined in feature space and not explicitely calculated.");
	}

	/**
	 * Normalizes the columns of a matrix.
	 * @param m The matrix whose columns will be normalized.
	 * @param scales The value for each column to which should be normalized
	 * @return A matrix with normalized column vectors.
	 * @throws Exception 
	 */
	private SimpleMatrix normalizeColumns(SimpleMatrix m, double[] scales) throws Exception {
		if (scales == null) {
			throw new Exception("Normalization values have not been set!");
		} else if (scales.length != m.getCols()) {
			throw new Exception("Normalization values don't match matrix dimensions!");
		}

		SimpleMatrix norm = new SimpleMatrix(m.getRows(), m.getCols());

		for (int j = 0; j < m.getCols(); j++) {
			double s = 0;

			for (int i = 0; i < m.getRows(); i++) {
				s += Math.pow(m.getElement(i, j), 2);
			}

			s = Math.sqrt(s) * scales[j];
			norm.setColValue(j, m.getCol(j).dividedBy(s));
		}

		return norm;
	}

	/**
	 * Plots the data in the array over its array index.
	 * @param data The data to be plotted.
	 */
	private void plot(double[] data) {
		if (PLOT_SINGULAR_VALUES) {
			new ImageJ();
			Plot plot = VisualizationUtil.createPlot(data, "Singular values of data matrix", "Singular value",
					"Magnitude");
			plot.show();
		}
	}

	/**
	 * Plots the first two pricnipal components of the training set. 
	 */
	public void plotScore() {
		assert (scores.getRow(1) != null) : new Exception("Calculate scores for minimum 2 components first!");

		double[] x = scores.getRow(0).copyAsDoubleArray();
		double[] y = scores.getRow(1).copyAsDoubleArray();

		double miny = Double.MAX_VALUE;
		double maxy = -Double.MAX_VALUE;
		double minx = Double.MAX_VALUE;
		double maxx = -Double.MAX_VALUE;
		for (int i = 0; i < x.length; i++) {
			miny = (y[i] < miny) ? y[i] : miny;
			maxy = (y[i] > maxy) ? y[i] : maxy;
			minx = (x[i] < minx) ? x[i] : minx;
			maxx = (x[i] > maxx) ? x[i] : maxx;

		}
		if (miny == maxy) {
			maxy++;
		}
		if (minx == maxx) {
			maxx++;
		}
		minx *= 1.1;
		maxx *= 1.1;
		miny *= 1.1;
		maxy *= 1.1;

		//Plot plot = VisualizationUtil.createPlot(x, y, "Training-data scores", "1st principal component", "2nd principal component");
		new ImageJ();
		Plot plot = new Plot("Training-data scores: " + kernel.getName(), "1st principal component",
				"2nd principal component", new double[1], new double[1]);
		plot.setLimits(minx, maxx, miny, maxy);
		plot.addPoints(x, y, Plot.CROSS);
		plot.draw();
		plot.show();

	}

	/**
	 * Transforms a SimpleMatrix into a SimpleVector by appending each consecutive row to the former.
	 * @param m The SimpleMatrix.
	 * @return The SimpleMatrix as SimpleVector.
	 */
	private SimpleVector toSimpleVector(SimpleMatrix m) {
		SimpleVector v = new SimpleVector(m.getRows() * m.getCols());
		for (int i = 0; i < m.getRows(); i++) {
			for (int j = 0; j < m.getCols(); j++) {
				v.setElementValue(i * m.getCols() + j, m.getElement(i, j));
			}
		}
		return v;
	}

	/**
	 * Aligns a shape matrix to the consensus object of the active shape model for fitting puposes.
	 * @param m2
	 * @return aligned shape
	 */
	private SimpleMatrix alignWithConsensus(SimpleMatrix m2) {
		SimpleMatrix m1 = data.consensus;

		// create matrix containing information about both point-clouds m1^T * m2
		SimpleMatrix m1Tm2 = SimpleOperators.multiplyMatrixProd(m1.transposed(), m2);
		// perform SVD such that:
		// m1^T * m2 = U sigma V^T
		DecompositionSVD svd = new DecompositionSVD(m1Tm2, true, true, true);
		// exchange sigma with new matrix s having only +/- 1 as singular values
		// this allows only for rotations but no scaling, e.g. sheer
		// signum is the same as in sigma, hence reflections are still taken into account
		int nColsS = svd.getS().getCols();
		SimpleMatrix s = new SimpleMatrix(nColsS, nColsS);
		for (int i = 0; i < nColsS; i++) {
			s.setElementValue(i, i, Math.signum(svd.getSingularValues()[i]));
		}
		// calculate rotation matrix such that:
		// H = V s U^T
		SimpleMatrix h = SimpleOperators.multiplyMatrixProd(svd.getV(),
				SimpleOperators.multiplyMatrixProd(s, svd.getU().transposed()));
		return SimpleOperators.multiplyMatrixProd(m2, h);
	}

	/**
	 * Centers the shape passed to the method, assuming that vertices are stored row-wise in the shape matrix.
	 * @param shapeMat
	 * @return centered shape
	 */
	private SimpleMatrix centerShape(SimpleMatrix shapeMat) {
		SimpleVector mean = new SimpleVector(shapeMat.getCols());
		for (int i = 0; i < shapeMat.getRows(); i++) {
			mean.add(shapeMat.getRow(i));
		}
		mean.divideBy(shapeMat.getRows());
		SimpleMatrix centered = new SimpleMatrix(shapeMat.getRows(), shapeMat.getCols());
		for (int i = 0; i < shapeMat.getRows(); i++) {
			SimpleVector row = shapeMat.getRow(i);
			row.subtract(mean);
			centered.setRowValue(i, row);
		}
		return centered;
	}
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/