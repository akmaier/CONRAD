package edu.stanford.rsl.conrad.calibration;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.DecompositionSVD;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * Implements the perspective Factorization by Sturm in order to obtain
 * projection matrices. Still yields errors and results are not reliable.
 * 
 * @author Philipp Roser
 * 
 */
public class Factorization {

	/**
	 * contains normalized image points
	 */
	private SimpleMatrix M;

	/**
	 * contains perspective depths
	 */
	private SimpleMatrix lambda;

	/**
	 * contains all matrices used for normalization
	 */
	private ArrayList<SimpleMatrix> normalizationMatrices;

	/**
	 * measurement matrix
	 */
	private SimpleMatrix W;

	/**
	 * matrix containing motion (projection matrices for each frame)
	 */
	private SimpleMatrix R;

	/**
	 * matrix containing shape (world points)
	 */
	private SimpleMatrix S;

	/**
	 * contains projection matrices for each frame
	 */
	private ArrayList<SimpleMatrix> projectionMatrices;

	/**
	 * contains world points
	 */
	private ArrayList<PointND> worldPoints;

	/**
	 * data set to be factorized. Note that each image point has to be visible
	 * in each frame.
	 */
	private ArrayList<ArrayList<PointND>> imagePoints;

	/**
	 * formerly used to save normalized image points
	 * 
	 * @deprecated actually no use
	 */
	private ArrayList<ArrayList<PointND>> registeredPoints;

	/**
	 * number of points in each frame
	 */
	private int numberOfPoints;

	/**
	 * number of frames
	 */
	private int numberOfFrames;

	/**
	 * Executes the complete perpective Factorization
	 * 
	 * @param points
	 *            , each point has to be visible in each frame.
	 */
	public Factorization(ArrayList<ArrayList<PointND>> points) {
		imagePoints = points;
		numberOfFrames = points.size();
		numberOfPoints = points.get(0).size();

		initLambda();
		initM();
		computeW();
		estimateLambda();
		denormalizePMatrices();
	}

	/**
	 * initializes perspective depths with 1
	 */
	private void initLambda() {
		lambda = new SimpleMatrix(this.numberOfFrames, this.numberOfPoints);
		lambda.ones();
	}

	/**
	 * normalized points and builds M. rank = 4 is enforced.
	 */
	private void initM() {
		normalizationMatrices = new ArrayList<SimpleMatrix>();
		M = new SimpleMatrix(3 * numberOfFrames, numberOfPoints);
		for (int i = 0; i < 3 * numberOfFrames; i += 3) {
			SimpleMatrix T = estimateNormalization(imagePoints.get(i / 3));
			for (int j = 0; j < numberOfPoints; j++) {
				M.setElementValue(i, j, imagePoints.get(i / 3).get(j).get(0)
						* T.getElement(0, 0) + T.getElement(0, 2));
				M.setElementValue(i + 1, j, imagePoints.get(i / 3).get(j)
						.get(1)
						* T.getElement(1, 1) + T.getElement(1, 2));
				M.setElementValue(i + 2, j, 1.0);
			}
		}
		DecompositionSVD svd = new DecompositionSVD(M);
		SimpleMatrix newS = new SimpleMatrix(svd.getS().getRows(), svd.getS()
				.getCols());
		newS.zeros();
		newS.setElementValue(0, 0, svd.getS().getElement(0, 0));
		newS.setElementValue(1, 1, svd.getS().getElement(1, 1));
		newS.setElementValue(2, 2, svd.getS().getElement(2, 2));
		newS.setElementValue(3, 3, svd.getS().getElement(3, 3));
		M = new SimpleMatrix(SimpleOperators.multiplyMatrixProd(SimpleOperators
				.multiplyMatrixProd(svd.getU(), newS), svd.getV().transposed()));
	}

	/**
	 * computes W out of M and lambda. rank = 4 is enforced.
	 */
	private void computeW() {
		W = new SimpleMatrix(M.getRows(), M.getCols());
		for (int i = 0; i < M.getRows(); i++) {
			for (int j = 0; j < M.getCols(); j++) {
				W.setElementValue(i, j, getValueAt(i, j));
			}
		}
		DecompositionSVD svd = new DecompositionSVD(W);
		SimpleMatrix newS = new SimpleMatrix(svd.getS().getRows(), svd.getS()
				.getCols());
		newS.zeros();
		newS.setElementValue(0, 0, svd.getS().getElement(0, 0));
		newS.setElementValue(1, 1, svd.getS().getElement(1, 1));
		newS.setElementValue(2, 2, svd.getS().getElement(2, 2));
		newS.setElementValue(3, 3, svd.getS().getElement(3, 3));
		W = new SimpleMatrix(SimpleOperators.multiplyMatrixProd(SimpleOperators
				.multiplyMatrixProd(svd.getU(), newS), svd.getV().transposed()));
	}

	/**
	 * formerly used to normalize points
	 * 
	 * @deprecated not used any longer
	 */
	private void registerPoints() {
		this.registeredPoints = new ArrayList<ArrayList<PointND>>();
		for (ArrayList<PointND> list : this.imagePoints) {
			double sumX = 0;
			double sumY = 0;
			for (PointND p : list) {
				sumX += p.get(0);
				sumY += p.get(1);
			}
			ArrayList<PointND> registered = new ArrayList<PointND>();
			PointND centroid = new PointND(sumX / list.size(), sumY
					/ list.size());
			for (PointND p : list) {
				PointND registeredP = new PointND(p.get(0) - centroid.get(0),
						p.get(1) - centroid.get(1));
				registered.add(registeredP);
			}
			this.registeredPoints.add(registered);
		}
	}

	/**
	 * formerly used to normalize points
	 * 
	 * @deprecated not used any longer
	 */
	private void normalizePoints() {
		for (ArrayList<PointND> list : this.registeredPoints) {
			int meanDistance = 0;
			for (PointND p : list) {
				meanDistance += p.getAbstractVector().normL2();
			}
			meanDistance = meanDistance / list.size();
			for (PointND p : list) {
				p.set(0, p.get(0) * Math.sqrt(2.0) / meanDistance);
				p.set(1, p.get(1) * Math.sqrt(2.0) / meanDistance);
			}
		}
	}

	/**
	 * estimates centroid of list
	 * 
	 * @param list
	 *            , containing all image points of one frame
	 * @return centroid of the data set
	 */
	private PointND estimateCentroid2D(ArrayList<PointND> list) {
		PointND centroid = new PointND(0, 0);
		for (PointND p : list) {
			centroid.set(0, centroid.get(0) + p.get(0) / list.size());
			centroid.set(1, centroid.get(1) + p.get(1) / list.size());
		}
		return centroid;
	}

	/**
	 * estimates the scale for uniform scaling
	 * 
	 * @param centroid
	 *            of list
	 * @param list
	 *            of image points
	 * @return scale
	 */
	private double estimateScale(PointND centroid, ArrayList<PointND> list) {
		double scale = 0.0;
		for (PointND p : list) {
			scale += p.euclideanDistance(centroid);
		}
		scale = Math.sqrt(2.0) / scale;
		return scale;
	}

	/**
	 * estimates the similarity transform matrix T
	 * 
	 * @param list
	 *            , containing image points of one frame
	 * @return similarity transform T
	 */
	private SimpleMatrix estimateNormalization(ArrayList<PointND> list) {
		// estimate centroid and scale
		PointND centroid = estimateCentroid2D(list);
		double scale = estimateScale(centroid, list);
		// build transformation matrix T
		SimpleMatrix T = new SimpleMatrix(3, 3);
		// first row
		T.setElementValue(0, 0, scale);
		T.setElementValue(0, 1, 0);
		T.setElementValue(0, 2, -centroid.get(0) * scale);
		// second row
		T.setElementValue(1, 0, 0);
		T.setElementValue(1, 1, scale);
		T.setElementValue(1, 2, -centroid.get(1) * scale);
		// third row
		T.setElementValue(2, 0, 0);
		T.setElementValue(2, 1, 0);
		T.setElementValue(2, 2, 1);
		normalizationMatrices.add(T);
		return T;
	}

	/**
	 * estimates the perspective depths iteratively
	 */
	private void estimateLambda() {

		double diff = 100.0;

		while (diff > CONRAD.DOUBLE_EPSILON) {

			diff = 0;

			for (int i = 0; i < W.getCols(); i++) {
				double norm = getCol(i).normL2();
				for (int j = 0; j < W.getRows(); j += 3) {
					setLambda(j / 3, i, getLambda(j / 3, i) / (norm));
				}
			}

			for (int i = 0; i < W.getRows(); i += 3) {
				double norm = getRowTriplet(i).normL2();
				for (int j = 0; j < W.getCols(); j++) {
					lambda.setElementValue(i / 3, j,
							lambda.getElement(i / 3, j) / (norm));
				}
			}

			/*
			 * for (int i = 0; i < lambda.getCols(); i++) { double norm =
			 * getLambdaCol(i).normL2(); for (int j = 0; j < lambda.getRows();
			 * j++) { setLambda(j, i, getLambda(j, i) / Math.sqrt(norm)); } }
			 * 
			 * for (int i = 0; i < lambda.getRows(); i++) { double norm =
			 * getLambdaRow(i).normL2(); for (int j = 0; j < lambda.getCols();
			 * j++) { lambda.setElementValue(i / 3, j, lambda.getElement(i / 3,
			 * j) / Math.sqrt(norm)); } }
			 */

			SimpleMatrix oldW = new SimpleMatrix(W);
			computeW();
			decomposeW();
			setProjectionMatrices();
			setWorldPoints();

			/*
			 * for (int i = 0; i < projectionMatrices.size(); i++) { for (int j
			 * = 0; j < worldPoints.size(); j++) { SimpleVector p3 =
			 * projectionMatrices.get(i).getSubRow(2, 0, 4); SimpleVector wj =
			 * worldPoints.get(j).getAbstractVector(); setLambda(i, j,
			 * SimpleOperators.multiplyInnerProd(p3, wj)); } }
			 * 
			 * computeW(); decomposeW(); setProjectionMatrices();
			 * setWorldPoints();
			 */

			SimpleMatrix delta = SimpleOperators.subtract(W, oldW);
			DecompositionSVD svdDelta = new DecompositionSVD(delta);
			DecompositionSVD svdM = new DecompositionSVD(oldW);
			diff = svdDelta.norm2() / svdM.norm2();
		}

	}

	/**
	 * factorizes the measurement matrix W to motion R and shape S.
	 */
	private void decomposeW() {
		DecompositionSVD svd = new DecompositionSVD(W);
		R = SimpleOperators.multiplyMatrixProd(svd.getU(), svd.getS());
		S = svd.getV();
	}

	/**
	 * extracts projection matrices of R
	 */
	private void setProjectionMatrices() {
		projectionMatrices = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < R.getRows(); i += 3) {
			SimpleMatrix P = R.getSubMatrix(i, 0, 3, 4);
			projectionMatrices.add(P);
		}
	}

	/**
	 * applies the inverse of T
	 */
	private void denormalizePMatrices() {
		ArrayList<SimpleMatrix> list = new ArrayList<SimpleMatrix>();
		for (SimpleMatrix P : projectionMatrices) {
			P = new SimpleMatrix(SimpleOperators.multiplyMatrixProd(
					normalizationMatrices.get(projectionMatrices.indexOf(P))
							.inverse(InversionType.INVERT_QR), P));
			list.add(P);
		}
		projectionMatrices = new ArrayList<SimpleMatrix>(list);
	}

	/**
	 * extracts world points of shape S
	 */
	private void setWorldPoints() {
		worldPoints = new ArrayList<PointND>();
		for (int i = 0; i < S.getCols(); i++) {
			SimpleVector col = S.getCol(i);
			PointND p = new PointND(col.getElement(0)/col.getElement(3), col.getElement(1)/col.getElement(3),
					col.getElement(2)/col.getElement(3));
			worldPoints.add(p);
		}
	}

	/**
	 * 
	 * @return projectionMatrices
	 */
	public ArrayList<SimpleMatrix> getProjectionMatrices() {
		return projectionMatrices;
	}

	/**
	 * returns specific projection matrix at frame no. slice
	 * 
	 * @param slice
	 * @return projectionMatrices.get(slice)
	 */
	public SimpleMatrix getProjectionMatrix(int slice) {
		return projectionMatrices.get(slice);
	}

	/**
	 * 
	 * @return world points
	 */
	public ArrayList<PointND> getWorldPoints() {
		return worldPoints;
	}

	/**
	 * 
	 * @return all image points of all frames
	 */
	public ArrayList<ArrayList<PointND>> getImagePoints() {
		return imagePoints;
	}

	/**
	 * 
	 * @return number of frames
	 */
	public int getNumberOfFrames() {
		return numberOfFrames;
	}

	/**
	 * 
	 * @return number of points
	 */
	public int getNumberOfPoints() {
		return numberOfPoints;
	}

	/**
	 * 
	 * @param i
	 * @param j
	 * @return returns value of W at row = i and col = j
	 */
	public double getValueAt(int i, int j) {
		return lambda.getElement(i / 3, j) * M.getElement(i, j);
	}

	/**
	 * 
	 * @param i
	 * @param j
	 * @return returns lambda of W at row = i and col = j
	 */
	public double getLambda(int i, int j) {
		return lambda.getElement(i / 3, j);
	}

	/**
	 * sets lambda
	 * 
	 * @param i
	 * @param j
	 * @param value
	 */
	public void setLambda(int i, int j, double value) {
		lambda.setElementValue(i / 3, j, value);
	}

	/**
	 * 
	 * @param col
	 * @return single column of W
	 */
	public SimpleVector getCol(int col) {
		SimpleVector ret = new SimpleVector(W.getRows());
		for (int i = 0; i < W.getRows(); i++) {
			ret.setElementValue(i, getValueAt(i, col));
		}
		return ret;
	}

	/**
	 * 
	 * @param startRow
	 * @return row triplet containg startRow, startRow + 1 and startRow + 2
	 *         linearized
	 */
	public SimpleVector getRowTriplet(int startRow) {
		SimpleVector ret = new SimpleVector(W.getCols() * 3);
		for (int i = 0; i < W.getCols(); i++) {
			for (int j = startRow; j < startRow + 3; j++) {
				ret.setElementValue(3 * i + j - startRow, getValueAt(j, i));
			}
		}
		return ret;
	}

	/**
	 * 
	 * @param col
	 * @return column of lambda
	 */
	public SimpleVector getLambdaCol(int col) {
		SimpleVector ret = new SimpleVector(lambda.getRows());
		for (int i = 0; i < lambda.getRows(); i++) {
			ret.setElementValue(i, getValueAt(i, col));
		}
		return ret;
	}

	/**
	 * 
	 * @param row
	 * @return row of lambda
	 */
	public SimpleVector getLambdaRow(int row) {
		SimpleVector ret = new SimpleVector(lambda.getRows());
		for (int i = 0; i < lambda.getCols(); i++) {
			ret.setElementValue(i, getValueAt(row, i));
		}
		return ret;
	}

	private static PointND compute2Dfrom3D(PointND point3D, Projection pMatrix) {

		// Compute coordinates in projection data.
		SimpleVector homogeneousPoint = SimpleOperators.multiply(pMatrix
				.computeP(), new SimpleVector(point3D.get(0), point3D.get(1),
				point3D.get(2), 1.0));
		// Do forward projection to 2D coordinates
		double coordU = homogeneousPoint.getElement(0)
				/ homogeneousPoint.getElement(2);
		double coordV = homogeneousPoint.getElement(1)
				/ homogeneousPoint.getElement(2);

		return new PointND(coordU, coordV);
	}

	public static void main(String[] args) {
		ArrayList<PointND> world = new ArrayList<PointND>();
		world.add(new PointND(10, 20, 30));
		world.add(new PointND(11, 19, 25));
		world.add(new PointND(12, 18, 20));
		world.add(new PointND(13, 17, 15));
		world.add(new PointND(14, 16, 10));
		world.add(new PointND(15, 15, 5));
		world.add(new PointND(16, 14, 0));
		world.add(new PointND(17, 13, -5));
		CONRAD.setup();
		ArrayList<ArrayList<PointND>> image = new ArrayList<ArrayList<PointND>>();
		for (int i = 0; i < Configuration.getGlobalConfiguration()
				.getGeometry().getProjectionMatrices().length; i++) {
			ArrayList<PointND> list = new ArrayList<PointND>();
			for (int j = 0; j < world.size(); j++) {
				list.add(compute2Dfrom3D(world.get(j), Configuration
						.getGlobalConfiguration().getGeometry()
						.getProjectionMatrix(i)));
			}
			image.add(list);
		}
		
		Factorization test = new Factorization(image);
		
	}

}
