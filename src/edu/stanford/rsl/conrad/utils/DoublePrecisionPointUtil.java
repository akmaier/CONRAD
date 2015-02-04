package edu.stanford.rsl.conrad.utils;

import ij.process.ImageProcessor;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;

public abstract class DoublePrecisionPointUtil {


	/*
	 * PCA:
	 *  cov = a * a'
	 *  eiv = eigen(cov)
	 *  sort eiv system by eigenvalues, largest first.
	 *  diag = diagonal matrix of eigenvalues
	 *  psi = a' * eigenvectors * diag
	 *  eigenimages = psi'
	 */



	/**
	 * Compute the geometric center of a set of points
	 * @param list the set of points
	 * @return the geometric center
	 */
	public static PointND getGeometricCenter(ArrayList<PointND> list){
		int dim = list.get(0).getDimension();
		double [] temp = new double [list.get(0).getDimension()];
		for (int i = 0; i < list.size(); i++){
			for (int j = 0; j < dim; j++){
				temp[j] += list.get(i).get(j);
			}
		}
		for (int j = 0; j < dim; j++){
			temp[j] /= list.size();
		}
		return new PointND(temp);
	}

	/**
	 * Compute the standard deviation of a set of points
	 * @param list the set of points
	 * @return the geometric center
	 */
	public static PointND getStandardDeviation(ArrayList<PointND> list){
		int dim = list.get(0).getDimension();
		PointND center = getGeometricCenter(list);
		double [] temp = new double [list.get(0).getDimension()];
		for (int i = 0; i < list.size(); i++){
			for (int j = 0; j < dim; j++){
				temp[j] += Math.pow(list.get(i).get(j)-center.get(j),2);
			}
		}
		for (int j = 0; j < dim; j++){
			temp[j] = Math.sqrt(temp[j]) / list.size();
		}
		return new PointND(temp);
	}


	/**
	 * Extract points from an ImageProcessor which exceed a certain value
	 * 
	 * @param houghSpace the ImageProcessor
	 * @param offset the threshold for extraction
	 * @return the list of candidate points
	 */
	public static ArrayList<PointND> extractCandidatePoints(ImageProcessor houghSpace, double offset){
		ArrayList<PointND> candidate = new ArrayList<PointND>();
		for (int j = 0; j< houghSpace.getHeight(); j++){
			for (int i = 0; i< houghSpace.getWidth(); i++){
				if (houghSpace.getPixelValue(i, j) > offset) {
					PointND point = new PointND(i, j);
					candidate.add(point);					
				}
			}
		}
		return candidate;
	}

	/**
	 * Extracts cluster centers from an ordered List of points. Points must be ordered first with respect to x, then to y coordinate. Algorithm assumes that only one point may appear in the same row, i.e.,  all clusters must be separable via the y direction.
	 * A cluster center is then computed as the geometric center of the points in the same cluster. Algorithm is fast, but very restricted.
	 * @param pointList the list of candidate points
	 * @param distance the minimal distance between clusters
	 * @return the list of cluster centers
	 */
	public static ArrayList<PointND> extractClusterCenter(ArrayList<PointND> pointList, double distance){
		ArrayList<PointND> centerPoint = new ArrayList<PointND>();
		while (pointList.size() > 0){
			PointND reference = pointList.get(0);
			ArrayList<PointND> currentSubset = new ArrayList<PointND>();
			//currentSubset.add(reference);
			for (int i = 0; i < pointList.size(); i++){
				PointND current = pointList.get(i);
				if (current.euclideanDistance(reference) < distance){
					currentSubset.add(current);
					pointList.remove(i);
					i--;
				} else {
					// points are ordered first in x and then in y direction
					// hence, end of current cluster if more than distance away in y direction
					if (Math.abs(reference.get(1) - current.get(1)) > distance) break;
				}
			}
			centerPoint.add(getGeometricCenter(currentSubset));
		}
		return centerPoint;
	}


	/**
	 * Computes the total distance between two list of points.
	 * We assume that the respectively same entry of each list refers to the point to be compared in the other list.
	 * @param list1 the one list
	 * @param list2 the other list
	 * @return the sum of all euclidian distances.
	 */
	public static double computePointWiseDifference(ArrayList<PointND> list1, ArrayList<PointND> list2){
		double revan =0;
		for (int i=0;i<list1.size();i++){
			revan += list1.get(i).euclideanDistance(list2.get(i));
		}
		return revan;
	}

	/**
	 * Transforms a list of given points and returns them as new instances in a new list of points.
	 * @param list the list
	 * @param t the transform
	 * @return the new list of points
	 */
	public static ArrayList<PointND> transformPoints(ArrayList<PointND> list, Transform t){
		ArrayList<PointND> revan = new ArrayList<PointND>();
		for (int i=0; i<list.size();i++){
			PointND newP = t.transform(list.get(i));
			revan.add(newP);
		}
		return revan;
	}

	/**
	 * Estimates an optimal rotation transform to transform list1 onto list2.
	 * @param list1 the first list of points
	 * @param list2 the second list of points
	 * @param maxAngle the maximal angle that is searched (in radians)
	 * @param iterations the maximal number of iterations per step.
	 * @return the scale rotation
	 */
	public static ScaleRotate estimateRotation(ArrayList<PointND> list1, ArrayList<PointND> list2, double maxAngle, int iterations){
		double angleX = 0;
		double angleY = 0;
		double angleZ = 0;
		ScaleRotate transform = new ScaleRotate(Rotations.createRotationMatrix(angleX, angleY, angleZ));
		for (int i = 0; i < iterations; i++){
			double angleMinX = -maxAngle;
			double angleMaxX = maxAngle;
			angleX = 0;
			double errorLeft = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleMinX, angleY, angleZ))), 
									list2);
			double errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
									list2);
			double errorRight = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleMaxX, angleY, angleZ))), 
									list2);
			for (int j = 0; j <iterations; j++){
				if (errorLeft < errorRight){
					errorRight = errorCenter;
					angleMaxX = angleX;
					angleX = (angleX+angleMinX) / 2.0;
				} else {
					errorLeft = errorCenter;
					angleMinX = angleX;
					angleX = (angleX+angleMaxX) / 2.0;

				}	
				errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
						DoublePrecisionPointUtil.transformPoints(
								list1, new ScaleRotate(
										Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
										list2);
			}
			double angleMinY = -maxAngle;
			double angleMaxY = maxAngle;
			angleY = 0;
			errorLeft = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleMinY, angleZ))), 
									list2);
			errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
									list2);
			errorRight = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleMaxY, angleZ))), 
									list2);
			for (int j = 0; j <iterations; j++){
				if (errorLeft < errorRight){
					errorRight = errorCenter;
					angleMaxY = angleY;
					angleY = (angleY+angleMinY) / 2.0;
				} else {
					errorLeft = errorCenter;
					angleMinY = angleY;
					angleY = (angleY+angleMaxY) / 2.0;
				}
				errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
						DoublePrecisionPointUtil.transformPoints(
								list1, new ScaleRotate(
										Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
										list2);
			}
			double angleMinZ = -maxAngle;
			double angleMaxZ = maxAngle;
			angleZ = 0;
			errorLeft = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleY, angleMinZ))), 
									list2);
			errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
									list2);
			errorRight = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, new ScaleRotate(
									Rotations.createRotationMatrix(angleX, angleY, angleMaxZ))), 
									list2);
			for (int j = 0; j <iterations; j++){
				if (errorLeft < errorRight){
					errorRight = errorCenter;
					angleMaxZ = angleZ;
					angleZ = (angleZ+angleMinZ) / 2.0;
				} else {
					errorLeft = errorCenter;
					angleMinZ = angleZ;
					angleZ = (angleZ+angleMaxZ) / 2.0;
				}
				errorCenter = DoublePrecisionPointUtil.computePointWiseDifference(
						DoublePrecisionPointUtil.transformPoints(
								list1, new ScaleRotate(
										Rotations.createRotationMatrix(angleX, angleY, angleZ))), 
										list2);
			}	
		}
		transform = new ScaleRotate(Rotations.createRotationMatrix(angleX, angleY, angleZ));
		if (CONRAD.DEBUGLEVEL > 0) {
			double finalError = DoublePrecisionPointUtil.computePointWiseDifference(
					DoublePrecisionPointUtil.transformPoints(
							list1, transform), list2);
			System.out.println(General.toDegrees(angleX) + " " + General.toDegrees(angleY) + " " + General.toDegrees(angleZ) + " " + finalError / list1.size());
		}
		return transform;
	}


}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/