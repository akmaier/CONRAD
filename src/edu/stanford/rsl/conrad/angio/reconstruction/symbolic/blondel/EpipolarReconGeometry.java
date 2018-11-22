/*
 * Copyright (C) 2010-2018 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel;

import java.util.ArrayList;
import java.util.Arrays;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix.InversionType;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.Skeleton;
import edu.stanford.rsl.conrad.angio.graphs.skeleton.util.SkeletonBranch;
import edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util.BlondelDynamicProgrammingDijkstra;
import edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util.DynamicCorrespondence;
import edu.stanford.rsl.conrad.angio.reconstruction.symbolic.blondel.util.Fuse;

public class EpipolarReconGeometry {

	public Projection[] pMatrices = null;
	private double maxDistance = 1.5;
	private double alpha = 2500;
	private ArrayList<DynamicCorrespondence> correspondences;
	private ArrayList<DynamicCorrespondence> evaluatedCorrespondences;
	private ArrayList<Skeleton> centerlineSkeleton;
	private ArrayList<PointND> corrAfterDijkst;
	public int withoutThis;
	private int dh;
	private int dl;
	private boolean getEvaluated;
	private boolean FusingMethodBlondel;
	private Grid3D costmap;

	private double REPROJECTION_ERROR = 2.5d; // in mm
	private double SUPPRESSION_RADIUS = 1.5d;//0.46d; // in mm
	
	
	public EpipolarReconGeometry(Projection[] pMat, ArrayList<Skeleton> centerlines, boolean getsEvaluated, boolean FuseBlondel) {
		this.pMatrices = pMat;
		this.centerlineSkeleton = centerlines;
		this.corrAfterDijkst = new ArrayList<PointND>();
		this.getEvaluated = getsEvaluated;
		this.FusingMethodBlondel = FuseBlondel;
	}

	public void setParameters(double maxDist, double alph, int without, int dl_, int dh_) {
		this.maxDistance = maxDist;
		this.alpha = alph;
		this.withoutThis = without;
		this.dh = dh_;
		this.dl = dl_;

	}

	public void setParameters(double maxDist, double alph, int dl_, int dh_) {
		this.maxDistance = maxDist;
		this.alpha = alph;
		this.dh = dh_;
		this.dl = dl_;
	}

	public ArrayList<PointND> getCorrAfterDijkst() {
		return corrAfterDijkst;
	}

	public void setCorrAfterDijkst(ArrayList<PointND> corrAfterDijkst) {
		this.corrAfterDijkst = corrAfterDijkst;
	}

	public void evaluate() {

		SimpleMatrix FundamentalMatrix;
		this.correspondences = new ArrayList<DynamicCorrespondence>();
		this.evaluatedCorrespondences = new ArrayList<DynamicCorrespondence>();
		System.out.println("MatchingStarted");
		ArrayList<ArrayList<PointND>> bestPoints = new ArrayList<ArrayList<PointND>>();
		int counter = 0;
		for (int i = 0; i < centerlineSkeleton.size(); i++) {
			for (int j = 0; j < centerlineSkeleton.size(); j++) {
				if (getEvaluated) {
					if (i != j && i < j && i != withoutThis && j != withoutThis) {
						System.out.println("Reference view combination Nr: " + counter);
						// Calculate the fundamental matrix
						FundamentalMatrix = calculateFundamentalMatrix(i, j);
						// Find all possible correspondences using the
						// fundamental matrix
						getPossibleCorrespondences(i, j, FundamentalMatrix);
						// Calculate the error of the correspondences by
						// comparing them to other views
						evaluateCorrespondences(i, j);
						// Calculating the connectivities between the
						// correspondences
						internalEnergy();
						// Finding the optimal set of correspondences
						BlondelDynamicProgrammingDijkstra shortestPath = new BlondelDynamicProgrammingDijkstra(evaluatedCorrespondences, centerlineSkeleton.get(i),
								alpha);
						shortestPath.run();
						bestPoints.add(shortestPath.getOptimized3DPoints());
						correspondences.clear();
						// test(bestPoints.get(counter));
						counter++;
					}
				} else {
					if (i != j && i < j) {
						System.out.println("Reference view combination Nr: " + counter);
						// Calculate the fundamental matrix
						FundamentalMatrix = calculateFundamentalMatrix(i, j);
						// Find all possible correspondences using the
						// fundamental matrix
						getPossibleCorrespondences(i, j, FundamentalMatrix);
						// Calculate the error of the correspondences by
						// comparing them to other views
						evaluateCorrespondences(i, j);
						// Calculating the connectivities between the
						// correspondences
						internalEnergy();
						// Finding the optimal set of correspondences
						BlondelDynamicProgrammingDijkstra shortestPath = 
								new BlondelDynamicProgrammingDijkstra(
										evaluatedCorrespondences, centerlineSkeleton.get(i), alpha);
						shortestPath.run();
						bestPoints.add(shortestPath.getOptimized3DPoints());
						correspondences.clear();
						counter++;
					}
				}
			}
		}
		System.out.println("Fusing the asymetric matching");
		
		if (FusingMethodBlondel == true){
			this.corrAfterDijkst = fuseBlondel(bestPoints);
		}else{
			this.corrAfterDijkst = exhaustiveRefinementStep(bestPoints);
		}
	}

	/**
	 * Performs the iterative refinement using the error-tolerances specified in the class members.
	 * The underlying idea is, that we should use all segmented (i.e. skeleton pixels) not only the ones 
	 * in the selected / extraordinary view. In a perfect scenario, the reconstructed points would explain all 
	 * of the skeleton slab voxels. In the real case, however, segmentation imperfections (too few or too many pixels)
	 * and the neighborhood smoothness constraint lead to erroneous reconstructions. 
	 * In order to improve this condition we propose two things:
	 * a) reconstructed points should agree with the majority of observations (segmentations in our case). This requirement 
	 * can be checked by computing the second largest reprojection error.
	 * b) the majority of skeleton points should be explained by the reconstruction, i.e. at least one reconstructed is 
	 * closest to this point ( and also closer than the accepted threshold). If a point has not been the closest match 
	 * but there is a reconstructed point close to the suggested 3D point of the not-considered candidate, the two 
	 * 3D points are averaged. If the there is no close 3D point present yet, the suggested 3D point is added to the 
	 * reconstruction.
	 * @return
	 */
	private ArrayList<PointND> exhaustiveRefinementStep(ArrayList<ArrayList<PointND>> reconstructedPoints){
		// exhaustive combination
		int numUsedBeforeMatching = 0;
		int totalPoints = 0;
		
		for(int v = 0; v < reconstructedPoints.size(); v++){
			System.out.println("Refining view: "+String.valueOf(v+1));
			ArrayList<PointND> recons = reconstructedPoints.get(v);
			ArrayList<PointND> refinedReco = new ArrayList<PointND>();
			for(int i = 0; i < recons.size(); i++){
				// if the current point has not been used in the reconstruction yet
				// check if it is in agreement with the rest of the views (median error)
				PointND recoProposal = recons.get(i);
				double err = calculatePointError(recoProposal);
				// if the current point is in agreement with the rest of the views
				// check if there is a point close by in the reconstruction already
				if(err < REPROJECTION_ERROR){
					refinedReco.add(recoProposal);
				}
			}
			numUsedBeforeMatching += refinedReco.size();
			totalPoints += recons.size();
			reconstructedPoints.set(v, refinedReco);
		}// end refinement one single view
		System.out.println("After suppresion using "+
				String.valueOf(numUsedBeforeMatching)+" out of "+ Integer.valueOf(totalPoints)+".");
		
		// matching and suppression
		ArrayList<PointND> finalRecon = new ArrayList<PointND>();
		for(int v = 0; v < reconstructedPoints.size(); v++){
			ArrayList<PointND> recons = reconstructedPoints.get(v);
			for(int i = 0; i < recons.size(); i++){
				PointND recoProposal = recons.get(i);
				ArrayList<PointND> same = new ArrayList<PointND>();
				same.add(recoProposal);
				for(int j = v+1; j < reconstructedPoints.size(); j++){
					ArrayList<PointND> reduced = new ArrayList<PointND>();
					for(int k = 0; k < reconstructedPoints.get(j).size(); k++){
						PointND testPoint = reconstructedPoints.get(j).get(k);
						double eucDist = testPoint.euclideanDistance(recoProposal);
						if(eucDist < SUPPRESSION_RADIUS){
							same.add(testPoint);
						}else{
							reduced.add(testPoint);
						}
					}
					reconstructedPoints.set(j, reduced);
				}
				SimpleVector average = new SimpleVector(3);
				for(int j = 0; j < same.size(); j++){
					average.add(same.get(j).getAbstractVector());
				}
				average.divideBy(same.size());
				finalRecon.add(new PointND(average.copyAsDoubleArray()));					
			}

		}
		System.out.println("After exhaustive merging using "+ 
				String.valueOf(finalRecon.size())+" out of "+ Integer.valueOf(totalPoints)+".");
				
		return finalRecon;
	}
	
	/**
	 * Calculates the error of a 3D point given the segmentations / observations in all views.
	 * The error is defined as the second largest reprojection error in all segmentations.
	 * The definition allows for erroneous segmentations, 
	 * meaning vessel structures not segmented in several views or erroneously segmented structures, such as parts of 
	 * catheters.
	 * @param p3D
	 * @return
	 */
	private double calculatePointError(PointND p3D){
		float[] err = new float[pMatrices.length];
		int idx = 0;
		for(int v = 0; v < pMatrices.length; v++){
			SimpleVector projectedInView = new SimpleVector(2);
			pMatrices[v].project(p3D.getAbstractVector(), projectedInView);
			err[idx] = InterpolationOperators.interpolateLinear(costmap.getSubGrid(v),
					projectedInView.getElement(0), projectedInView.getElement(1));
			idx++;
		}
		Arrays.sort(err);
		
		return err[err.length-2];	
	}
	
	// compares the 3D points of the matching of different views and takes the
	// average point due to the neighbors.
	// also helps against outliers
	private ArrayList<Fuse> markAndFuse(ArrayList<ArrayList<PointND>> bestPoints) {
		ArrayList<Fuse> fused = new ArrayList<Fuse>();
		for (int i = 0; i < bestPoints.size(); i++) {
			for (int j = 0; j < bestPoints.get(i).size(); j++) {
				Fuse newFuse = new Fuse(bestPoints.get(i).get(j), i);
				fused.add(newFuse);
			}
		}
		return fused;
	}

	@SuppressWarnings("unused")
	private ArrayList<PointND> fuseNew(ArrayList<ArrayList<PointND>> bestPoints) {
		ArrayList<Fuse> fused = markAndFuse(bestPoints);
		ArrayList<PointND> output = new ArrayList<PointND>();
		for (int j = 0; j < fused.size(); j++) {
			Fuse compareThis = fused.get(j);
			ArrayList<PointND> buffer = new ArrayList<PointND>();

			for (int q = 0; q < bestPoints.size(); q++) {
				double minDist = Double.MAX_VALUE;
				Fuse minFuse = new Fuse(new PointND(1000, 1000, 1000));
				for (int k = 0; k < fused.size(); k++) {
					if (q != compareThis.getNumberOfResponses()) {
						Fuse withThis = fused.get(k);
						if (withThis.getNumberOfResponses() == q) {
							double dist = compareThis.getPoint().euclideanDistance(withThis.getPoint());
							if (dist < minDist) {
								minDist = dist;
								minFuse = withThis;
							}
						}
					}

				}
				if (minDist < 5) {
					buffer.add(minFuse.getPoint());
				}
			}
			PointND newPoint = new PointND(0, 0, 0);
			if (2 < buffer.size()) {
				for (int r = 0; r < buffer.size(); r++) {
					double[] newPointCoor = newPoint.getCoordinates();
					double[] buffPointCoor = buffer.get(r).getCoordinates();
					newPoint.set(0, newPointCoor[0] + buffPointCoor[0]);
					newPoint.set(1, newPointCoor[1] + buffPointCoor[1]);
					newPoint.set(2, newPointCoor[2] + buffPointCoor[2]);
					newPointCoor = newPoint.getCoordinates();
					if (r == buffer.size() - 1) {
						newPoint.set(0, newPointCoor[0] / buffer.size());
						newPoint.set(1, newPointCoor[1] / buffer.size());
						newPoint.set(2, newPointCoor[2] / buffer.size());
					}
				}
				output.add(newPoint);
			}
		}
		output = removeRedundant(output);
		return output;
	}
	//remove redundant data after fusing
	private ArrayList<PointND> removeRedundant(ArrayList<PointND> input) {
		System.out.println("Remove redundant data. sampling size 0.03");
		ArrayList<PointND> output = new ArrayList<PointND>();
		while (input.isEmpty() == false) {
			PointND compare = input.remove(0);
			double[] coorComp = compare.getCoordinates();
			for (int i = 0; i < input.size(); i++) {
				PointND withThis = input.get(i);
				double[] coorWithThis = withThis.getCoordinates();
				double distance = Math.abs(coorWithThis[0] - coorComp[0]) + Math.abs(coorWithThis[1] - coorComp[1])
						+ Math.abs(coorWithThis[2] - coorComp[2]);
				if (distance < 0.03) {
					input.remove(i);
				}
			}
			output.add(compare);

		}
		return output;
	}
	private ArrayList<ArrayList<Fuse>> setToFuse(ArrayList<ArrayList<PointND>> bestPoints) {
		ArrayList<ArrayList<Fuse>> fused = new ArrayList<ArrayList<Fuse>>();
		for (int i = 0; i < bestPoints.size(); i++) {
			ArrayList<Fuse> buff = new ArrayList<Fuse>();
			for (int j = 0; j < bestPoints.get(i).size(); j++) {
				PointND toFuse = bestPoints.get(i).get(j);
				Fuse converted = new Fuse(toFuse, 1);
				buff.add(converted);
			}
			fused.add(buff);
		}
		return fused;
	}

	
	private ArrayList<PointND> fuseBlondelOld(ArrayList<ArrayList<PointND>> reconstructedPoints){
		// exhaustive combination
		int totalPoints = 0;
		// matching and suppression
		ArrayList<PointND> finalRecon = new ArrayList<PointND>();
		for(int v = 0; v < reconstructedPoints.size(); v++){
			ArrayList<PointND> recons = reconstructedPoints.get(v);
			for(int i = 0; i < recons.size(); i++){
				PointND recoProposal = recons.get(i);
				ArrayList<PointND> same = new ArrayList<PointND>();
				same.add(recoProposal);
				for(int j = v+1; j < reconstructedPoints.size(); j++){
					ArrayList<PointND> reduced = new ArrayList<PointND>();
					for(int k = 0; k < reconstructedPoints.get(j).size(); k++){
						PointND testPoint = reconstructedPoints.get(j).get(k);
						double eucDist = testPoint.euclideanDistance(recoProposal);
						if(eucDist < SUPPRESSION_RADIUS){
							same.add(testPoint);
						}else{
							reduced.add(testPoint);
						}
					}
					reconstructedPoints.set(j, reduced);
				}
				SimpleVector average = new SimpleVector(3);
				for(int j = 0; j < same.size(); j++){
					average.add(same.get(j).getAbstractVector());
				}
				average.divideBy(same.size());
				finalRecon.add(new PointND(average.copyAsDoubleArray()));					
			}

		}
				
		return finalRecon;
	}
	
	private ArrayList<PointND> fuseBlondel(ArrayList<ArrayList<PointND>> bestPoints) {
		ArrayList<ArrayList<Fuse>> fused = setToFuse(bestPoints);
		if (bestPoints.size() == 1) {
			return bestPoints.get(0);
		}
		ArrayList<Fuse> output = new ArrayList<Fuse>();
		output = fused.get(0);
		// for (int i = 0; i < bestPoints.size(); i++) {
		for (int r = 1; r < fused.size(); r++) {

			ArrayList<Fuse> points3D1V = fused.get(r);
			ArrayList<Fuse> buffer = new ArrayList<Fuse>();
			while (output.isEmpty() == false) {
				Fuse point1V = output.remove(0);
				for (int k = 0; k < points3D1V.size(); k++) {
					Fuse compareWithFuse = points3D1V.remove(0);
					PointND compareWith = compareWithFuse.getPoint();
					double dist = point1V.getPoint().euclideanDistance(compareWith);
					if (dist < 5) {
						PointND newPoint = new PointND(0, 0, 0);

						double[] newPointCoor = point1V.getPoint().getCoordinates();
						double[] buffPointCoor = compareWith.getCoordinates();
						newPoint.set(0, (newPointCoor[0] + buffPointCoor[0]) / 2);
						newPoint.set(1, (newPointCoor[1] + buffPointCoor[1]) / 2);
						newPoint.set(2, (newPointCoor[2] + buffPointCoor[2]) / 2);
						int soFar = point1V.getNumberOfResponses();
						Fuse newFuse = new Fuse(newPoint, soFar + 1);
						buffer.add(newFuse);
					} else {
						Fuse minFuse = new Fuse(compareWith, 1);
						buffer.add(minFuse);
						buffer.add(point1V);
					}
				}
				
			}
			output.addAll(buffer);
		}
		ArrayList<PointND> outputPND = setToPointND(output);
		return outputPND;

	}

	private ArrayList<PointND> setToPointND(ArrayList<Fuse> fused) {
		ArrayList<PointND> points = new ArrayList<PointND>();
		for (int j = 0; j < fused.size(); j++) {
			Fuse toConvert = fused.get(j);
			if (toConvert.getNumberOfResponses() > 0) {
				PointND converted = toConvert.getPoint();
				points.add(converted);
			}
		}
		ArrayList<PointND> pointsReduced = new ArrayList<PointND>();
		for(int i = 0; i < points.size(); i++){
			PointND p = points.get(i);
			double minDist = Double.MAX_VALUE;
			for(int j = 0; j < pointsReduced.size(); j++){
				minDist = Math.min(minDist, p.euclideanDistance(pointsReduced.get(j)));
			}
			if(minDist > 0.1){
				pointsReduced.add(p);
			}
		}
		return pointsReduced;
	}

	// calculates the connectivities
	public void internalEnergy() {
		for (int pointer = 0; pointer < evaluatedCorrespondences.size() - 1; pointer++) {
			DynamicCorrespondence corr1 = evaluatedCorrespondences.get(pointer);
			DynamicCorrespondence corr2 = evaluatedCorrespondences.get(pointer + 1);
			ArrayList<PointND> pointsSecondView1 = corr1.getCorrespondences();
			ArrayList<PointND> pointsSecondView2 = corr2.getCorrespondences();
			double[][] internalconnectivity = new double[pointsSecondView1.size()][pointsSecondView2.size()];
			for (int comb = 0; comb < pointsSecondView1.size(); comb++) {
				for (int comb2 = 0; comb2 < pointsSecondView2.size(); comb2++) {
					PointND q1 = pointsSecondView1.get(comb);
					PointND q2 = pointsSecondView2.get(comb2);
					double distance = q1.euclideanDistance(q2);
					double connectivity = 0;
					if (distance < dl) {
						connectivity = 0;
					} else if (dl < distance && distance < dh) {
						connectivity = (distance - dl) / (dh - dl);
						// connectivity = distance;
					} else {
						connectivity = 1;
					}
					internalconnectivity[comb][comb2] = connectivity;
				}
			}
			evaluatedCorrespondences.get(pointer).setConnectivity(internalconnectivity);
		}
	}

	
	public ArrayList<DynamicCorrespondence> getCorrespondences() {
		return correspondences;
	}

	public ArrayList<PointND> eraseNulls(ArrayList<PointND> toEraseIn) {
		PointND nullPoint = new PointND();
		for (int i = 0; i < toEraseIn.size(); i++) {
			while (toEraseIn.get(i).equals(nullPoint)) {
				toEraseIn.remove(i);
				System.out.println(i);
			}
		}
		return toEraseIn;
	}

	// gives us those sweet correspondences
	private void getPossibleCorrespondences(int viewNumber1, int viewNumber2, SimpleMatrix fundMat) {
		for (int branchNumber = 0; branchNumber < centerlineSkeleton.get(viewNumber1).size(); branchNumber++) {
			SkeletonBranch branch = centerlineSkeleton.get(viewNumber1).get(branchNumber);
			for (int skelBranchPoint = 0; skelBranchPoint < branch.size(); skelBranchPoint++) {
				DynamicCorrespondence dynCorr = new DynamicCorrespondence();
				ArrayList<PointND> corrPoints = new ArrayList<PointND>();
				dynCorr.setSlidesNr(new int[] { viewNumber1, viewNumber2 });
				dynCorr.setBranchNr(branchNumber);
				dynCorr.setPointNr(skelBranchPoint);
				PointND p1 = branch.get(skelBranchPoint).get2DPointND();
				dynCorr.setPoint(p1);
				// the important part: calculate all possible correspondences by
				// looking for points close the the
				// epipolar line in view 2
				corrPoints = getCorrespondenceList(p1, viewNumber2, fundMat);
				dynCorr.setCorrespondences(corrPoints);
				correspondences.add(dynCorr);
			}
		}
	}

	// draw the epipolar line and get points lying on it
	private ArrayList<PointND> getCorrespondenceList(PointND point1, int viewNumber2, SimpleMatrix fundMat) {
		ArrayList<PointND> corrPoints = new ArrayList<PointND>();
		for (int i = 0; i < centerlineSkeleton.get(viewNumber2).size(); i++) {
			SkeletonBranch branchView2 = centerlineSkeleton.get(viewNumber2).get(i);
			for (int skelBranchPointv2 = 0; skelBranchPointv2 < branchView2.size(); skelBranchPointv2++) {
				PointND p2 = branchView2.get(skelBranchPointv2).get2DPointND();
				double d = Math.abs(distanceFromEpipolarLine(point1, p2, fundMat));
				if (d < maxDistance) {
					corrPoints.add(p2);
				}
			}
		}

		return corrPoints;
	}

	// calculating the error of each point by comparing the calculated 3D point
	// with its projections in other views
	// if the point does not exist in another view an error of 5 is added.
	private void evaluateCorrespondences(int viewNumber1, int viewNumber2) {
		// go through all points
		for (int point = 0; point < correspondences.size(); point++) {
			{
				DynamicCorrespondence dynCorr = correspondences.get(point);
				ArrayList<PointND> correspondencesForAPointInView1 = dynCorr.getCorrespondences();
				ArrayList<PointND> threeDPoints = new ArrayList<PointND>();
				double[] errorList = new double[correspondencesForAPointInView1.size()];
				// go through all correspondences
				for (int pointer = 0; pointer < correspondencesForAPointInView1.size(); pointer++) {
					// calculate the 3DPoint
					PointND point3D = calculate3DPoint(dynCorr.getPoint(), correspondencesForAPointInView1.get(pointer),
							viewNumber1, viewNumber2);
					threeDPoints.add(point3D);
					double error = 0;
					// look into the other views
					for (int otherView = 0; otherView < centerlineSkeleton.size(); otherView++) {
						if (otherView != viewNumber1 && otherView != viewNumber2 && otherView != withoutThis) {
							// project the 3DPoint
							PointND point3DProjected = project(point3D, otherView);
							// get the point with the minimal distance across
							// all branches
							double distanceMIN = Double.MAX_VALUE;
							for (int checkAllBranches = 0; checkAllBranches < centerlineSkeleton.get(otherView)
									.size(); checkAllBranches++) {
								SkeletonBranch branchOtherViews = centerlineSkeleton.get(otherView)
										.get(checkAllBranches);
								for (int branchPointOtherViews = 0; branchPointOtherViews < branchOtherViews
										.size(); branchPointOtherViews++) {
									double distance = centerlineSkeleton.get(otherView).get(checkAllBranches)
											.get(branchPointOtherViews).get2DPointND()
											.euclideanDistance(point3DProjected);
									if (distance < distanceMIN) {
										distanceMIN = distance;
									}
								}
							}
							// if the point does not exist in the other view do
							// not punish too hard
							if (distanceMIN > 50) {
								distanceMIN = 50;
							}
							error += distanceMIN;
						}
					}

					errorList[pointer] = error / (centerlineSkeleton.size() - 2);
				}
				dynCorr.setErrorList(errorList);
				dynCorr.setPoints3D(threeDPoints);
				this.evaluatedCorrespondences.add(dynCorr);
			}

		}
	}

	public SimpleMatrix calculateFundamentalMatrix(int viewNumber1, int viewNumber2) {
		Projection pMat1 = pMatrices[viewNumber1];
		Projection pMat2 = pMatrices[viewNumber2];
		PointND nullVec1 = toHomogeneous(toPointND(pMatrices[viewNumber1].computeCameraCenter()));
		SimpleVector pMat2c = SimpleOperators.multiply(pMat2.computeP(),
				toSimpleVector(nullVec1));
		SimpleMatrix e2Skew = getSkewMatrix(pMat2c);
		SimpleMatrix pMat2pMat1dagger = SimpleOperators.multiplyMatrixProd(pMat2.computeP(),
				pMat1.computeP().inverse(InversionType.INVERT_SVD));
		SimpleMatrix fundamentalMat = SimpleOperators.multiplyMatrixProd(e2Skew, pMat2pMat1dagger);
		return fundamentalMat;
	}

	public double distanceFromEpipolarLine(PointND point1, PointND point2, SimpleMatrix fundamentalMat) {
		SimpleVector res1 = SimpleOperators.multiply(fundamentalMat, toSimpleVector(toHomogeneous(point1)));
		double val = SimpleOperators.multiplyInnerProd(toSimpleVector(toHomogeneous(point2)), res1);
		val /= Math.sqrt(res1.getElement(0) * res1.getElement(0) + res1.getElement(1) * res1.getElement(1));
		return val;
	}

	public PointND calculate3DPoint(PointND point1, PointND point2, int viewNumber1, int viewNumber2) {

		double x = point1.get(0);
		double y = point1.get(1);
		double xPrime = point2.get(0);
		double yPrime = point2.get(1);
		SimpleMatrix p1 = pMatrices[viewNumber1].computeP();
		SimpleMatrix p2 = pMatrices[viewNumber2].computeP();

		SimpleMatrix mat = new SimpleMatrix(4, 4);
		mat.setRowValue(0, SimpleOperators.subtract(p1.getRow(2).multipliedBy(x), p1.getRow(0)));
		mat.setRowValue(1, SimpleOperators.subtract(p1.getRow(2).multipliedBy(y), p1.getRow(1)));
		mat.setRowValue(2, SimpleOperators.subtract(p2.getRow(2).multipliedBy(xPrime), p2.getRow(0)));
		mat.setRowValue(3, SimpleOperators.subtract(p2.getRow(2).multipliedBy(yPrime), p2.getRow(1)));

		SimpleMatrix matA = mat.getSubMatrix(0, 0, 4, 3);
		SimpleVector b = mat.getCol(3).negated();

		SimpleVector solution = SimpleOperators.multiply(matA.inverse(InversionType.INVERT_SVD), b);

		return toPointND(solution);
	}

	public PointND project(PointND point3D, int viewNumber) {
		SimpleMatrix mat = pMatrices[viewNumber].computeP();
		PointND p3Dhom = toHomogeneous(point3D);
		SimpleVector projected = SimpleOperators.multiply(mat, p3Dhom.getAbstractVector());
		return toCanonical(new PointND(projected.copyAsDoubleArray()));
	}

	private PointND toPointND(SimpleVector v) {
		return new PointND(v.copyAsDoubleArray());
	}

	private PointND toHomogeneous(PointND p) {
		int n = p.getDimension();
		double[] h = new double[n + 1];
		for (int i = 0; i < n; i++) {
			h[i] = p.get(i);
		}
		h[n] = 1;
		return new PointND(h);
	}

	private SimpleVector toSimpleVector(PointND p) {
		return p.getAbstractVector();
	}

	private SimpleMatrix getSkewMatrix(SimpleVector v) {
		if (v.getLen() != 3) {
			System.out.println("Skew Matrix not defined for dimensions other than 3.");
			System.exit(-1);
		}
		SimpleMatrix skew = new SimpleMatrix(3, 3);
		// first row of skew mat
		skew.setElementValue(0, 1, -v.getElement(2));
		skew.setElementValue(0, 2, +v.getElement(1));
		// second row
		skew.setElementValue(1, 0, +v.getElement(2));
		skew.setElementValue(1, 2, -v.getElement(0));
		// third row
		skew.setElementValue(2, 0, -v.getElement(1));
		skew.setElementValue(2, 1, +v.getElement(0));
		return skew;
	}

	private PointND toCanonical(PointND h) {
		int n = h.getDimension() - 1;
		double[] c = new double[n];
		for (int i = 0; i < n; i++) {
			c[i] = h.get(i) / h.get(n);
		}
		return new PointND(c);
	}

	public void setCostMap(Grid3D cm){
		this.costmap = cm;
	}
	
	public void setExhaustiveReconParameters(double reprojErr, double suppressionRad){
		this.REPROJECTION_ERROR = reprojErr;
		this.SUPPRESSION_RADIUS = suppressionRad;
	}
}
