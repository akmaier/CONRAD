package edu.stanford.rsl.conrad.reconstruction;


import ij.process.FloatProcessor;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionUtil;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * The VOIBasedReconstructionFilter is an implementation of the backprojection which employs a volume-of-interest (VOI) to
 * speed up reconstruction. Only voxels within the VOI will be regarded in the backprojection step. Often this can save up to 30 to 40 % in computation time
 * as volumes are usually described as boxes but the VOI is just a cylinder.
 * 
 * This version of the reconstruction algorithm applies the motion field stored in 4D_SPLINE_LOCATION before the backprojection.
 * 
 * @author akmaier
 *
 */
public class MotionCompensatedVOIBasedReconstructionFilterMatrix extends VOIBasedReconstructionFilter {


	/**
	 * 
	 */
	private static final long serialVersionUID = 3728916582817241495L;
	protected MotionField motionField;
	private PointND[][][][] pointCorrespondences;

	/**
	 * @return the pointCorrespondences
	 */
	public PointND[][][][] getPointCorrespondences() {
		return pointCorrespondences;
	}

	/**
	 * @param pointCorrespondences the pointCorrespondences to set
	 */
	public void setPointCorrespondences(
			PointND [][][][] pointCorrespondences) {
		this.pointCorrespondences = pointCorrespondences;
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		motionField = null;
	}

	protected synchronized void initialize(Grid2D projection){
		if (!init){
			super.initialize(projection);
			if (motionField == null) {
				System.out.println("loading Motion Field");
				motionField = MotionUtil.get4DSpline();
			}
		}
	}

	public void backproject(Grid2D projection, int projectionNumber){
		int count = 0;
		//System.out.println(projectionVolume);
		if ((!init)){
			initialize(projection);
		}
		// Constant part of distance weighting (D^2) + additional weighting for arbitrary scan ranges
		double D =  getGeometry().getSourceToDetectorDistance();
		double scalingFactor = (10*D*D * 2* Math.PI * getGeometry().getPixelDimensionX()/ getGeometry().getNumProjectionMatrices());
		
		if (pointCorrespondences == null) pointCorrespondences = new PointND[getGeometry().getNumProjectionMatrices()][maxK][maxJ][maxI];
		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(), projection.getHeight(), projection.getBuffer(), null);
		//ImageProcessor currentProjection = projection;
		int p = projectionNumber;
		double[] voxel = new double [4];
		SimpleMatrix mat = getGeometry().getProjectionMatrix(p).computeP();
		SimpleVector centerTranlation = null;
		if (motionField instanceof AnalyticPhantom4D){
			AnalyticPhantom4D phantom = (AnalyticPhantom4D) motionField;
			centerTranlation = SimpleOperators.add(phantom.getMax().getAbstractVector(), phantom.getMin().getAbstractVector()).dividedBy(2);
		}
		voxel[3] = 1;
		double [] times = new double [getGeometry().getNumProjectionMatrices()];
		for (int i=0; i< times.length; i++){
			times [i]= ((double)i) / (getGeometry().getNumProjectionMatrices()-1.0);
		}
		System.out.println("Processing projection " + p);
		if (mat != null){
			boolean nanHappened = false;
			for (int k = 0; k < maxK ; k++){ // for all slices
				if (debug) System.out.println("here: " + " " + k);
				voxel[2] = (this.getGeometry().getVoxelSpacingZ() * (k)) - offsetZ;
				for (int j = 0; j < maxJ; j++){ // for all voxels
					voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
					for (int i=0; i < maxI; i++){ // for all lines
						voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;
						// compute real world coordinates in homogenious coordinates;
						boolean project = true;
						if (useVOImap){
							if (voiMap != null){
								project = voiMap[i][j][k];
							}
						}
						if (project){			
							PointND point = new PointND(voxel[0], voxel[1], voxel[2]);
							if (centerTranlation !=null){
								point.getAbstractVector().add(centerTranlation);
							}
							// compute compensated position
							PointND timePoint = pointCorrespondences[p][k][j][i];
							if (timePoint == null){
								ArrayList<PointND> timePoints = motionField.getPositions(point, 0, times);
								for (int h = 0; h < timePoints.size(); h++){
									pointCorrespondences[h][k][j][i]= timePoints.get(h); 
								}
								timePoint = timePoints.get(p);
							}
							point = timePoint;
							if (centerTranlation != null){
								point.getAbstractVector().subtract(centerTranlation);
							}
							
							// Compute coordinates in projection data.
							SimpleVector homogeniousPoint = SimpleOperators.multiply(mat, new SimpleVector(point.get(0), point.get(1), point.get(2), voxel[3]));
							// Transform to 2D coordinates
							double coordX = homogeniousPoint.getElement(0) / homogeniousPoint.getElement(2);
							double coordY = homogeniousPoint.getElement(1) / homogeniousPoint.getElement(2);
							// back project
							double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY) / (homogeniousPoint.getElement(2)*homogeniousPoint.getElement(2));
							if (Double.isNaN(increment)){
								nanHappened = true;
								if (count < 10) System.out.println("NAN Happened at i = " + i + " j = " + j + " k = " + k + " projection = " + projectionNumber + " x = " + coordX + " y = " + coordY  );
								increment = 0;
								count ++;
							}
							updateVolume(i, j, k, scalingFactor*increment);
						}
					}
				}
			}
			if (nanHappened) {
				throw new RuntimeException("Encountered NaN in projection!");
			}
			if (debug) System.out.println("done with projection");
		}
	}

	@Override
	public String getName() {
		return "Motion-compensated CPU-based Backprojector";
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}


	@Override
	public String getToolName() {
		return "Motion-compensated VOI-based Backprojector";
	}

	@Override
	protected synchronized void initializeVOIMap() {
		// TODO Auto-generated method stub
		super.initializeVOIMap();
	}
	
	
	/**
	 * @return the motionField
	 */
	public MotionField getMotionField() {
		return motionField;
	}


	/**
	 * @param motionField the motionField to set
	 */
	public void setMotionField(MotionField motionField) {
		this.motionField = motionField;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/