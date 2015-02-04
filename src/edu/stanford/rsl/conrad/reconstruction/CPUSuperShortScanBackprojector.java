package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import ij.process.FloatProcessor;

/**
 * Implementation of the backprojector which is required for Noo's super-short-scan reconstruction. Note that the distance weight in the weighted sum of the backprojection is different.
 * @author akmaier
 *
 */
public class CPUSuperShortScanBackprojector extends VOIBasedReconstructionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1248364198978750688L;
	protected boolean mapped = false;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		mapped = false;
	}


	@Override
	public void backproject(Grid2D projection, int projectionNumber) {
		int count = 0;
		//System.out.println(projectionVolume);
		if (!init){
			initialize(projection);
		}
		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(), projection.getHeight(), projection.getBuffer(), null);
		//ImageProcessor currentProjection = projection;
		int p = projectionNumber;
		double[] voxel = new double [4];
		double[] homogeniousPointi = new double[3];
		double[] homogeniousPointj = new double[3];
		double[] homogeniousPointk = new double[3];
		double[][] updateMatrix = new double [3][4];
		SimpleMatrix mat = getGeometry().getProjectionMatrix(p).computeP();
		if (mat !=null) {
			//mat.print(NumberFormat.getInstance(), 6);
			voxel[3] = 1;
			updateMatrix[0][3] = mat.getElement(0,3);
			updateMatrix[1][3] = mat.getElement(1,3);
			updateMatrix[2][3] = mat.getElement(2,3);
			boolean nanHappened = false;
			double min = 20000;
			double max = 0;
			for (int k = 0; k < maxK ; k++){ // for all slices
				if (debug) System.out.println("here: " + " " + k);
				voxel[2] = (this.getGeometry().getVoxelSpacingZ() * k) - offsetZ;
				updateMatrix[0][2] = mat.getElement(0,2) * voxel[2];
				updateMatrix[1][2] = mat.getElement(1,2) * voxel[2];
				updateMatrix[2][2] = mat.getElement(2,2) * voxel[2];
				homogeniousPointk[0] = updateMatrix[0][3] + updateMatrix[0][2];
				homogeniousPointk[1] = updateMatrix[1][3] + updateMatrix[1][2];
				homogeniousPointk[2] = updateMatrix[2][3] + updateMatrix[2][2];
				for (int i=0; i < maxI; i++){ // for all lines
					voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;

					updateMatrix[0][0] = mat.getElement(0,0) * voxel[0];
					updateMatrix[1][0] = mat.getElement(1,0) * voxel[0];
					updateMatrix[2][0] = mat.getElement(2,0) * voxel[0];
					homogeniousPointi[0] = homogeniousPointk[0] + updateMatrix[0][0];
					homogeniousPointi[1] = homogeniousPointk[1] + updateMatrix[1][0];
					homogeniousPointi[2] = homogeniousPointk[2] + updateMatrix[2][0];
					for (int j = 0; j < maxJ; j++){ // for all voxels
						// compute real world coordinates in homogenious coordinates;
						boolean project = true;
						if (useVOImap){
							if (voiMap != null){
								project = voiMap[i][j][k];
							}
						}
						if (project){			
							voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
							updateMatrix[0][1] = mat.getElement(0,1) * voxel[1];
							updateMatrix[1][1] = mat.getElement(1,1) * voxel[1];
							updateMatrix[2][1] = mat.getElement(2,1) * voxel[1];
							homogeniousPointj[0] = homogeniousPointi[0] + updateMatrix[0][1];
							homogeniousPointj[1] = homogeniousPointi[1] + updateMatrix[1][1];
							homogeniousPointj[2] = homogeniousPointi[2] + updateMatrix[2][1];
							//Matrix point3d = new Matrix(voxel);
							//Compute coordinates in projection data.
							//Matrix point2d = geometry.getProjectionMatrix(p).times(point3d);
							double coordX = homogeniousPointj[0] / homogeniousPointj[2];
							double coordY = homogeniousPointj[1] / homogeniousPointj[2];
							// back project
							double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY);//*homogeniousPointj[2]);
							if (Double.isNaN(increment)){
								nanHappened = true;
								if (count < 10) System.out.println("NAN Happened at i = " + i + " j = " + j + " k = " + k + " projection = " + projectionNumber + " x = " + coordX + " y = " + coordY  );
								increment = 0;
								count ++;
							}
							// Weighting according to super-short-scan formula for cone beam geometry.
							// homogeniousPointj[2] => ||a- a(lambda)
							// conversion of coordX (the element index) back to metric detector coordinates (\tilde{u})required:
							// \tilde{u}  = (coordX-(this.detectorWidth/2))*this.detectorElementSizeX
							//homogeniousPointj[2] /= this.geometry.getSourceToDetectorDistance();
							//homogeniousPointj[2] *= homogeniousPointj[2];
							//homogeniousPointj[2] = geometry.getSourceToDetectorDistance() - homogeniousPointj[2];
							min = (homogeniousPointj[2]< min)? homogeniousPointj[2]: min;
							max = (homogeniousPointj[2]> max)? homogeniousPointj[2]: max;
							double utilde = (coordX-(this.getGeometry().getDetectorWidth()/2))*this.getGeometry().getPixelDimensionX();
							//double weight =  Math.sqrt(Math.pow(utilde,2) + Math.pow(geometry.getSourceToDetectorDistance(), 2))/ (Math.pow(geometry.getSourceToDetectorDistance(),1)*Math.abs(homogeniousPointj[2])*Math.abs(homogeniousPointj[2]));
							double weight =  Math.sqrt(Math.pow(utilde,2) + Math.pow(getGeometry().getSourceToDetectorDistance(), 2))/ (Math.pow(getGeometry().getSourceToDetectorDistance(),1) * homogeniousPointj[2]);

							// Not bad:
							// double weight = 1.0/ (200+ (Math.sqrt(Math.pow(((maxI/2)-i)*this.geometry.getVoxelSpacingX(),2)+ Math.pow(((maxJ/2)-j)*this.geometry.getVoxelSpacingY(), 2))));

							//double weight = 1.0 / (Math.pow(homogeniousPointj[2],1));

							increment *= weight ;//* this.parkerLikeWeight(primaryAngles[projectionNumber] /180.0 * Math.PI, utilde);
							updateVolume(i, j, k, increment);
						}
					}
				}
				System.out.println("Min max: " + min + " " + max);
			}
			if (nanHappened) {
				throw new RuntimeException("Encountered NaN in projection!");
			}
			if (debug) System.out.println("done with projection");
		}
	}

	@Override
	public String getName(){
		return "Specialized Cone-beam Super Short Scan Backprojector";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@ARTICLE{Noo02-IRF,\n" +
		"  author = {Noo, F. and  Defrise, M. and Clackdoyle, R. and Kudo, H.},\n" +
		"  title = \"{{Image reconstruction from fan-beam projections on less than a short scan}}\",\n" +
		"  journal = {Physics in Medicine and Biology},\n" +
		"  year = 2002,\n" +
		"  volume = 47,\n"+
		"  number = 14,\n" +
		"  pages = {2525-2546}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation(){
		return "Noo F, Defrise M, Clackdoyle R, Kudo H. Image reconstruction from fan-beam projections on less than a short scan. " +
		"Phys Med Biol 47(14):2525-46. 2002.";
	}

	@Override
	public String getToolName(){
		return "Super Short Scan Backprojector";
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/