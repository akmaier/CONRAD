package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.filtering.rampfilters.RampFilter;
import edu.stanford.rsl.conrad.filtering.redundancy.ParkerWeightingTool;
import edu.stanford.rsl.conrad.filtering.redundancy.SilverWeightingTool;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.process.FloatProcessor;

/**
 * A FBP-based reconstruction algorithm which evaluates whether 180 degrees of angular coverage have been achieved during the reconstruction. If less than 180 degrees were observed, the method tries to extrapolate the missing data by a linear estimation.
 * @author akmaier
 *
 */
public class RayWeightCorrectingCPUSuperShortScanBackprojector extends VOIBasedReconstructionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -9058395760356947846L;
	protected FloatProcessor rayMap1, rayMap2;
	protected boolean mapped = false;
	protected ParkerWeightingTool silver = null;
	protected RampFilter ramp;
	

	
	@Override
	public void backproject(Grid2D projection, int projectionNumber){
		int count = 0;
		//System.out.println(projectionVolume);
		if (!init){
			initialize(projection);
			rayMap1 = new FloatProcessor(maxI, maxJ);
			rayMap2 = new FloatProcessor(maxI, maxJ);
		}
		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(),projection.getHeight(), projection.getBuffer(), null);
		double [] minmax = DoubleArrayUtil.minAndMaxOfArray(getGeometry().getPrimaryAngles());
		// Compute redundancy weights
		double [] weights = silver.computeParkerWeights1D(projectionNumber);
		double [] circle = new double [weights.length];
		// Simulate uniform object
		for (int i = 0; i< weights.length; i++){
			double dist = ((double)(((weights.length-10)/2) - Math.abs(i - (weights.length/2))))/ ((weights.length-10)/2);
			if (dist <= 0) {
				circle [i] = 0;
			} else {
				circle [i] = Math.sin(Math.acos(1-dist));
			}
			weights[i]*=circle[i];
		}
		if (projectionNumber == 1){
			VisualizationUtil.createPlot("Cirlce", circle).show();
		}
		// ramp Filter
		weights = FFTUtil.applyRampFilter(weights, ramp);
		//ImageProcessor currentProjection = projection;
		int p = projectionNumber;
		double[] voxel = new double [4];
		double[] homogeniousPointi = new double[3];
		double[] homogeniousPointj = new double[3];
		double[] homogeniousPointk = new double[3];
		double[][] updateMatrix = new double [3][4];
		SimpleMatrix mat = getGeometry().getProjectionMatrix(p).computeP();
		//mat.print(NumberFormat.getInstance(), 6);
		voxel[3] = 1;
		updateMatrix[0][3] = mat.getElement(0,3);
		updateMatrix[1][3] = mat.getElement(1,3);
		updateMatrix[2][3] = mat.getElement(2,3);
		boolean nanHappened = false;
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
						boolean inProjection = true;
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
						if ((coordX < 0)
								||(coordX>=this.getGeometry().getDetectorWidth())
								||(coordY<0)
								||(coordY >=this.getGeometry().getDetectorHeight())){
							inProjection = false;
						}
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
						homogeniousPointj[2] /= this.getGeometry().getSourceToDetectorDistance();
						double zweight = (Math.pow(Math.abs(homogeniousPointj[2]),2));
					
						if (voxel[0] < 0){
							//zweight = ;
						}
						double utilde = (coordX-(this.getGeometry().getDetectorWidth()/2.0))*this.getGeometry().getPixelDimensionX();
						double angle = Math.atan(utilde/getGeometry().getSourceToDetectorDistance());

						double beta = ((getGeometry().getPrimaryAngles()[projectionNumber] - minmax[0]) /180.0 * Math.PI);
						//double delta = Math.atan(((detectorWidth/2.0)*detectorElementSizeX)/sourceToDetectorDistance);
						double redundancyWeight = weights[(int)Math.round(coordX)];
						double weight =  Math.sqrt(Math.pow(utilde,2) + Math.pow(getGeometry().getSourceToDetectorDistance(), 2))/ (Math.pow(this.getGeometry().getSourceToDetectorDistance(),1)*zweight);
						increment *= redundancyWeight*weight ;//* this.parkerLikeWeight(primaryAngles[projectionNumber] /180.0 * Math.PI, utilde);
						if(inProjection){				
							double ray = (beta - angle);
							//double otherray = (beta) - (3* angle) + Math.PI;
							
							if(ray < 0) ray += Math.PI;
							if(ray > Math.PI) ray = Math.PI;
							
							//if (ray < 0) System.out.println(primaryAngles[projectionNumber] + " " + utilde + " " + ray + " " + angle / Math.PI * 180.0);
							//if (ray > Math.PI) System.out.println(primaryAngles[projectionNumber] + " " + utilde + " " + ray);
							rayMap1.putPixelValue(i, j, ((redundancyWeight*(weight)) + rayMap1.getPixelValue(i, j)));
							rayMap2.putPixelValue(i, j, (Math.max(rayMap2.getPixelValue(i, j),ray)));
						}
			
						updateVolume(i, j, k, increment);
					}
				}
			}
		}
		if (nanHappened) {
			throw new RuntimeException("Encountered NaN in projection!");
		}
		if (debug) System.out.println("done with projection");
	}
	
	@Override 
	public void configure() throws Exception{
		super.configure();
		silver = new SilverWeightingTool();
		silver.configure();
		ramp = new RamLakRampFilter();
		ramp.configure();
	}
	
	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		this.rayMap1 = null;
		this.rayMap2 = null;
		this.silver = null;
	}
	
	@Override
	public String getName(){
		return "Ray Weighting Cone-beam Super Short Scan Backprojector (Testing)";
	}
	
	@Override
	public String getBibtexCitation(){
		return CONRAD.CONRADBibtex;
	}
	
	@Override
	public String getMedlineCitation(){
		return CONRAD.CONRADMedline;
	}
	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/