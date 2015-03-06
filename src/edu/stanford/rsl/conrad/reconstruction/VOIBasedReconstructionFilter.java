package edu.stanford.rsl.conrad.reconstruction;


import java.text.NumberFormat;
import java.util.Locale;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.reconstruction.voi.CylinderBasedVolumeOfInterest;
import edu.stanford.rsl.conrad.reconstruction.voi.VolumeOfInterest;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;
import ij.process.FloatProcessor;


/**
 * The VOIBasedReconstructionFilter is an implementation of the backprojection which employs a volume-of-interest (VOI) to
 * speed up reconstruction. Only voxels within the VOI will be regarded in the backprojection step. Often this can save up to 30 to 40 % in computation time
 * as volumes are usually described as boxes but the VOI is just a cylinder.
 * 
 * @author akmaier
 *
 */
public class VOIBasedReconstructionFilter extends FBPReconstructionFilter {


	/**
	 * 
	 */
	private static final long serialVersionUID = -4469835421293447988L;

	protected long time;

	protected double lineOffset;

	protected int maxK = 0;
	protected int maxI = 0;
	protected int maxJ = 0;



	protected boolean fastVOIMode = true;

	protected boolean [][][] voiMap;
	protected boolean useVOImap = true;
	protected VolumeOfInterest interestedInVolume = null;
	//protected boolean done = false;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		init = false;
		voiMap = null;
	}

	public void setMaximumVolumeOfInterest(String maxVOIFile) {
		interestedInVolume = VolumeOfInterest.openAsVolume(maxVOIFile);
		if (interestedInVolume == null) {
			useVOImap =false;
			CONRAD.log("No VOI being used.");
		}
	}

	public void setMaximumVolumeOfInterest(VolumeOfInterest maxVOI){
		interestedInVolume = maxVOI;
		if (interestedInVolume == null) {
			useVOImap =false;
			CONRAD.log("No VOI being used.");
		}
	}



	public VOIBasedReconstructionFilter () {

	}


	@Deprecated
	public void setConfiguration(Configuration config){
		if (debug) System.out.println("Running config");
		//super.setConfiguration(config);

		if (debug) System.out.println("config done.");
	}







	protected synchronized void initialize(Grid2D projection){
		if (!init){
			super.init();
			// Precompute offsets
			lineOffset = 0;
			if (getGeometry().getDetectorWidth() != -1){
				CONRAD.log("row size projection: " + projection.getWidth() + "\nrow size detector: " + getGeometry().getDetectorWidth());
				lineOffset = (projection.getWidth() - getGeometry().getDetectorWidth()) / 2; 
			}
			maxI = getGeometry().getReconDimensionX();
			maxJ = getGeometry().getReconDimensionY();
			maxK = getGeometry().getReconDimensionZ();
			try {
				if (Configuration.getGlobalConfiguration().getVolumeOfInterestFileName() != null) {
					this.setMaximumVolumeOfInterest(Configuration.getGlobalConfiguration().getVolumeOfInterestFileName());
				} else {
					if (Configuration.getGlobalConfiguration().getImportFromDicomAutomatically()) {
						String filename = Configuration.getGlobalConfiguration().getRegistryEntry(RegKeys.PATH_TO_CALIBRATION) + "/" + Configuration.getGlobalConfiguration().getDeviceSerialNumber() + "/active/-903/" + "/d/" + Configuration.getGlobalConfiguration().getIntensifierSize() + "/" + Configuration.getStandardNumberFormat().format(Configuration.getGlobalConfiguration().getGeometry().getAverageAngularIncrement()) + "/maxVOI.txt";
						this.setMaximumVolumeOfInterest(filename);
					}
				}
			} catch (Exception e){
				System.out.println(e.getLocalizedMessage());
			}
			initializeVOIMap();
			time = System.currentTimeMillis();
		}
	}

	public synchronized void initializeVOIMap(){
		if (useVOImap) {
			voiMap = new boolean[maxI][maxJ][maxK];
			CONRAD.log("Creating Voi map - Current Time:" + System.currentTimeMillis());
			for (int k = 0; k < maxK; k++){ // for all slices
				double[] voxel = new double [4];
				voxel[2] = (this.getGeometry().getVoxelSpacingZ() * k) - offsetZ;
				for (int i=0; i < maxI; i++){ // for all lines
					voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;
					for (int j = 0; j < maxJ; j++){ // for all voxels
						voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
						if (interestedInVolume == null) {
							voiMap[i][j][k] = true;
						} else {
							if (interestedInVolume.contains(voxel[0], voxel[1], voxel[2])) {
								voiMap[i][j][k] = true;
							} else {
								voiMap[i][j][k] = false;
							}
						}
					}
				}
				if (fastVOIMode){
					int lowerLimit = (int) (maxK*0.1);
					if (interestedInVolume instanceof CylinderBasedVolumeOfInterest){
						lowerLimit = (int) (maxK*0.2);
					}
					if (k > lowerLimit) {
						int upperLimit = (int) (maxK*0.9);
						if (interestedInVolume instanceof CylinderBasedVolumeOfInterest){
							upperLimit = (int) (maxK*0.8);
						}
						if (k < upperLimit){
							int l;
							for (l = lowerLimit + 1; l <= upperLimit; l++){
								// Copy the values in between lower and upper limit.
								// This improves the execution speed drastically
								for (int i=0; i< maxI; i++){ // for all lines
									for (int j = 0; j < maxJ; j++){ // for all voxels
										voiMap[i][j][l] = voiMap[i][j][lowerLimit];
									}
								}
							}
							// set k to the upper limit - 1 as it will be increased in the next iteration
							k=l-1;
						}
					}
				}
			}
			if (interestedInVolume == null){
				CONRAD.log("VOIBasedBackprojector: interestedInVolume was null.");
			} else {
				CONRAD.log("VOIBasedBackprojector: VOI map created." + System.currentTimeMillis() + " " + interestedInVolume);
			}
		} else {			
			CONRAD.log("VOIBasedBackprojector: Omitting creation of VOI map");
		}
	}
	
	private Grid3D voiMapToGrid3D(){
		Grid3D out = null;
		if(voiMap!=null){
			out = new Grid3D(voiMap.length, voiMap[0].length, voiMap[0][0].length);
			for (int i = 0; i < voiMap.length; i++) {
				for (int j = 0; j < voiMap[0].length; j++) {
					for (int j2 = 0; j2 < voiMap[0][0].length; j2++) {
						out.setAtIndex(i, j, j2, (voiMap[i][j][j2]) ? 1.0f : -1.0f);
					}
				}
			}
		}
		return out;
	}

	public void backproject(Grid2D projection, int projectionNumber){
		int count = 0;
		//System.out.println(projectionVolume);
		if (!init){
			initialize(projection);
		}
		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(), projection.getHeight(), projection.getBuffer(), null);
		//ImageProcessor currentProjection = projection;

		// Constant part of distance weighting (D^2) + additional weighting for arbitrary scan ranges
		double D =  getGeometry().getSourceToDetectorDistance();
		currentProjection.multiply(D*D * 2* Math.PI / getGeometry().getNumProjectionMatrices());

		int p = projectionNumber;
		double[] voxel = new double [4];
		double[] homogeniousPointi = new double[3];
		double[] homogeniousPointj = new double[3];
		double[] homogeniousPointk = new double[3];
		double[][] updateMatrix = new double [3][4];
		SimpleMatrix mat = getGeometry().getProjectionMatrix(p).computeP();
		//mat.print(NumberFormat.getInstance(), 6);
		voxel[3] = 1;
		if (mat != null){
			updateMatrix[0][3] = mat.getElement(0,3);
			updateMatrix[1][3] = mat.getElement(1,3);
			updateMatrix[2][3] = mat.getElement(2,3);
			boolean nanHappened = false;
			for (int k = 0; k < maxK ; k++){ // for all slices
				if (debug) System.out.println("here: " + " " + k);
				voxel[2] = (this.getGeometry().getVoxelSpacingZ() * (k)) - offsetZ;
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
							//System.out.println(coordY);
							// back project
							double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY) / (homogeniousPointj[2]*homogeniousPointj[2]);
							if (Double.isNaN(increment)){
								nanHappened = true;
								if (count < 10) System.out.println("NAN Happened at i = " + i + " j = " + j + " k = " + k + " projection = " + projectionNumber + " x = " + coordX + " y = " + coordY  );
								increment = 0;
								count ++;
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
	}

	@Override
	public String getName() {
		return "Parallel CPU-based Backprojector";
	}




	@Override
	public void configure() throws Exception{
		boolean success = true;
		NumberFormat nf = NumberFormat.getInstance(Locale.US);
		nf.setMaximumFractionDigits(1);
		nf.setMinimumFractionDigits(1);
		nf.setMaximumIntegerDigits(1);
		nf.setMinimumIntegerDigits(1);
		initializeProjectionVolume();
		configured = success;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Rohkohl08-CCR,\n" +
				"  author = {{Rohkohl}, C. and {Lauritsch}, G. and { N{\"o}ttling}, A. and {Pr{\"u}mmer}, M. and {Hornegger}, J.},\n" +
				"  title = {{C-Arm CT: Reconstruction of Dynamic High Contrast Objects Applied to the Coronary Sinus}},\n" +
				"  booktitle = {{Nuclear Science Symposium and Medical Imaging Conference Record}},\n" +
				"  publisher = {IEEE},\n" +
				"  address = {Dresden, Germany},\n" +
				"  pages= {no pagination},\n" +
				"  year = {2008}\n" +
				"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Rohkohl C, Lauritsch G, Nöttling A, Prümmer M, Hornegger J. C-Arm CT: Reconstruction of Dynamic High Contrast Objects Applied to the Coronary Sinus. In Nuclear Science Symposium and Medical Imaging Conference Record, IEEE, Dresden, Germany, 2008.";
	}

	@Override
	public synchronized void close() {
		super.close();
		//if (true) throw new RuntimeException("Happended");
		//System.out.println("Closing Backprojector");
		//done = true;
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	@Override
	public String getToolName() {
		return "VOI-based Backprojector";
	}

	/**
	 * @return the fastVOIMode
	 */
	public boolean isFastVOIMode() {
		return fastVOIMode;
	}

	/**
	 * @param fastVOIMode the fastVOIMode to set
	 */
	public void setFastVOIMode(boolean fastVOIMode) {
		this.fastVOIMode = fastVOIMode;
	}

	/**
	 * @param maxI the maxI to set
	 */
	public void setMaxI(int maxI) {
		this.maxI = maxI;
	}
	
	/**
	 * @return maxI the maxI index
	 */
	public int getMaxI() {
		return maxI;
	}
	
	/**
	 * 
	 * @param maxJ the maxJ index to set
	 */
	public void setMaxJ(int maxJ) {
		this.maxJ = maxJ;
	}
	
	/**
	 * @return maxJ the maxJ index
	 */
	public int getMaxJ() {
		return maxJ;
	}
	
	/**
	 * 
	 * @param maxK the maxK index to set
	 */
	public void setMaxK(int maxK) {
		this.maxK = maxK;
	}
	
	/**
	 * @return maxK the maxK index
	 */
	public int getMaxK() {
		return maxK;
	}
	
	/**
	 * 
	 * @param lineOffset the line offset to set
	 */
	public void setLineOffset(double lineOffset) {
		this.lineOffset = lineOffset;
	}
	
	/**
	 * 
	 * @return the line offset
	 */
	public double getLineOffset() {
		return lineOffset;
	}

	public VolumeOfInterest getInterestedInVolume() {
		return interestedInVolume;
	}
	
	public boolean getUseVOImap(){
		return useVOImap;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */