package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;

import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLCompensatedBackProjector1DCompressionField;

/**
 * This class implements an initial optimization of the motion field generated
 * by the diaphragm tracking and triangulation filters. The motion field stored
 * in the Configuration is loaded and refined. The motion signal is first scaled
 * to optimize the correct maximum amplitude. A compensated reconstruction using
 * the scaled signal is done. The reconstruction quality is evaluated based on
 * the contrast of diaphragm and lung.
 * 
 * @author Marco Boegel
 * 
 */
public class InitialOptimization {

	/**
	 * Motionfield that was detected by the DiaphragmTrackingTool and
	 * triangulated with the TriangulationTool Loaded from Configuration. The
	 * field is scaled down to [0,1] after initialization.
	 */
	private static double[] motionfieldOrig;

	/**
	 * Offset for the compressionfield, to take small triangulation errors into
	 * account
	 */
	private static int diaOffset;
	public static final int diaOffsetMM = 4;

	/**
	 * Width of the ROI the reconstruction is evaluated on
	 */
	private static final int ROIwidth = 10;

	/**
	 * Motion field. Loaded from Configuration
	 */
	private static double[] respMotionField;

	/**
	 * 3-D positions of diaphragm vertices.
	 */
	private static double[] diaPositionField;
	/**
	 * Number of projections.
	 */
	private static int maxProjs;

	/**
	 * Index of motionfield minimum.
	 */
	private static int minIndex;
	private static int expiration;
	
	private static double voxelSpacingX;
	private static double voxelSpacingY;
	private static double voxelSpacingZ;

	private double mVal;
	private static int recDimensionZ;
	// Width of ROI in voxels and position of diaphragm in voxels
	private static int diaX, diaY;
	// origin of volume
	private static double ox, oy, oz;
	// Start and end voxels of the ROI
	private static int xStart, xEnd, yStart, yEnd, zStart, zEnd;

	/**
	 * Contains filtered projection image data. Needs to be loaded outside of
	 * class.
	 */
	private ProjectionLoader pLoad;

	/**
	 * Constructor. Initializes all important fields.
	 * 
	 * @param pLoad
	 *            Projection loader with prefiltered projection images
	 */
	public InitialOptimization(ProjectionLoader pLoad) {
		this.pLoad = pLoad;
		initialize();
	}
	
	

	/**
	 * This method initializes all fields and reconstruction parameters.
	 */
	private void initialize() {

		Configuration.loadConfiguration();
		Configuration conf = Configuration.getGlobalConfiguration();
		Trajectory tra = conf.getGeometry();

		// Load tracked motion and positionfields
		respMotionField = conf.getRespiratoryMotionField();
		diaPositionField = conf.getDiaphragmPositionField();
		minIndex = minPosition(respMotionField);
		expiration = maxPosition(respMotionField);
		// Reconstruction parameters
		voxelSpacingX = conf.getGeometry().getVoxelSpacingX();
		voxelSpacingY = conf.getGeometry().getVoxelSpacingY();
		voxelSpacingZ = conf.getGeometry().getVoxelSpacingZ();
		recDimensionZ = conf.getGeometry().getReconDimensionZ();
		maxProjs = tra.getProjectionStackSize();

		// Scale the motionfield to [0,1]
		motionfieldOrig = new double[maxProjs];
		double minVal = respMotionField[minIndex];
		double expVal = respMotionField[expiration];
		mVal = Math.abs(minVal)<Math.abs(expVal)?expVal:minVal;
		
		for (int i = 0; i < maxProjs; i++) {
			motionfieldOrig[i] = respMotionField[i] / mVal;
		}
	
		
		diaOffset = (int) (diaOffsetMM / voxelSpacingZ);

		diaX = (int) Math.round(diaPositionField[0]);
		diaY = (int) Math.round(diaPositionField[1]);

		System.out.println("Diaphragmposition x: " + diaX + "  y: " + diaY);

		ox = Configuration.getGlobalConfiguration().getGeometry().getOriginX();
		oy = Configuration.getGlobalConfiguration().getGeometry().getOriginY();
		oz = Configuration.getGlobalConfiguration().getGeometry().getOriginZ();

		xStart = (int) ((diaX - ox - ROIwidth) / voxelSpacingX);
		xEnd = (int) ((diaX - ox + ROIwidth) / voxelSpacingX);
		yStart = (int) ((diaY - oy - ROIwidth) / voxelSpacingY);
		yEnd = (int) ((diaY - oy + ROIwidth) / voxelSpacingY);

		int offsetZ = (int) (50.0 / voxelSpacingZ);
		zStart = offsetZ;
		zEnd = 52;//recDimensionZ / 2;

	}

	/**
	 * This method creates a compression motion field based on a motionfield
	 * input and an area in which the motion should be compressed. Values
	 * smaller than "min" retain the initial motion field
	 * 
	 * @param max
	 *            Max position (diaphragm top)
	 * @param min
	 *            min position (lower border of diaphragm)
	 * @param motion
	 *            motionfield
	 * @return compressed motionfield
	 */
	private float[][] computeCompressionField(int lowerBorder,
			int diaphragmTop, double motion[], int recDimensionZ) {
		float[][] compField = new float[maxProjs][recDimensionZ];

		int min = lowerBorder;
		int max = diaphragmTop;
		double dist = max - min;
		for (int i = 0; i < maxProjs; i++) {
			for (int z = 0; z < recDimensionZ; z++) {
				if (z >= min && z <= max) {
					compField[i][z] = (float) (motion[i] * (dist - (max - z)) / dist);
				} else if (z < min) {
					compField[i][z] = 0;
				} else if (z > max) {
					compField[i][z] = (float) motion[i];
				}
			}
		}

		return compField;
	}

	/**
	 * Method to evaluate the reconstruction in the pre-defined ROI
	 * 
	 * @param volume
	 *            reconstruction
	 * @return evaluation score, the lower - the better
	 */
	private float evalReconstruction(Grid3D volume) {

		float evalScore = 0.f;
		evalScore = computeEntropy(volume, xStart, xEnd, yStart, yEnd, zStart,
				zEnd);

		return evalScore;
	}

	/**
	 * This method scales the motionfield.
	 * 
	 * @param func
	 *            motionfield
	 * @param step
	 *            step
	 * @param scaleSize
	 *            scaling per step
	 * @return scaled motionfield
	 */
	private double[] adjustMotionFieldAmplitude(double func[], int step,
			double scaleSize) {

		double[] motionField = new double[maxProjs];

		for (int i = 0; i < maxProjs; i++) {
			motionField[i] = func[i] * (double) step * scaleSize;
		}

		return motionField;
	}

	/**
	 * Detect position of minimum in motionfield array.
	 * 
	 * @param motionfield
	 *            Motionfield to be searched
	 * @return minimum index
	 */
	private int minPosition(double[] motionfield) {

		int i = 0;
		double min = Double.MAX_VALUE;
		int idx = 0;
		while (i < motionfield.length) {
			if (motionfield[i] < min) {
				idx = i;
				min = motionfield[i];
			}
			i++;
		}

		return idx;
	}
	
	/**
	 * Detect position of maximum in motionfield array.
	 * 
	 * @param motionfield
	 *            Motionfield to be searched
	 * @return maximum index
	 */
	private int maxPosition(double[] motionfield) {

		int i = 0;
		double max = Double.MIN_VALUE;
		int idx = 0;
		while (i < motionfield.length) {
			if (motionfield[i] > max) {
				idx = i;
				max = motionfield[i];
			}
			i++;
		}

		return idx;
	}

	/**
	 * Computes image entropy in a given grid and a defined region
	 * 
	 * @param grid
	 *            image
	 * @param xStart
	 *            ROI x start
	 * @param xEnd
	 *            ROI x end
	 * @param yStart
	 *            ROI y start
	 * @param yEnd
	 *            ROI y end
	 * @param zStart
	 *            ROI z start
	 * @param zEnd
	 *            ROI z end
	 * @return image entropy
	 */
	private float computeEntropy(Grid3D grid, int xStart, int xEnd, int yStart,
			int yEnd, int zStart, int zEnd) {

		float entropy = 0.f;
		int bins = 100;
		int sizeX = 1 + xEnd - xStart;
		int sizeY = 1 + yEnd - yStart;
		int sizeZ = 1 + zEnd - zStart;
		int size = sizeX * sizeY * sizeZ;
		int[] histo = getHistogram(grid, bins, xStart, xEnd, yStart, yEnd,
				zStart, zEnd);

		for (int i = 0; i < bins; i++) {
			double val = (double)histo[i]/(double)size;
			entropy -= val *Math.log(val+0.00001);
		}

		return entropy;
	}

	/**
	 * Computes histogram with specific bins of a given image in a defined
	 * region.
	 * 
	 * @param grid
	 *            image
	 * @param bins
	 *            amount of bins
	 * @param xStart
	 *            ROI x start
	 * @param xEnd
	 *            ROI x end
	 * @param yStart
	 *            ROI y start
	 * @param yEnd
	 *            ROI y end
	 * @param zStart
	 *            ROI z start
	 * @param zEnd
	 *            ROI z end
	 * @return histogram with x bins
	 */
	private int[] getHistogram(Grid3D grid, int bins, int xStart, int xEnd,
			int yStart, int yEnd, int zStart, int zEnd) {

		int[] histo = new int[bins];
		double[] histof = new double[bins];
		float val = 0;
		float max = -Float.MAX_VALUE;
		float min = Float.MAX_VALUE;
		for (int i = xStart; i <= xEnd; i++) {
			for (int j = yStart; j <= yEnd; j++) {
				for (int k = zStart; k <= zEnd; k++) {
					val = grid.getAtIndex(i, j, k);
					if (val > max) {
						max = val;
					}
					if (val < min) {
						min = val;
					}
				}
			}
		}
		float range = max - min;
		float binSize = range / (float) (bins - 1);

		for (int i = xStart; i <= xEnd; i++) {
			for (int j = yStart; j <= yEnd; j++) {
				for (int k = zStart; k <= zEnd; k++) {
					val = grid.getAtIndex(i, j, k);
					int b = (int) ((val - min) / binSize);
					histo[b]++;
					histof[b]++;
				}
			}
		}
//VisualizationUtil.createPlot(histof, "", "intensity", "count").show();
		return histo;
	}



	public float[] optimizeCompressedWithPrior()
			throws IOException {

		OpenCLCompensatedBackProjector1DCompressionField obp = new OpenCLCompensatedBackProjector1DCompressionField();
		obp.loadInputQueue(pLoad.getProjections());
		Grid3D result;
		float evalMin = Float.MAX_VALUE;
		int iMin = 0, cMin = 0;
		float eval = 0;

		int diaphragmPositionZ = (int) ((diaPositionField[expiration*3+2] - oz) / voxelSpacingZ)
				+ diaOffset;




//		for (int i = -24; i < 40; i++) {
		for (int i = (int)mVal-5; i < mVal+5; i++) {
			for (int c = 0; c <= 5; c++) {

				double[] motionfield = adjustMotionFieldAmplitude(
						motionfieldOrig, i, 1);
				
				int compressionLowerBorder;

				compressionLowerBorder = (int) ((c - 2) * recDimensionZ / 10.f);

				if (compressionLowerBorder >= diaphragmPositionZ)
					continue;

				float[][] compressedMotionfield = computeCompressionField(
						compressionLowerBorder, diaphragmPositionZ,
						motionfield, recDimensionZ);
				

				result = obp.reconstructCL(compressedMotionfield);

//				result.show();
				eval = evalReconstruction(result);
				System.out.println(i + "  " + (c - 2) + "     " + eval
						+ "||||   " + evalMin + "  " 
						+ "   " + iMin + "   " + cMin);
				result = null;
				if (eval < evalMin) {
					iMin = i;
					cMin = c - 2;
					evalMin = eval;
				}
			}
		}

		System.out.println( iMin + "  " + cMin);
		float params[] = new float[] { (float) (iMin), 1.f, -1.f, -1.f,
		cMin, (float) diaPositionField[expiration*3+2] };
			
		return params;
	}

	

	/**
	 * This method computes the motionfield given the optimized parameter array
	 * from the optimization functions
	 * 
	 * @param params
	 *            parameter vector, returned from optimizeCompressedWithPrior
	 * @param reconSizeZ
	 *            size of reconstruction in z-dimension
	 * @param vZ
	 *            voxelSpacing in z-dimension
	 * @param oZ
	 *            origin in z-dimension
	 * @return the motion field 
	 */
	public float[][] getMotionField(float[] params, int reconSizeZ, double vZ,
			double oZ) {
		double[] func = new double[maxProjs];
	
		func = motionfieldOrig;

		double[] motionfield = adjustMotionFieldAmplitude(func,
				(int) params[0], params[1]);

		int compressionLowerBorder;

		compressionLowerBorder = (int) ((params[4]) * reconSizeZ / 10.f);
		float[][] m = computeCompressionField(compressionLowerBorder,
				(int) ((params[5] - oZ) / vZ) + diaOffset, motionfield,
				reconSizeZ);

		return m;
	}

	public static void main(String[] args) throws Exception {

		new ImageJ();
		Configuration.loadConfiguration();
		Trajectory geom = Configuration.getGlobalConfiguration().getGeometry();
		geom.setReconDimensionX(128);
		geom.setReconDimensionY(128);
		geom.setReconDimensionZ(128);
		geom.setVoxelSpacingX(2.0);
		geom.setVoxelSpacingY(2.0);
		geom.setVoxelSpacingZ(2.0);
		double xOriginWorld = -(128 - 1.0) / 2.0 * 2.0;
		double yOriginWorld = -(128 - 1.0) / 2.0 * 2.0;
		double zOriginWorld = -(128 - 1.0) / 2.0 * 2.0;
		geom.setOriginInPixelsX(General.worldToVoxel(0.0, 2.0, xOriginWorld));
		geom.setOriginInPixelsY(General.worldToVoxel(0.0, 2.0, yOriginWorld));
		geom.setOriginInPixelsZ(General.worldToVoxel(0.0, 2.0, zOriginWorld));

		Configuration.saveConfiguration();
		Configuration.loadConfiguration();

		ProjectionLoader pLoad = new ProjectionLoader();
		String filename = FileUtil.myFileChoose(".zip", false);
		pLoad.loadAndFilterImages(filename);
		// pLoad.loadAndFilterImages("C:\\Users\\Mago\\Desktop\\breathonlySegmentiert250330.zip");
		InitialOptimization i = new InitialOptimization(pLoad);


		OpenCLCompensatedBackProjector1DCompressionField obpCompressed = new OpenCLCompensatedBackProjector1DCompressionField();
		obpCompressed.loadInputQueue(pLoad.getProjections());

		float[] params = i.optimizeCompressedWithPrior();

		Configuration.loadConfiguration();
		geom = Configuration.getGlobalConfiguration().getGeometry();
		geom.setReconDimensionX(512);
		geom.setReconDimensionY(512);
		geom.setReconDimensionZ(256);
		geom.setVoxelSpacingX(0.5);
		geom.setVoxelSpacingY(0.5);
		geom.setVoxelSpacingZ(1);
		xOriginWorld = -(512 - 1.0) / 2.0 * 0.5;
		yOriginWorld = -(512 - 1.0) / 2.0 * 0.5;
		zOriginWorld = -(256 - 1.0) / 2.0 * 1;
		geom.setOriginInPixelsX(General.worldToVoxel(0.0, 0.5, xOriginWorld));
		geom.setOriginInPixelsY(General.worldToVoxel(0.0, 0.5, yOriginWorld));
		geom.setOriginInPixelsZ(General.worldToVoxel(0.0, 1, zOriginWorld));

		Configuration.saveConfiguration();

		float[][] m = i.getMotionField(params, 256, geom.getVoxelSpacingZ(),
				geom.getOriginZ());

		Grid3D result = obpCompressed.reconstructCL(m);
		CylinderVolumeMask mask = new CylinderVolumeMask(result.getSize()[0],
				result.getSize()[1], result.getSize()[0] / 2,
				result.getSize()[1] / 2, result.getSize()[0] * 0.5);
		mask.applyToGrid(result);
		result.show();

		OpenCLCompensatedBackProjector1DCompressionField op = new OpenCLCompensatedBackProjector1DCompressionField();
		ProjectionLoader pLoad2 = new ProjectionLoader();
		String file = FileUtil.myFileChoose(".zip", false);
		pLoad2.loadAndFilterImages(file);

		op.loadInputQueue(pLoad2.getProjections());

		Grid3D resulte = op.reconstructCL(m);
		CylinderVolumeMask maske = new CylinderVolumeMask(result.getSize()[0],
				result.getSize()[1], result.getSize()[0] / 2,
				result.getSize()[1] / 2, result.getSize()[0] * 0.5);
		maske.applyToGrid(resulte);
		resulte.show();

		
	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/