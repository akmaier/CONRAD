package edu.stanford.rsl.conrad.reconstruction;

import ij.process.ImageProcessor;

import java.text.NumberFormat;
import java.util.Locale;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;


public abstract class ModelBasedIterativeReconstruction extends IterativeReconstruction{

	private static final long serialVersionUID = 1L;

	protected boolean Debug = false; 
	
	protected long time;
	protected Grid3D volumeImage = null;
	protected Grid3D projectionViews = null;
	protected double lineOffset;

	protected int maxK = 0;
	protected int maxI = 0;
	protected int maxJ = 0;
	protected int maxU = 0;
	protected int maxV = 0;

	protected double dx = 0.0;
	protected double dy = 0.0;
	protected double dz = 0.0;

	protected Trajectory dataTrajectory;

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		init = false;
	}
	
	protected synchronized void initialize(ImageProcessor projection) throws Exception{
		if (!init){
			super.init();
			// Precompute offsets
			lineOffset = 0;
			if (getGeometry().getDetectorWidth() != -1){
				System.out.println("row size projection: " + projection.getWidth() + "\nrow size detector: " + getGeometry().getDetectorWidth());
				lineOffset = (projection.getWidth() - getGeometry().getDetectorWidth()) / 2; 
			}
						
			maxI = getGeometry().getReconDimensionX();
			maxJ = getGeometry().getReconDimensionY();
			maxK = getGeometry().getReconDimensionZ();
			maxU = getGeometry().getDetectorWidth(); //or it should be projection.getWidth();
			maxV = getGeometry().getDetectorHeight();
			dx = getGeometry().getVoxelSpacingX();
			dy = getGeometry().getVoxelSpacingY();
			dz = getGeometry().getVoxelSpacingZ();
			time = System.currentTimeMillis();

			projectionViews = InitializeProjectionViews();
			volumeImage = InitializeVolumeImage();
		}
	}

	protected Grid3D InitializeProjectionViews() throws Exception{
		if (Debug) System.out.println("Created projection views");
		if (maxI == 0 || maxJ == 0 || maxK == 0 ){
			System.out.println("Errpr: Wrong projection views size!");			
		}
		
		Grid3D views;
		views = new Grid3D(nImages,maxU,maxV);
		computeOffsets();
		return views;
	}

	protected Grid3D InitializeVolumeImage() throws Exception{
		if (Debug) System.out.println("Created volume image");
		if (nImages == 0 || maxU == 0 || maxV == 0 ){
			System.out.println("Errpr: Wrong volume image size!");			
		}
		
		Grid3D image;
		image = new Grid3D(maxI,maxJ,maxK);
		computeOffsets();
		return image;
	}
	
	
	protected void copyProjectionViews()throws Exception{
		Grid2D currentProjection;
		for ( int ip = 0; ip < nImages; ip++ ){
			try {	
				currentProjection = inputQueue.get(ip);
				for (int iu = 0; iu <= maxU ; iu ++ ){
					for (int iv = 0; iv <= maxV ; iv++){
						// there may be a problem
						projectionViews.setAtIndex(ip, iu, iv, currentProjection.getPixelValue(iu, iv));
					}
				}	
			} catch (Exception e){
				System.out.println("An error occured during copying projection views " + ip);
			}
		}

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


	/**
	 * Back projects a single projection into the reconstruction space.
	 * @param projection the projection to back project
	 * @throws Exception may happen.
	 */
	protected abstract void backproject( Grid3D projImage, Grid3D volImage ) throws Exception;

	/**
	 * Forward projects the object onto the projection image.
	 * @param the object volume to back project
	 * @throws Exception may happen.
	 */
	protected abstract void forwardproject( Grid3D projImage, Grid3D volImage ) throws Exception;


	@Override
	public String getName() {
		return "Model-based Iterative Reconstruction.";
	}

	@Override
	public String getToolName() {
		return "Model-based Iterative Reconstruction";
	}

	@Override
	public synchronized void close() {
		super.close();
		//if (true) throw new RuntimeException("Happended");
		System.out.println("Closing iterative reconstruction");
		//done = true;
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}
	
	
	
	//----------------Methods for testing ----------------//
	
	
	public static void printSimpleMatrix( SimpleMatrix A ){
		int n = A.getRows();
		int m = A.getCols();

		for (int i = 0; i < n ; i++ ){
			for (int j = 0; j < m ; j++){
				System.out.print( A.getElement(i, j) + "\t");
			}
			System.out.print("\n");
		}
	}
	
	public static void printSimpleVector( SimpleVector B ){
		int n = B.getLen();
		for (int i = 0; i < n ; i++ ){
			System.out.print( B.getElement(i) + "\t");
		}
		System.out.print("\n");
	}
	
	
	
	//constructor overrides geometry for testing
	public ModelBasedIterativeReconstruction( Trajectory dataTraj ){
		dataTrajectory = dataTraj;
	}

	
	//override for testing 
	public Trajectory getGeometry() {
		return dataTrajectory;
	}

	
	public synchronized void initializeTest() throws Exception{
		if (!init){
			super.init();
			nImages = getGeometry().getNumProjectionMatrices();
			maxI = getGeometry().getReconDimensionX();
			maxJ = getGeometry().getReconDimensionY();
			maxK = getGeometry().getReconDimensionZ();
			maxU = getGeometry().getDetectorWidth(); //or it should be projection.getWidth();
			maxV = getGeometry().getDetectorHeight();
			dx = getGeometry().getVoxelSpacingX();
			dy = getGeometry().getVoxelSpacingY();
			dz = getGeometry().getVoxelSpacingZ();
			time = System.currentTimeMillis();
		
			projectionViews = InitializeProjectionViews();
			volumeImage = InitializeVolumeImage();
		}
	}

	public void printOutGeometry(){
		System.out.println("Detector Size: " + maxU + " X " + maxV );
		System.out.println("Volume Size: " + maxI + " X " + maxJ + " X " + maxK );
		System.out.println("Voxel Dimension: " + dx + " X " + dy + " X " + dz );
		System.out.println( "Offsets: " + offsetX  + ", " + offsetY + ", " + offsetZ) ;
		if (volumeImage != null && projectionViews != null ){
			System.out.println( "volumeImage and projectionViews created successfully!");
		}
	}

	
}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/