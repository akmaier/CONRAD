package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.fitting.Function;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;

/**
 * All reconstruction algorithms are based on the reconstruction filter. The reconstruction filter is the abstract class that gives the general outline of any reconstruction algorithm. 
 * @author akmaier
 *
 */
public abstract class ReconstructionFilter extends MultiProjectionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3325343968370683491L;
	//protected ProjectionGeometry geometry;

	public Trajectory getGeometry() {
		return Configuration.getGlobalConfiguration().getGeometry();
	}


	protected double offsetX;
	protected double offsetY;
	protected double offsetZ;
	protected boolean init = false;
	protected Grid3D projectionVolume;
	protected int nImages = 0;

	protected static boolean debug = false;

	@Override
	protected void processProjectionData(int projectionNumber) throws Exception {
		if (isLastBlock(projectionNumber)) {
			nImages = inputQueue.size();
			if (projectionNumber == nImages - 1){ // last projection streamed;
				// reconstruct!
				//System.out.println(projectionNumber + " " +nImages);
				reconstruct();
			}
		}
	}
	
	/**
	 * reads all projections from inputQueue, reconstructs the volume and writes the reconstruction to the next sink. Is called only once.
	 * Must handle parallelization on its own.
	 * @throws Exception 
	 */
	protected abstract void reconstruct() throws Exception;


	/**
	 * updates the projection volume. Note that direct access to pixels is much faster than accessing the image data with getProcessor(), if the data type is known.
	 * @param i x pixel entry number 
	 * @param j y pixel entry number
	 * @param k z pixel entry number
	 * @param increment the value to add.
	 */
	public synchronized void updateVolume(int i, int j, int k, double increment){
		//float [] pixels = (float []) projectionVolume.getStack().getPixels(k + 1);
		//pixels[(j * geometry.getReconDimensionX()) + i]+= increment;
		projectionVolume.setAtIndex(i, j, k, (float) (projectionVolume.getAtIndex(i, j, k) + increment));
	}

	/**
	 * Used to set the projection volume. This is used in "SubVolumeBackprojector" in order to 
	 * reconstruct parts of the volume in a parallel manner. Note that projection volumes should be
	 * of type FloatProcessor
	 * @param projectionVolume the volume
	 */
	public void setProjectionVolume(Grid3D projectionVolume) {
		this.projectionVolume = projectionVolume;
	}

	/**
	 * creates an empty projection volume.
	 */
	public void initializeProjectionVolume(){
		projectionVolume = new Grid3D(getGeometry().getReconDimensionX(),getGeometry().getReconDimensionY(),getGeometry().getReconDimensionZ());
		projectionVolume.setOrigin(getGeometry().getOriginX(), getGeometry().getOriginY(), getGeometry().getOriginZ());
		projectionVolume.setSpacing(getGeometry().getVoxelSpacingX(),getGeometry().getVoxelSpacingY(),getGeometry().getVoxelSpacingZ());
		if (debug) System.out.println("Created Projection Volume");
		computeOffsets();
	}

	/**
	 * Sets the correct offset values that were set in the configuration.
	 * This offset is exactly -1 * the origin in pixels.
	 */
	protected void computeOffsets(){
		offsetX = -getGeometry().getOriginX();
		offsetY = -getGeometry().getOriginY();
		offsetZ = -getGeometry().getOriginZ();
	}

	/**
	 * applies the Hounsfield scaling as defined in the current global configuration. (Calls Configuration.getGlobalConfiguration().getHounsfieldScaling()).
	 */
	public void applyHounsfieldScaling(){
		Grid3D revan = projectionVolume;
		if (revan != null){
			Function hounsfield = Configuration.getGlobalConfiguration().getHounsfieldScaling();
			System.out.println("Scaling with : " + hounsfield.toString());
			for (int k = 0; k < revan.getSize()[2]; k++){
				for (int j=0; j < revan.getSize()[1]; j++) {
					for (int i=0; i < revan.getSize()[0]; i++){
						double value = hounsfield.evaluate(revan.getAtIndex(i, j, k));
						if (value < -1024) {
							value = -1024;
						} 
						revan.setAtIndex(i, j, k, (float)value);
					}
				}
			}
			revan.setSpacing(getGeometry().getReconVoxelSizes());
			double [] origin = new double [] {offsetX, offsetY , offsetZ};
			revan.setOrigin(origin);	
		}
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		projectionVolume = null;
	}


	protected synchronized void init() {
		// create volume if required
		if (!init) {
			context = 50;
			if (projectionVolume == null){
				initializeProjectionVolume();
				if (debug) System.out.println("Volume created");
			} else {
				if (debug) System.out.println("Volume already existing...");
			}
			init = true;
		}
	}
	
	public void setOffsetX(double offsetX) {
		this.offsetX = offsetX;
	}
	
	public double getOffsetX() {
		return offsetX;
	}
	
	public void setOffsetY(double offsetY) {
		this.offsetY = offsetY;
	}
	
	public double getOffsetY() {
		return offsetY;
	}
	
	public void setOffsetZ(double offsetZ) {
		this.offsetZ = offsetZ;
	}
	
	public double getOffsetZ() {
		return offsetZ;
	}
	
	public void setInit(boolean init) {
		this.init = init;
	}
	
	public boolean getInit(){
		return init;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/