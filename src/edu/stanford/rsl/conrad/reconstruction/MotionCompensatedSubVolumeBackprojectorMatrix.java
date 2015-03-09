package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionUtil;
import edu.stanford.rsl.conrad.utils.CONRAD;

/**
 * This FBP-based mathod splits the reconstruction volume into sub volumes which can be processed in parallel to speed up the reconstruction further.
 * Internally motion-compensated voi reconstructors are used.
 * @author akmaier
 *
 */
public class MotionCompensatedSubVolumeBackprojectorMatrix extends SubVolumeBackprojector  {


	/**
	 * 
	 */
	private static final long serialVersionUID = 1944993618039325981L;
	MotionField motionField;
	
	public void setMotionField(MotionField motionField) {
		this.motionField = motionField;
	}
	
	public MotionField getMotionField() {
		return motionField;
	}
	
	public MotionCompensatedSubVolumeBackprojectorMatrix() {
		numThreads = CONRAD.getNumberOfThreads();
		projectors = new VOIBasedReconstructionFilter[numThreads];
		subVolumes = new Grid3D[numThreads];
		for (int i = 0; i < numThreads; i++){
			projectors[i] = new MotionCompensatedVOIBasedReconstructionFilterMatrix();
		}
	}

	@Override
	public String getName(){
		return "Parallel Motion Compensated CPU-based Sub-volume Backprojector (Affine Matrix Motion Field)";
	}

	@Override
	public void initializeProjectionVolume(){
		projectionVolume = new Grid3D(getGeometry().getReconDimensionX(),getGeometry().getReconDimensionY(),getGeometry().getReconDimensionZ());
		projectionVolume.setOrigin(getGeometry().getOriginX(), getGeometry().getOriginY(), getGeometry().getOriginZ());
		projectionVolume.setSpacing(getGeometry().getVoxelSpacingX(),getGeometry().getVoxelSpacingY(),getGeometry().getVoxelSpacingZ());
		if (debug) System.out.println("Created Projection Volume");
		computeOffsets();
		
		if(motionField==null)
			motionField =  MotionUtil.get4DSpline();
		int reconDimensionX = getGeometry().getReconDimensionX();
		int reconDimensionY = getGeometry().getReconDimensionY();
		int reconDimensionZ = getGeometry().getReconDimensionZ();
		double voxelSpacingX = getGeometry().getVoxelSpacingX();
		double voxelSpacingY = getGeometry().getVoxelSpacingY();
		double voxelSpacingZ = getGeometry().getVoxelSpacingZ();
		int subVolumeReconDimensionX = (int) Math.ceil((reconDimensionX + 0.0) / numThreads);
		int lastSubVolumeReconDimensionX = reconDimensionX - ((numThreads - 1) * subVolumeReconDimensionX);
		for (int i = 0; i < numThreads; i++){
			if (debug) {
				projectors[i].setMaxI(subVolumeReconDimensionX);
				projectors[i].initializeProjectionVolume();
			} else {
				if (i < numThreads - 1)
					subVolumes[i] = new Grid3D(subVolumeReconDimensionX,reconDimensionY,reconDimensionZ);
				else
					subVolumes[i] = new Grid3D(lastSubVolumeReconDimensionX,reconDimensionY,reconDimensionZ);
				projectors[i].setProjectionVolume(subVolumes[i]);
				projectors[i].offsetX = (-getGeometry().getOriginX()) - (i * subVolumeReconDimensionX * voxelSpacingX) ;
				projectors[i].offsetY = ((reconDimensionY-1) * voxelSpacingY) / 2;
				projectors[i].offsetZ = ((reconDimensionZ-1) * voxelSpacingZ) / 2;
				if (i < numThreads - 1) {
					projectors[i].setMaxI(subVolumeReconDimensionX);
				} else {
					projectors[i].setMaxI(lastSubVolumeReconDimensionX);					
				}
				projectors[i].maxK = this.maxK;
				projectors[i].maxJ = this.maxJ;
				projectors[i].lineOffset = lineOffset;
				if(projectors[i].useVOImap && projectors[i].interestedInVolume!=null){
					projectors[i].setFastVOIMode(false);
					projectors[i].initializeVOIMap();
				}
				((MotionCompensatedVOIBasedReconstructionFilterMatrix)projectors[i]).setMotionField(motionField);
				projectors[i].init = true;
			}

		}
		volumeRewritten = false;
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		motionField = null;
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
	public String getToolName(){
		return "Motion-compensated Subvolume CPU-based Backprojector";
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/