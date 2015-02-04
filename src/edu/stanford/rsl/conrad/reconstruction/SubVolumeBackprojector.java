package edu.stanford.rsl.conrad.reconstruction;

import java.text.NumberFormat;
import java.util.Locale;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.parallel.ParallelizableRunnable;
import edu.stanford.rsl.conrad.parallel.SimpleParallelThread;
import edu.stanford.rsl.conrad.reconstruction.voi.VolumeOfInterest;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.RegKeys;

/**
 * This FBP-based mathod splits the reconstruction volume into sub volumes which can be processed in parallel to speed up the reconstruction further.
 * @author akmaier
 *
 */
public class SubVolumeBackprojector extends VOIBasedReconstructionFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3452972435131680120L;
	protected int numThreads = - 1;
	protected VOIBasedReconstructionFilter [] projectors;
	protected Grid3D [] subVolumes;
	protected boolean debug = false;

	public SubVolumeBackprojector() {
		numThreads = CONRAD.getNumberOfThreads();
		projectors = new VOIBasedReconstructionFilter[numThreads];
		subVolumes = new Grid3D[numThreads];
		for (int i = 0; i < numThreads; i++){
			projectors[i] = new VOIBasedReconstructionFilter();
		}
	}

	@Override
	public String getName(){
		return "Parallel CPU-based Sub-volume Backprojector";
	}

	public void configure() throws Exception{
		Configuration config = Configuration.getGlobalConfiguration();
		boolean success = true;
		NumberFormat nf = NumberFormat.getInstance(Locale.US);
		nf.setMaximumFractionDigits(1);
		nf.setMinimumFractionDigits(1);
		nf.setMaximumIntegerDigits(1);
		nf.setMinimumIntegerDigits(1);
		if (debug) System.out.println("loading");
		// volume of interest is optional
		if (config.getVolumeOfInterestFileName() != null) {
			this.setMaximumVolumeOfInterest(config.getVolumeOfInterestFileName());
		} else {
			String filename = config.getRegistry().get(RegKeys.PATH_TO_CALIBRATION) + "/" + config.getDeviceSerialNumber() + "/active/-903/" + "/d/" + config.getIntensifierSize() + "/" + nf.format(config.getGeometry().getAverageAngularIncrement()) + "/maxVOI.txt";
			this.setMaximumVolumeOfInterest(filename);
		}
		for (int i = 0; i < numThreads; i++){
			projectors[i].setMaximumVolumeOfInterest(interestedInVolume);
			System.out.println("new Geometry");
		}
		initializeProjectionVolume();
		configured = success;
	}

	public void initializeProjectionVolume(){
		super.initializeProjectionVolume();
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
				projectors[i].offsetY = -getGeometry().getOriginY();
				projectors[i].offsetZ = -getGeometry().getOriginZ();
				projectors[i].setMaxI(subVolumeReconDimensionX);
				projectors[i].maxK = this.maxK;
				projectors[i].maxJ = this.maxJ;
				projectors[i].lineOffset = lineOffset;
				projectors[i].init = true;
				if (i < numThreads - 1) {
					projectors[i].setMaxI(subVolumeReconDimensionX);
				} else {
					projectors[i].setMaxI(lastSubVolumeReconDimensionX);
				}
			}

		}
		volumeRewritten = false;
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		for (int i = 0; i < numThreads; i++){
			subVolumes[i] = null;
			if (projectors[i] != null) projectors[i].prepareForSerialization();
		}
		volumeRewritten = false;
	}

	protected boolean volumeRewritten = false;

	@Override
	protected void reconstruct() throws Exception {

		if (!init){
			initialize(inputQueue.get(0));
		}
		initializeProjectionVolume();
		ParallelizableRunnable [] runnables = new ParallelizableRunnable[numThreads];
		for (int j= 0; j<numThreads; j++) {
			runnables[j]= new SimpleParallelThread(j) {
				@Override
				public void execute() {
					for (int i = 0; i < nImages; i++){
						projectors[threadNum].backproject(inputQueue.get(i), i);
					}
				}
			};  	
		}
		ParallelThreadExecutor executor = new ParallelThreadExecutor(runnables);
		executor.execute();
		if (!volumeRewritten) {
			//System.out.println("Rewriting Volume");
			int subVolumeReconDimensionX = (int) Math.ceil((getGeometry().getReconDimensionX() + 0.0) / numThreads);
			int lastSubVolumeReconDimensionX = getGeometry().getReconDimensionX() - ((numThreads - 1) * subVolumeReconDimensionX);
			for (int i = 0; i < numThreads; i++){
				int offsetx = subVolumeReconDimensionX * i;
				int currentlimit = subVolumeReconDimensionX;
				if (i == numThreads - 1) currentlimit = lastSubVolumeReconDimensionX;
				for (int z = 0; z < getGeometry().getReconDimensionZ(); z ++){
					for (int y = 0; y < getGeometry().getReconDimensionY(); y++){
						for (int x = 0; x < currentlimit; x ++){
							float value = subVolumes[i].getAtIndex(x, y, z);
							projectionVolume.setAtIndex(x + offsetx, y, z, value);
						}
					}
				}
			}
			volumeRewritten = true;
		}
		if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) applyHounsfieldScaling();
		for (int k = 0; k < projectionVolume.getSize()[2]; k++){
			sink.process(projectionVolume.getSubGrid(k), k);
		}
		init = false;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@inproceedings{Scherl07-FGB,\n" +
		"  author = {{Scherl}, H. and {Keck}, B. and {Kowarschik}, M. and {Hornegger}, J.},\n" +
		"  title = {{Fast GPU-Based CT Reconstruction using the Common Unified Device Architecture (CUDA)}},\n" +
		"  booktitle = {{Nuclear Science Symposium, Medical Imaging Conference 2007}},\n" +
		"  publisher = {IEEE},\n" +
		"  volume={6},\n" +
		"  address = {Honolulu, HI, United States},\n" +
		"  year = {2007}\n" +
		"  pages= {4464--4466},\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Scherl H, Keck B, Kowarschik M, Hornegger J. Fast GPU-Based CT Reconstruction using the Common Unified Device Architecture (CUDA). In Nuclear Science Symposium, Medical Imaging Conference Record, IEEE, Honolulu, HI, United States, 2008 6:4464-6.";
	}


	@Override
	public String getToolName(){
		return "Subvolume CPU-based Backprojector";
	}
	
	@Override
	public void setMaximumVolumeOfInterest(VolumeOfInterest maxVOI) {
		for (int i = 0; i < projectors.length; i++) {
			projectors[i].setMaximumVolumeOfInterest(maxVOI);
			projectors[i].useVOImap = true;
		}
	}
	
	@Override
	public void setFastVOIMode(boolean fastVOIMode) {
		for (int i = 0; i < projectors.length; i++) {
			projectors[i].setFastVOIMode(fastVOIMode);
		}
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/