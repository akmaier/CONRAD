package edu.stanford.rsl.conrad.phantom.workers;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.reconstruction.voi.VolumeOfInterest;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import ij.process.FloatProcessor;

public class VolumeOfInterestPhantom extends SliceWorker {

	private VolumeOfInterest voi = null;
	private Trajectory geometry = null;

	@Override
	public void workOnSlice(int k) {
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		int dimx = traj.getReconDimensionX();
		int dimy = traj.getReconDimensionY();
		int dimz = traj.getReconDimensionZ();
		FloatProcessor current = new FloatProcessor(dimx, dimy);
		double offsetX = (dimx /2) * geometry.getVoxelSpacingX();
		double offsetY = (dimy /2) * geometry.getVoxelSpacingY();
		double offsetZ = (dimz /2) * geometry.getVoxelSpacingZ();
		for (int i=0; i< dimx; i++) {
			for (int j = 0; j< dimy; j++){
				double value = 0;
				if (voi.contains((this.geometry.getVoxelSpacingX() * i) - offsetX, (this.geometry.getVoxelSpacingY() * j) - offsetY, (this.geometry.getVoxelSpacingZ() * k) - offsetZ)) value = 1;
				current.putPixelValue(i, j, value);
			}
		}
		Grid2D grid = new Grid2D((float[])current.getPixels(), current.getWidth(), current.getHeight());
		this.imageBuffer.add(grid, k);
	}

	@Override
	public String getProcessName() {
		return "Volume of Interest Phantom";
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
	public void configure() throws Exception{
		String file = FileUtil.myFileChoose(".txt", false);
		voi = VolumeOfInterest.openAsVolume(file);
		geometry = Configuration.getGlobalConfiguration().getGeometry();
		super.configure();
	}

	@Override
	public SliceWorker clone() {
		VolumeOfInterestPhantom clone = new VolumeOfInterestPhantom();
		clone.geometry = geometry;
		clone.voi = voi;
		return clone;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/