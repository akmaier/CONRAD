package edu.stanford.rsl.conrad.phantom.workers;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.phantom.SheppLogan3D;
import edu.stanford.rsl.conrad.utils.Configuration;
import ij.process.FloatProcessor;

/**
 * Wrapper class to create a volume phantom from the SheppLogan3D Class.
 * 
 * @author Andreas Maier
 *
 */
public class SheppLoganPhantomWorker extends SliceWorker {

	private SheppLogan3D shepplogan = new SheppLogan3D();

	@Override
	public void workOnSlice(int k) {
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		int dimx = traj.getReconDimensionX();
		int dimy = traj.getReconDimensionY();
		int dimz = traj.getReconDimensionZ();
		FloatProcessor current = new FloatProcessor(dimx, dimy);
		for (int i=0; i< dimx; i++) {
			for (int j = 0; j< dimy; j++){
				double value = shepplogan.ImageDomainSignal(((float)((dimx/2) - i))/(dimx/2), 
						((float)((dimy/2) - j))/(dimy/2), 
						((float)((dimz/2) - k))/(dimz/2));
				current.putPixelValue(i, j, value);
			}
		}
		Grid2D grid = new Grid2D((float[])current.getPixels(), current.getWidth(), current.getHeight());
		this.imageBuffer.add(grid, k);
	}

	@Override
	public String getProcessName() {
		return "Shepp-Logan Phantom";
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@Article{Shepp80-CTA,\n" +
		"  author = {{Shepp}, L. A.},\n" +
		"  title = {{Computerized tomography and nuclear magnetic resonance}},\n" +
		"  journal = {Journal of Computer Assisted Tomography},\n" +
		"  volume = {4},\n" +
		"  pages = {94-107},\n" +
		"  year = {1980}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Shepp LA. Computerized tomography and nuclear magnetic resonance. J Comput Assist Tomogr 4:94�107. 1980";
	}

	@Override
	public SliceWorker clone() {
		return new SheppLoganPhantomWorker();
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/