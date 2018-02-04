package edu.stanford.rsl.tutorial.motion.estimation;

import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;

public class ProjectionLoader {

	private static ImageGridBuffer projections;

	public ImageGridBuffer getProjections() {
		return projections;
	}

	public void loadAndFilterImages(String filename) throws IOException {
		projections = new ImageGridBuffer();
		Configuration.loadConfiguration();
		Configuration c = Configuration.getGlobalConfiguration();
		Trajectory geo = c.getGeometry();
		double focalLength = geo.getSourceToDetectorDistance();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		float D = (float) focalLength;
		int maxProjs = geo.getProjectionStackSize();
		ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength, maxU, maxV, deltaU, deltaV);
		RamLakKernel ramK = new RamLakKernel(maxU_PX, deltaU);
		ProjectionSource pSource = FileProjectionSource.openProjectionStream(filename);
		Grid2D proj;
		for (int i = 0; i < maxProjs; i++) {
			proj = pSource.getNextProjection();
			cbFilter.applyToGrid(proj);
			//ramp
			for (int j = 0; j < maxV_PX; ++j)
				ramK.applyToGrid(proj.getSubGrid(j));
			NumericPointwiseOperators.multiplyBy(proj, (float) (D * D * Math.PI / geo.getNumProjectionMatrices()));
			projections.add(proj, i);
		}
	}

}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/