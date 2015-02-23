/*
 * Copyright (C) 2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.motion;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.opencl.OpenCLProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.motion.compensation.OpenCLMotionCompensatedBackProjector;

public class MotionCompensatedReconExample {

	public static void main(String[] args) {
		
		new ImageJ();
		Configuration.loadConfiguration();
		Trajectory geo = Configuration.getGlobalConfiguration().getGeometry();
		double focalLength = geo.getSourceToDetectorDistance();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		float D = (float) focalLength;
		
		try {
			OpenCLProjectionPhantomRenderer phantom = new OpenCLProjectionPhantomRenderer();
			phantom.configure();
			
			Grid3D projections = PhantomRenderer.generateProjections(phantom);
			
			
			projections.show(phantom.toString());
			
			ConeBeamCosineFilter cbFilter = new ConeBeamCosineFilter(focalLength, maxU, maxV, deltaU, deltaV);
			RamLakKernel ramK = new RamLakKernel(maxU_PX, deltaU);
			
			for(int i = 0; i < projections.getSize()[2]; i++){
				Grid2D proj = projections.getSubGrid(i);
				cbFilter.applyToGrid(proj);
				//ramp
				for (int j = 0;j <maxV_PX; ++j)
					ramK.applyToGrid(proj.getSubGrid(j));
				NumericPointwiseOperators.multiplyBy(proj, (float) (D*D * Math.PI / geo.getNumProjectionMatrices()));
			}

			
			OpenCLBackProjector projector = new OpenCLBackProjector();
			projector.loadInputQueue(projections);
			Grid3D recon = projector.reconstructCompleteQueue();
			recon.show("Recontruction");
			
			OpenCLMotionCompensatedBackProjector compensatedProjector = new OpenCLMotionCompensatedBackProjector();
			compensatedProjector.setMotion(((AnalyticPhantom4D)phantom.getPhantom()).getMotionField());
			compensatedProjector.loadInputQueue(projections);
			Grid3D compensatedRecon = compensatedProjector.reconstructCompleteQueue();
			compensatedRecon.show("Compensated Recontruction");
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		

	}

}
