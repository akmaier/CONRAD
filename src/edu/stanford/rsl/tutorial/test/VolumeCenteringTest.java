package edu.stanford.rsl.tutorial.test;

import ij.ImagePlus;
import edu.stanford.rsl.apps.gui.XCatMetricPhantomCreator;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.opencl.OpenCLBackProjector;
import edu.stanford.rsl.conrad.opencl.OpenCLProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.phantom.renderer.MetricPhantomRenderer;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.RamLakKernel;
import edu.stanford.rsl.tutorial.cone.ConeBeamCosineFilter;

public class VolumeCenteringTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int timesteps = 2; 
		CONRAD.setup();
		Configuration config = Configuration.getGlobalConfiguration();
		Trajectory geom = config.getGeometry();
		PointND saveWorldOrigin = new PointND(geom.getOriginX(),geom.getOriginY(),geom.getOriginZ());
		
		XCatMetricPhantomCreator creator = new XCatMetricPhantomCreator();
		creator.setSteps(timesteps);
		AnalyticPhantom4D scene = creator.instantiateScene();
		ImagePlus hyper = creator.renderMetricVolumePhantom(scene);
		Grid3D hyperGrid = ImageUtil.wrapImagePlus(hyper);
		hyperGrid.show("Phantom Creator");
		
		geom.setOriginInWorld(saveWorldOrigin);
		
		MetricPhantomRenderer renderFromDialogBox = new MetricPhantomRenderer();
		try {
			renderFromDialogBox.configure();
			Grid3D slices = PhantomRenderer.generateProjections(renderFromDialogBox);
			
			
			slices.show(renderFromDialogBox.toString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		Trajectory geo = Configuration.getGlobalConfiguration().getGeometry();
		double focalLength = geo.getSourceToDetectorDistance();
		int maxU_PX = geo.getDetectorWidth();
		int maxV_PX = geo.getDetectorHeight();
		double deltaU = geo.getPixelDimensionX();
		double deltaV = geo.getPixelDimensionY();
		double maxU = (maxU_PX) * deltaU;
		double maxV = (maxV_PX) * deltaV;
		float D = (float) focalLength;
		
		
		OpenCLProjectionPhantomRenderer phantom = new OpenCLProjectionPhantomRenderer();
		try {
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
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		


	}

}
