import ij.IJ;
import ij.ImagePlus;
import ij.measure.Calibration;
import ij.plugin.PlugIn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSplineVolumePhantom;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.renderer.MetricPhantomRenderer;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;


public class Nurbs_Reader extends ImagePlus implements PlugIn {

	public void run(String arg) {
		try {
			SurfaceBSplineVolumePhantom phantomWorker = new SurfaceBSplineVolumePhantom();
			//int dimx = Configuration.getGlobalConfiguration().getGeometry().getReconDimensionX();
			//int dimy = Configuration.getGlobalConfiguration().getGeometry().getReconDimensionY();
			int dimz = Configuration.getGlobalConfiguration().getGeometry().getReconDimensionZ();
			ImageGridBuffer buffer = new ImageGridBuffer();
			phantomWorker.setImageProcessorBuffer(buffer);
			phantomWorker.readSplineListFromFile(arg);
			phantomWorker.resizeVolumeToMatchSplineSpace();
			phantomWorker.generateDefaultScene(64, 64);
			
			SimpleVector bounds = phantomWorker.getBounds();
			System.out.println(bounds);
			MetricPhantomRenderer phantom = new MetricPhantomRenderer();
			
			ArrayList<Integer> processors = new ArrayList<Integer>();
			for (int i = 0; i < dimz; i++){
				processors.add(new Integer(i));
			}
			phantomWorker.setSliceList(Collections.synchronizedList(processors).iterator());
			phantom.setModelWorker(phantomWorker);
			phantom.createPhantom();
			ImagePlus renderedBSpline = buffer.toImagePlus(arg);
			setStack(renderedBSpline.getStack());
			setTitle(renderedBSpline.getTitle());
			Calibration cal = renderedBSpline.getCalibration();
			cal.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
			cal.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
			cal.zOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
			cal.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
			cal.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
			cal.pixelDepth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
			setCalibration(renderedBSpline.getCalibration());
			IJ.showProgress(1.0);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
