/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.renderer;

import edu.stanford.rsl.conrad.geometry.splines.SurfaceBSplineVolumePhantom;
import edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantom3DVolumeRenderer;
import edu.stanford.rsl.conrad.phantom.workers.SliceWorker;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class MetricPhantomRenderer extends SliceParallelVolumePhantomRenderer {

	protected double voxelSizeX = 0.5;
	protected double voxelSizeY = 0.5;
	protected double voxelSizeZ = 0.5;

	@Override
	public void configure() throws Exception {
		Configuration config = Configuration.getGlobalConfiguration();
		projectionNumber = -1;
		buffer = new ImageGridBuffer();
		readDimensionsFromGlobalConfig();
		voxelSizeX = config.getGeometry().getVoxelSpacingX();
		voxelSizeY = config.getGeometry().getVoxelSpacingY();
		voxelSizeZ = config.getGeometry().getVoxelSpacingZ();
		modelWorker = (SliceWorker) UserUtil.queryObject("Select slice worker", "Worker Selection", SliceWorker.class);
		configureWorkers();
		configured = true;
	}
	
	/**
	 * Reports a list of all known subclasses of SliceWorker for physical phantoms 
	 * @return the worker list.
	 */
	public static SliceWorker [] getAvailableSliceWorkers(){
		return new SliceWorker [] {
			new SurfaceBSplineVolumePhantom(),
			new AnalyticPhantom3DVolumeRenderer()
			//new AnalyticPhantomProjectorWorker()
		};
	}
	
	/**
	 * Gives a list of the available Workers as String []
	 * @return the worker names;
	 */
	public static String [] getAvailableWorkersAsString(){
		SliceWorker [] workers = getAvailableSliceWorkers();
		String [] strings = new String [workers.length];
		for (int i = 0; i < workers.length; i++){
			strings[i] = workers[i].toString();
		}
		return strings;
	}
	
	/**
	 * Method to select a worker given it's String representation
	 * @param name the String
	 * @return the Worker
	 */
	public static SliceWorker getWorkerFromString(String name){
		SliceWorker worker = null;
		SliceWorker [] workers = getAvailableSliceWorkers();
		for (SliceWorker w: workers){
			if (w.toString().equals(name)) worker = w;
		}
		return worker;
	}

	@Override
	public String toString(){
		if (modelWorker != null) { 
			return modelWorker.toString();
		} else {
			return "Metric Volume Phantom";
		}
	}

}
