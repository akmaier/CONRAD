package edu.stanford.rsl.conrad.phantom.renderer;

import java.util.ArrayList;
import java.util.Collections;

import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantomProjectorWorker;
import edu.stanford.rsl.conrad.phantom.workers.SliceWorker;
import edu.stanford.rsl.conrad.physics.absorption.AbsorptionModel;
import edu.stanford.rsl.conrad.physics.detector.XRayDetector;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;

/**
 * Class to enable parallel processing of slices of the target volume.
 * 
 * @author akmaier
 *
 */
public class ParallelProjectionPhantomRenderer extends SliceParallelVolumePhantomRenderer {

	@Override
	public String toString() {
		if (modelWorker != null) { 
			return modelWorker.toString();
		} else {
			return "Projection Phantom";
		}
	}

	@Override
	public void configure() throws Exception{
		buffer = new ImageGridBuffer();
		projectionNumber = -1;
		dimx = Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth();
		dimy = Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight();
		dimz = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize() * Configuration.getGlobalConfiguration().getNumSweeps();
		originIndexX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		originIndexY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		originIndexZ = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
		
		modelWorker = new AnalyticPhantomProjectorWorker();
	    configureWorkers();
		configured = true;
	}

	public void configure(AnalyticPhantom phan, XRayDetector model) throws Exception{
		buffer = new ImageGridBuffer();
		projectionNumber = -1;
		dimx = Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth();
		dimy = Configuration.getGlobalConfiguration().getGeometry().getDetectorHeight();
		dimz = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize() * Configuration.getGlobalConfiguration().getNumSweeps();
		originIndexX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		originIndexY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		originIndexZ = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
		
		modelWorker = new AnalyticPhantomProjectorWorker();
	    configureWorkers(phan, model);
		configured = true;
	}
	
	protected void configureWorkers(AnalyticPhantom phan, XRayDetector model) throws Exception{
		modelWorker.setImageProcessorBuffer(buffer);
		AnalyticPhantomProjectorWorker worker = (AnalyticPhantomProjectorWorker)modelWorker;
		worker.configure(phan, model);
		ArrayList<Integer> processors = new ArrayList<Integer>();
		for (int i = 0; i < dimz; i++){
			processors.add(new Integer(i));
		}
		modelWorker.setSliceList(Collections.synchronizedList(processors).iterator());
	}
	
	/**
	 * Reports a list of all known subclasses of SliceWorker
	 * @return the worker list.
	 */
	public static SliceWorker [] getAvailableSliceWorkers(){
		return new SliceWorker [] {
				new AnalyticPhantomProjectorWorker()
		};
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/