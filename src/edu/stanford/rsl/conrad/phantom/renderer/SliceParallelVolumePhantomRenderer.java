package edu.stanford.rsl.conrad.phantom.renderer;

import java.util.ArrayList;
import java.util.Collections;

import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.phantom.workers.AnalyticPhantomProjectorWorker;
import edu.stanford.rsl.conrad.phantom.workers.SliceWorker;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;
import ij.gui.GenericDialog;

/**
 * Class to enable parallel processing of slices of the target volume.
 * 
 * @author akmaier
 *
 */
public class SliceParallelVolumePhantomRenderer extends StreamingPhantomRenderer {

	protected SliceWorker modelWorker;
	
	@Override
	public void createPhantom() {
		SliceWorker [] workers = new SliceWorker[CONRAD.getNumberOfThreads()];
		for (int i = 0; i < workers.length; i++){
			workers[i] = modelWorker.clone();
			modelWorker.copyInternalElementsTo(workers[i]);
		}
		ParallelThreadExecutor executor = new ParallelThreadExecutor(workers);
		executor.setShowStatus(false);
		try {
			executor.execute();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if (modelWorker instanceof AnalyticPhantomProjectorWorker){
			AnalyticPhantomProjectorWorker projectorWorker = (AnalyticPhantomProjectorWorker) modelWorker;
			projectorWorker.getDetector().notifyEndOfRendering();
		}
	}

	@Override
	public String toString() {
		if (modelWorker != null) { 
			return modelWorker.toString();
		} else {
			return "Numeric Volume Phantom";
		}
	}

	@Override
	public String getBibtexCitation() {
		return modelWorker.getBibtexCitation();
	}

	@Override
	public String getMedlineCitation() {
		return modelWorker.getMedlineCitation();
	}
	
	@Override
	public void configure() throws Exception{
		GenericDialog gd = createDimensionDialog();
		projectionNumber = -1;
		buffer = new ImageGridBuffer();
		String [] names = SliceWorker.getAvailableWorkersAsString();
		gd.addChoice("Select method:", names, names[0]);
		gd.showDialog();
		if (gd.wasCanceled()){
			throw new RuntimeException("User cancelled selection.");
		}
		readDimensions(gd);
		readWorker(gd);
		configureWorkers();
		configured = true;
	}
	
	/**
	 * reads the model worker from the dialog.
	 * @param gd
	 */
	protected void readWorker (GenericDialog gd){
		modelWorker = SliceWorker.getWorkerFromString(gd.getNextChoice());
	}
	
	/**
	 * Configures the model worker and creates clones.
	 * 
	 * @throws Exception
	 */
	protected void configureWorkers() throws Exception{
		modelWorker.setImageProcessorBuffer(buffer);
		modelWorker.configure();
		ArrayList<Integer> processors = new ArrayList<Integer>();
		for (int i = 0; i < dimz; i++){
			processors.add(new Integer(i));
		}
		modelWorker.setSliceList(Collections.synchronizedList(processors).iterator());
	}

	/**
	 * @return the modelWorker
	 */
	public SliceWorker getModelWorker() {
		return modelWorker;
	}

	/**
	 * @param modelWorker the modelWorker to set
	 */
	public void setModelWorker(SliceWorker modelWorker) {
		this.modelWorker = modelWorker;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/