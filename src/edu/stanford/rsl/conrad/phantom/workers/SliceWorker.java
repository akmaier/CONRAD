package edu.stanford.rsl.conrad.phantom.workers;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.parallel.NamedParallelizableRunnable;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;

/**
 * Class to model an abstract slice worker.
 * 
 * @author akmaier
 *
 */
public abstract class SliceWorker implements NamedParallelizableRunnable, Cloneable, GUIConfigurable, Citeable {

	private CountDownLatch latch;
	private Iterator<Integer> sliceList;
	protected ImageGridBuffer imageBuffer;
	private boolean configured = false;
	protected boolean showStatus = true;


	@Override
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}

	@Override
	public void run() {
		// Must guarantee the call of latch.countDown(). Otherwise program may not terminate.
		try {
			Integer image = sliceList.next();
			while (image != null){
				int i = image.intValue();
				if (showStatus) {
					//IJ.showProgress(((double) i)/size);
					//IJ.showStatus(getProcessName());
					//CONRAD.log("Processing " + i);
				}
				workOnSlice(i);
				if (sliceList.hasNext()) {
					image = sliceList.next();
				} else {
					image = null;
				}
			}
		} catch (NoSuchElementException ele){
			System.out.println("Other processes faster than queue...");
		}catch (Exception e){
			e.printStackTrace();
		}
		latch.countDown();
	}

	/**
	 * Sets the volume to work on.
	 * @param image the image buffer
	 */
	public void setImageProcessorBuffer(ImageGridBuffer image){
		imageBuffer = image;
	}
	
	public ImageGridBuffer getImageProcessorBufferValue(){
		return imageBuffer;
	}


	/**
	 * Sets the sliceList which is being processed
	 * @param sliceList the list of slice as Iterator of slice numbers
	 */
	public void setSliceList(Iterator<Integer> sliceList){
		this.sliceList = sliceList;
	}

	/**
	 * Method will voxelize the phantom into the current slice
	 * @param sliceNumber the slice number
	 */
	public abstract void workOnSlice(int sliceNumber);

	@Override
	public abstract SliceWorker clone();

	/**
	 * Method to be called in the clones of Subclasses to copy the information of this class into the new clone.
	 * @param other the clone of the subclass.
	 */
	public void copyInternalElementsTo(SliceWorker other){
		other.latch = latch;
		other.imageBuffer = imageBuffer;
		other.sliceList = sliceList;
		other.configured = configured;
	}

	/**
	 * Reports a list of all known subclasses of SliceWorker
	 * @return the worker list.
	 */
	public static SliceWorker [] getAvailableSliceWorkers(){
		return new SliceWorker [] {
				new SheppLoganPhantomWorker(),
				new DiracProjectionPhantom()
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
		//System.out.println(name);
		for (SliceWorker w: workers){
			//System.out.println(w);
			if (w.toString().equals(name)) worker = w;
		}
		return worker;
	}

	@Override
	public void configure() throws Exception{
		configured = true;
	}

	@Override
	public boolean isConfigured(){
		return configured;
	}

	@Override
	public String toString(){
		return getProcessName();
	}

	/**
	 * @return the showStatus
	 */
	public boolean isShowStatus() {
		return showStatus;
	}

	/**
	 * @param showStatus the showStatus to set
	 */
	public void setShowStatus(boolean showStatus) {
		this.showStatus = showStatus;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/