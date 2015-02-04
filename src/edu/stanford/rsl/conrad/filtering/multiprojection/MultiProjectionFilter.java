/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.filtering.multiprojection;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.pipeline.ProjectionSink;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;


/**
 * Abstract class to handle multi projection filter. The complete projection stack must be streamed into this filter using it as a ProjectionSink.
 * All projections are buffered internally. As soon as the context constraint gets valid the filter is available as a
 * ProjectionSource. The garbage collector is invoked after the processing of the data is finished.
 * 
 * If memory limitations are a problem the garbage collector may also be invoked after collecting some data from the ProjectionSource as the
 * references are set to null internally during this process.
 * 
 * @author Andreas Maier
 */
public abstract class MultiProjectionFilter extends ImageFilteringTool
implements ProjectionSink, Runnable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2775871921675277534L;
	protected ImageGridBuffer inputQueue;
	protected ArrayList<Boolean> processed;
	protected ImageGridBuffer outputQueue;
	protected ProjectionSink sink;
	protected int context = 3;

	public void setContext(int context){
		this.context = context;
	}

	public int getContext(){
		return context;
	}

	protected boolean showStatus = false;
	private boolean closed = false;
	private int finalIndex = -1;
	boolean init = false;

	public void setShowStatus(boolean showStatus){
		this.showStatus = showStatus;
	}
	
	protected int getFinalIndex(){
		return finalIndex;
	}
	
	protected int debug = 1;

	public MultiProjectionFilter(){
	}

	/**
	 * Used to connect the filter with the rest of the pipeline.
	 * @param sink
	 */
	public void setSink(ProjectionSink sink){
		this.sink = sink;
	}

	/**
	 * Packs the filter into a Thread and starts it.
	 */
	public void start(){
		Thread thread = new Thread(this);
		thread.start();
	}

	@Override 
	public void prepareForSerialization(){
		inputQueue = null;
		outputQueue = null;
		processed = null;
		sink = null;
		init = false;
		finalIndex = -1;
		closed = false;
	}

	@Override
	public synchronized void process(Grid2D projection, int projectionNumber)
	throws Exception {
		init();
		if (inputQueue == null){
			throw new Exception ("InputQueue was null");
		}
		inputQueue.add(projection, projectionNumber);
		processed.add(new Boolean(false));
	}

	public void run(){
		int processedIndex = 0;
		boolean init = this.init;
		while (!init){
			try {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				init = this.init;
				if (debug > 2)System.out.println(init);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		if (debug > 1)System.out.println("MultiProjectionFilter: Processing");
		boolean allProjectionsProcessed = false;
		while (!allProjectionsProcessed){
			if (debug > 2) System.out.println("MultiProjectionFilter: Processing Data. Queue Size: " + (processed.size() - processedIndex) );
			if (debug > 2) System.out.println("MultiProjectionFilter: Processing Data at Projection " + processedIndex );
			// Iterate over all Images in buffer.
			if (processedIndex < processed.size()) {
				if (processed.get(processedIndex) != null){
					if (!processed.get(processedIndex).booleanValue()) {
						// found an image which is not yet processed
						boolean available = isContextAvailable(processedIndex);
						if (debug > 2) System.out.println("MultiProjectionFilter: Available " + available );
						if (available) {
							try {
								processProjectionData(processedIndex);
								processed.set(processedIndex, new Boolean(true));
								processedIndex++;
							} catch (Exception e) {
								// Index must be increased otherwise we end in a deadlock.
								e.printStackTrace();
								processed.set(processedIndex, new Boolean(true));
								processedIndex++;
							}
						}
					}
				}
			}
			if (closed) {
				if (debug > 2) System.out.println("MultiProjectionFilter: All data arrived.");
				// Are we done yet?
				if (processedIndex == finalIndex) {
					if (debug > 1)System.out.println("MultiProjectionFilter: All Data Streamed");
					break;
				}
			}
			try {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		if (debug > 1)System.out.println("MultiProjectionFilter: Finished Cleaning up ...");
		cleanup();
		if(debug>1) System.out.println("MultiProjectionFilter processed " + (processedIndex) + " of " + finalIndex + " Projections.");
	}

	private synchronized void init(){
		if (debug > 2)System.out.println("MultiProjectionFilter: init " +init);
		if (!init){
			// Create arrays
			inputQueue = new ImageGridBuffer();
			outputQueue = new ImageGridBuffer();
			processed = new ArrayList<Boolean>();
			// done.
			init = true;
		}
	}

	protected void cleanup(){
		if (debug > 1) System.out.println("MultiProjectionFilter: Cleaning up");
		// remove old data.
		for(int i = inputQueue.size()-1; i >= 0; i--){
			inputQueue.remove(i);
		}
		inputQueue = null;
		outputQueue = null;
		processed = null;
		init = false;
		// close sink;
		try {
			if(debug>1)System.out.println("Closing next");
			sink.close();
			if(debug>1)System.out.println("Closed next");
			sink = null;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// invoke garbage collection
		CONRAD.gc();

	}

	/**
	 * Calculates the lower limit of the current context and the given projection.
	 * Is used by the sub classes of MultiprojectionFilter to iterate over the
	 * inputQueue buffer. Accessible projections are greater or equal to the return value of
	 * this method.<br>
	 * The sub-classes use the following structure:<br>
	 * <pre>
	 * for (int i=lowerEnd(projectionNumber); i< upperEnd(projectionNumber); i++{
	 *    // Filtering steps for the current block.
	 * }
	 * <pre>
	 * 
	 * @param projectionNumber the projectionNumber which defines the context.
	 * @return the lower limit.
	 */
	protected int lowerEnd(int projectionNumber){
		int lowerEnd = projectionNumber - context; 
		if (lowerEnd < 0) lowerEnd = 0;
		return lowerEnd;
	}
	
	/**
	 * Calculates the upper limit of the current context and the given projection.
	 * Is used by the sub classes of MultiprojectionFilter to iterate over the
	 * inputQueue buffer. Accessible projections are less than the return value of
	 * this method.<br>
	 * The sub-classes use the following structure:<br>
	 * <pre>
	 * for (int i=lowerEnd(projectionNumber); i< upperEnd(projectionNumber); i++{
	 *    // Filtering steps for the current block.
	 * }
	 * <pre>
	 * 
	 * @param projectionNumber the projectionNumber which defines the context.
	 * @return the upper limit.
	 */
	protected int upperEnd(int projectionNumber){
		int upperEnd = projectionNumber + context +1;
		if (closed){
			if (upperEnd >= finalIndex) upperEnd = finalIndex;
		}
		return upperEnd;
	}
	
	/**
	 * Determines whether the current context around projectionNumber touches the end of the stream;
	 * <BR><BR>
	 * Contexts at the end of the stream will be shorter than (context * 2) + 1.<br>
	 * Some sub classes require altered processing in these cases. 
	 * 
	 * @param projectionNumber the projectionNumber
	 * @return whether the context touches the end of the projection stream.
	 */
	protected boolean isLastBlock(int projectionNumber){
		boolean lastBlock = false;
		if (closed) {
			if (upperEnd(projectionNumber) == finalIndex) lastBlock = true;
		}
		return lastBlock;
	}
	
	/**
	 * Checks whether the required context of the projectionNumber is available.<br>
	 * For context = 3, three projections to the left and three projections to the right
	 * must be available. Intervals are cut-off at the beginning and the end of the stream,<br>
	 * @param projectionNumber the number to the projection to be processed
	 * @return whether this projection's context is available
	 */
	protected boolean isContextAvailable(int projectionNumber){
		int lowerEnd = projectionNumber - context;
		if (lowerEnd < 0) lowerEnd = 0;
		int upperEnd = projectionNumber + context + 1;
		boolean revan = true;
		if (!closed){
			if (upperEnd >= inputQueue.size()) revan = false;
		} else {
			if (upperEnd >= finalIndex) upperEnd = finalIndex;
		}
		if (revan) {
			for (int i = lowerEnd ; i <upperEnd; i++){
				if (inputQueue.get(i) == null) revan = false;
			}
		}
		return revan;
	}

	@Override
	public void close(){
		if (debug > 1) System.out.println("Closed " + closed + " " + finalIndex + " "+ inputQueue.size());
		if (!closed){
			finalIndex = inputQueue.size();
			closed = true;
			if (debug > 1) System.out.println("MultiProjectionFilter done: finalIndex = " + finalIndex);
			//System.exit(0);
		}
	}

	/**
	 * Processes the data from the protected array inputQueue.
	 * Results are written into the ProjectionSink in this method internally.
	 * Data from input queue should be discarded in the actual implementation in order to save memory.
	 * 
	 * @param projectionNumber the projection to process
	 * @throws Exception may happen
	 */
	protected abstract void processProjectionData(int projectionNumber) throws Exception;

	@Override
	public String getName(){
		return getToolName();
	}

	/**
	 * feeds the filter from a projection Source.
	 * @param source the source
	 * @param showStatus displays whether the status should be displayed using ImageJ
	 * @throws Exception may happen.
	 */
	public void feedFilter(ProjectionSource source, boolean showStatus) throws Exception{
		// Stream data into filter
		int stackSize = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize();
		for(int i=0; i<stackSize; i++){
			process(source.getNextProjection(), i);
		}
	}

}

