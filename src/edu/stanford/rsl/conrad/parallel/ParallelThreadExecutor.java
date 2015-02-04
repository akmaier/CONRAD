package edu.stanford.rsl.conrad.parallel;

import ij.IJ;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import edu.stanford.rsl.conrad.utils.CONRAD;


/**
 * Class to execute multiple ParallelizeableRunnables in parallel. The Executor will dispatch
 * as many ParallelizedRunnables as there are CPUs in the machine on which the code is executed in one batch.
 * Then it will wait until all processes in the current batch are done. As soon as this is the case a new batch
 * of processes is started. This is done until all ParallelizableRunnables are processed.
 * 
 * @author Andreas Maier
 *
 */
public class ParallelThreadExecutor {

	private boolean showStatus;
	ParallelizableRunnable [] runnables;
	boolean debug = false;
	private CountDownLatch latch;
	public static boolean parallel = true;

	public void setShowStatus(boolean showStatus) {
		this.showStatus = showStatus;
	}

	public boolean isShowStatus() {
		return showStatus;
	}

	/**
	 * In order to have the threads performed in parallel just an Array of ParallelizableRunnables is
	 * passed to the contructor of the ParallelThreadExecutor.
	 * @param runnables the processes to run.
	 */
	public ParallelThreadExecutor(ParallelizableRunnable [] runnables){
		this.runnables = runnables;
	}

	public ParallelThreadExecutor(ParallelizableRunnable [] runnables, int latchSize){
		this.runnables = runnables;
		latch = new CountDownLatch(latchSize);
	}

	/**
	 * This method will start the processing.
	 * @throws InterruptedException may occur.
	 */
	public void execute() throws InterruptedException{
		int numThreads = CONRAD.getNumberOfThreads();
		if (showStatus) CONRAD.log("Number of used processors: " + numThreads);
		if (numThreads > 7) if (showStatus) CONRAD.log("I like this machine ... ");
		Future <?> [] futures = new Future<?>[runnables.length];
		if (debug) System.out.println("Starting new batch ...");
		ExecutorService e = Executors.newFixedThreadPool(numThreads);
		// initialize the parallel processing.
		long latchSize = 0; 
		if(latch == null) {
			latch = new CountDownLatch(runnables.length);
			latchSize = runnables.length;
		} else {
			latchSize = latch.getCount();
		}
		if (parallel) {
			parallel = false;
			// invoke the threads
			for (int i = 0; i < runnables.length; i++){
				runnables[i].setLatch(latch);
				futures[i] = e.submit(runnables[i]);
			}
			// wait for all jobs to be done
			while (latch.getCount() > 0){
				if (showStatus){
					int i = (int) (latchSize - latch.getCount());
					if (i < runnables.length){
						if (runnables[i] instanceof NamedParallelizableRunnable)IJ.showStatus("Running " + ((NamedParallelizableRunnable)runnables[i]).getProcessName());
					} 
					IJ.showProgress((i + 0.0) / latchSize);
				}
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			}
			e.awaitTermination(1000, TimeUnit.MILLISECONDS);//e.shutdownNow();
			e = null;
			if (showStatus) IJ.showProgress(1.0);
			parallel = true;
		} else {
			//System.out.println("Debug mode. Invoking sequentially.");
			for (int i = 0; i < runnables.length; i++){
				runnables[i].setLatch(latch);
				//System.out.println("Thread " + i);
				runnables[i].run();
			}
			//System.out.println("All done.");
		}
	}
		
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/