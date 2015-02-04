package edu.stanford.rsl.conrad.parallel;

import java.util.concurrent.CountDownLatch;

/**
 * Interface for parallel runnables. Each runnable is required to have a CountDownLatch.
 * When the processing is done, the Runnable should call latch.countDown() in order
 * to inform the dispatching ParallelThreadExecutor about having done the computation.
 * 
 * @author Maier
 *
 */
public interface ParallelizableRunnable extends Runnable{
	public void setLatch(CountDownLatch latch);
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/