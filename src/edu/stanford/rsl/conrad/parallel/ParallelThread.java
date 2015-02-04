package edu.stanford.rsl.conrad.parallel;

import java.util.concurrent.CountDownLatch;

/**
 * Thread to be run with a ParallelThreadExecutor. Wraps the run method using the abstract method execute.
 * Avoids deadlocks as exceptions in execute are caught and displayed. Nonetheless, the count down in the latch is performed. Hence, the ParallelThreadExecutor will not wait until the end of time.
 * 
 * @author akmaier
 *
 */
public abstract class ParallelThread extends Thread implements
		NamedParallelizableRunnable {

	CountDownLatch latch;

	@Override
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}
	
	@Override
	public void run(){
		try {
			execute();
		} catch (Exception e){
			e.printStackTrace();
		}
		latch.countDown();
	}

	/**
	 * Defines the code to be executed. Is called from the run method of the Thread.
	 */
	abstract public void execute();
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/