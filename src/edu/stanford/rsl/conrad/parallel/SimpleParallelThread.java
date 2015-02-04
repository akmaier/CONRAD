package edu.stanford.rsl.conrad.parallel;

public abstract class SimpleParallelThread extends ParallelThread{
	protected int threadNum;

	public SimpleParallelThread (int threadNum){
		this.threadNum = threadNum;
	}
	
	@Override
	public String getProcessName() {
		return "SimpleParallelThread " + threadNum;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/