package edu.stanford.rsl.conrad.pipeline;


import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.geometry.trajectories.MultiSweepTrajectory;
import edu.stanford.rsl.conrad.parallel.NamedParallelizableRunnable;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;

public class ParallelProjectionDataSinkFeeder  implements NamedParallelizableRunnable{
	
	private Grid2D projection = null;
	private int projectionNumber = 0;
	private ProjectionSink sink = null;
	
	public Grid2D getProjection() {
		return projection;
	}

	public void setProjection(Grid2D projection) {
		this.projection = projection;
	}

	public int getProjectionNumber() {
		return projectionNumber;
	}

	public void setProjectionNumber(int projectionNumber) {
		this.projectionNumber = projectionNumber;
	}

	public ProjectionSink getSink() {
		return sink;
	}

	public void setSink(ProjectionSink sink) {
		this.sink = sink;
	}
	
	@Override
	public String getProcessName() {
		String revan = "Parallel Projector";
		if (sink != null){
			revan += " " + sink.getName();
		}
		return revan;
	}

	@Override
	public void run() {
		try {
			sink.process(projection, projectionNumber);
		} catch (Exception e){
			e.printStackTrace();
		}
		latch.countDown();
	}

	protected CountDownLatch latch = null;
	
	@Override
	public void setLatch(CountDownLatch latch) {
		this.latch = latch;
	}
	
	public static void projectParallel(Grid3D projections, ProjectionSink sink, boolean showStatus){
		ParallelProjectionDataSinkFeeder [] threads = new ParallelProjectionDataSinkFeeder[projections.getSize()[2]];
		for (int p = 0; p < projections.getSize()[2]; p++){ // for all projections
			Grid2D currentProjection = projections.getSubGrid(p);
			threads[p] =  new ParallelProjectionDataSinkFeeder();
			threads[p].setProjection(currentProjection);
			threads[p].setSink(sink);
			threads[p].setProjectionNumber(MultiSweepTrajectory.getImageIndexInSingleSweepGeometry(p));
		}
		ParallelThreadExecutor exec = new ParallelThreadExecutor(threads);
		try {
			exec.setShowStatus(showStatus);
			exec.execute();
			sink.close();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/