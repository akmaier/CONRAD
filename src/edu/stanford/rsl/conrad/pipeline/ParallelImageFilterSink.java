package edu.stanford.rsl.conrad.pipeline;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.parallel.ParallelThreadExecutor;
import edu.stanford.rsl.conrad.parallel.ParallelizableRunnable;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;


public class ParallelImageFilterSink implements Runnable, ProjectionSink, ProjectionSource {

	private ProjectionSink sink;
	private IndividualImageFilteringTool [] pipeline;
	private ImageGridBuffer buffer;
	private int currentIndex = 0;
	private boolean debug = false;
	private int lastIndex = -1;
	private boolean closed = false;
	private boolean showStatus = false;
	private boolean init = false;
	private int cpus = 1;

	public void setShowStatus(boolean status){
		this.showStatus = status;
	}

	public void setDebug(boolean debug){
		this.debug = debug;
	}

	public void setPipeline(IndividualImageFilteringTool [] pipeline){
		this.pipeline = pipeline;
	}

	public void setSink(ProjectionSink sink){
		this.sink = sink;
	}

	@Override
	public void close() throws Exception {
		//System.out.println("Closing Parallel sink " + closed + " " +lastIndex +  " " + buffer.size());
		//if (debug) throw new RuntimeException("Closing parallel tool");
		if (!closed) {
			lastIndex = buffer.size();
			closed = true;
		}
	}

	private void configureTool(IndividualImagePipelineFilteringTool tool, ProjectionSource source, ProjectionSink sink){
		tool.setPipeline(getPipelineClone(pipeline));
		tool.setSink(sink);
		tool.setProjectionSource(source);
	}

	private static IndividualImageFilteringTool [] getPipelineClone(IndividualImageFilteringTool [] tools){
		IndividualImageFilteringTool [] pipelineClone = new IndividualImageFilteringTool[tools.length];
		for (int i = 0; i < tools.length; i++){
			pipelineClone[i] = ((IndividualImageFilteringTool)tools[i]).clone();
		}
		return pipelineClone;
	}

	public void start(int cpus){
		this.cpus = cpus;
		Thread thread = new Thread(this);
		thread.start();
	}

	private void init(){
		if (debug) System.out.println("ParallelImageFilterSink: init " + init);
		if (!init){
			buffer = new ImageGridBuffer();
			init = true;
		}
	}

	@Override
	public String getName() {
		return "Parallel Image Filter Sink";
	}

	@Override
	public void process(Grid2D projection, int projectionNumber)
	throws Exception {
		if (debug) System.out.println("ParallelImageFilterSink: project " + projectionNumber);
		init();
		buffer.add(projection, projectionNumber);
	}

	@Override
	public int getCurrentProjectionNumber() {
		return currentIndex;
	}

	@Override
	public synchronized void getNextProjection(IndividualImagePipelineFilteringTool tool) {
		Grid2D next = getNextProjection();
		if (next != null) {
			tool.setImageProcessor((next));
			tool.setImageIndex(currentIndex-1);
		} else {
			tool.setImageProcessor(null);
			tool.setImageIndex(currentIndex-1);
		}
	}

	@Override
	public synchronized Grid2D getNextProjection() {
		Grid2D revan = null;
		if (debug) {
			System.out.println("ParallelImageFilterSink: Projection requested: " + currentIndex);
		}
		while (revan == null) {
			if (debug) System.out.println("ParallelImageFilterSink: buffer = " + buffer);
			if (buffer != null) {
				revan = buffer.get(currentIndex);
				if (closed){
					if (debug) {
						System.out.println("ParallelImageFilterSink: All data arrived. currentIndex: " + currentIndex + " of " + lastIndex+ " buffer: " + buffer.size());
					}
					// we are trying to read after the last index.
					if (currentIndex >= lastIndex) {
						try {
							if (debug) {
								System.out.println("ParallelImageFilterSink: End of Stream reached.");
							}
							break;
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
				}
			}
			try {
				Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		if (closed) {
			if (currentIndex < lastIndex) {
				buffer.remove(currentIndex);
			}
		} else {
			buffer.remove(currentIndex);
		}
		currentIndex ++;
		return revan;
	}

	@Override
	public void run() {

		if (debug) {
			// In debug mode we process only on a single processor.
			// Java sometimes eats an exception in parallel processing mode.
			// Debug mode will be n times slower than normal mode, where n is the
			// number of processors.
			System.out.println("Running in Debug Mode. Only one Processor.");
			IndividualImagePipelineFilteringTool tool = new IndividualImagePipelineFilteringTool();
			configureTool(tool, this, sink);
			tool.setLatch(new CountDownLatch(1));
			tool.run();
			try {
				sink.close();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}else {
			// Real parallel processing mode.
			IndividualImagePipelineFilteringTool [] threads = new IndividualImagePipelineFilteringTool[cpus];
			for (int i = 0; i < cpus; i++){
				// configure the threads.
				threads[i] = new IndividualImagePipelineFilteringTool();
				configureTool(threads[i], this, sink);
			}
			// Start the processing.
			ParallelThreadExecutor exec = new ParallelThreadExecutor((ParallelizableRunnable[])threads, cpus);
			exec.setShowStatus(false);
			try {
				exec.execute();
				sink.close();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		if(debug) System.out.println("ParallelImageFilterSink: done.");
	}

	@Override
	public void initStream(String filename) throws IOException {
		// nothing to do
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/