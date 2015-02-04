package edu.stanford.rsl.conrad.pipeline;


import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.multiprojection.MultiProjectionFilter;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FFTUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;


/**
 * Class for running multiple ImageFilteringTools in parallel. Creates a thread for each ImageFilteringTool
 * and executes them using a ParallelThreadExecutor. It will start one thread on each processor of the
 * machine at the same time.
 * 
 * @author Andreas Maier
 *
 */
public class ParallelImageFilterPipeliner {

	private ProjectionSource source;
	private ImageFilteringTool [] tools;
	private BufferedProjectionSink sink;
	private boolean debug = false;

	/**
	 * Constructor requires an ImagePlus. If it has multiple slices the processing is performed in parallel.
	 * 
	 * @param image the ImagePlus to be filtered
	 * @param tools the ImageFilteringTool to be applied.
	 * @param sink the image sink
	 */
	public ParallelImageFilterPipeliner(ProjectionSource image, ImageFilteringTool [] tools, BufferedProjectionSink sink){
		this.tools = tools;
		this.source = image;
		this.sink = sink;
	}

	private boolean isMultiProjectionFilter(ImageFilteringTool filter){
		return (filter instanceof MultiProjectionFilter);
	}

	private int getIndividualProjectionFilterBlockLength(ImageFilteringTool [] tool, int start){
		int revan = -1;
		for (int i = start; i >= 0; i--){
			if (tool[i] instanceof IndividualImageFilteringTool) {
				revan = i;
			} else {
				break;
			}
		}
		return revan;
	}

	/**
	 * This method starts the actual filtering.
	 * 
	 * @param showStatus displays the current status if true
	 * @throws Exception may occur.
	 */
	public synchronized void project(boolean showStatus) throws Exception{
		FFTUtil.init1DFFT(Configuration.getGlobalConfiguration().getGeometry().getDetectorWidth());	
		int currentTool = tools.length - 1;
		ProjectionSource source = this.source;
		ProjectionSink currentSink = this.sink;
		currentSink.setShowStatus(showStatus);
		int availableCPUs = CONRAD.getNumberOfThreads();
		while (currentTool >= 0) {
			if (isMultiProjectionFilter(tools[currentTool])){
				// Add a the MultiProjectionFilter to the pipeline.
				MultiProjectionFilter filter = (MultiProjectionFilter) tools[currentTool];
				filter.setSink(currentSink);
				currentSink = filter;
				filter.start();
				// next
				currentTool--;
			} else {
				// Compute end of this parallel block.
				int blockStart  = getIndividualProjectionFilterBlockLength(tools, currentTool);
				if (blockStart >= 0) {
					ParallelImageFilterSink parallel = new ParallelImageFilterSink();
					parallel.setShowStatus(false);
					parallel.setSink(currentSink);
					parallel.setPipeline(ParallelImageFilterPipeliner.getSubPipeline(tools, blockStart, currentTool+1));
					parallel.setDebug(debug);
					parallel.start(availableCPUs);
					currentTool = blockStart-1;
					currentSink = parallel;
				} else {
					throw new Exception("Block too long");
				}
			}
			currentSink.setShowStatus(showStatus);
		}
		Grid2D img = source.getNextProjection();
		if (img == null) throw new Exception ("ImageJ not ready");
		int projectionNumber = source.getCurrentProjectionNumber();
		while (img != null){
			if (debug) {
				System.out.println("Streaming into pipeline projection: " + projectionNumber + " "  + currentSink.getName());
			}
			currentSink.process(img, projectionNumber);
			img = source.getNextProjection();
			projectionNumber = source.getCurrentProjectionNumber();
			// Here we should consider how much memory we have left
			Thread.sleep(CONRAD.INVERSE_SPEEDUP);
			double free = CONRAD.getFreeMemoryAsDouble();
			if (free < 0.25){
				int slowdown = Configuration.getGlobalConfiguration().queryIntFromRegistry(RegKeys.SLOW_DOWN_MS);
				Thread.sleep(slowdown);
				System.err.println("ParallelImageFilterPipeliner: Memory almost full, slowing down processing speed by " + slowdown + ". Buy more memory to increase processing speed! Projection " + source.getCurrentProjectionNumber() + " free: " + free );			
			}
		}
		if (debug) System.out.println("ParallelImageFilterPipeliner: Projections Streamed.");
		currentSink.close();
		this.sink.getResult();
		if (debug) System.out.println("ParallelImageFilterPipeliner: All Processors done.");
	}

	public static IndividualImageFilteringTool [] getSubPipeline(ImageFilteringTool [] tools, int start, int end){
		IndividualImageFilteringTool [] pipelineClone = new IndividualImageFilteringTool[end - start];
		for (int i = start; i < end; i++){
			pipelineClone[i-start] = ((IndividualImageFilteringTool)tools[i]);
		}
		return pipelineClone;
	}

	public static ImageFilteringTool [] getPipelineClone(ImageFilteringTool [] tools){
		ImageFilteringTool [] pipelineClone = new ImageFilteringTool[tools.length];
		for (int i = 0; i < tools.length; i++){
			pipelineClone[i] = tools[i].clone();
		}
		return pipelineClone;
	}

	public void project() throws Exception{
		project(true);
	}
	


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/