package edu.stanford.rsl.conrad.pipeline;

import ij.IJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.IndividualImageFilteringTool;
import edu.stanford.rsl.conrad.geometry.trajectories.MultiSweepTrajectory;
import edu.stanford.rsl.conrad.utils.CONRAD;



/**
 * Class to describe the thread which runs in a parallel image pipeline. In the end data is written into the ProjectionDataSink
 * 
 * @author Andreas Maier
 *
 */
public class IndividualImagePipelineFilteringTool extends IndividualImageFilteringTool {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1573824644614146404L;
	private boolean debug = false;
	private IndividualImageFilteringTool [] tools;
	private ProjectionSink sink;
	private ProjectionSource projectionSource;
	private Grid2D [] stack;

	/**
	 * returns the name of the actual tool which was used.
	 * @return the tool name
	 */
	public String getToolName(){
		return "Image Filtering Pipeline";
	}

	public void setPipeline(IndividualImageFilteringTool [] tools) {
		this.tools = tools;
	}

	public ImageFilteringTool [] getPipeline() {
		return tools;
	}

	public void setSink(ProjectionSink sink) {
		this.sink = sink;
	}

	public ProjectionSink getSink() {
		return sink;
	}

	public void setProjectionSource(ProjectionSource projectionSource) {
		this.projectionSource = projectionSource;
	}

	public ProjectionSource getProjectionSource() {
		return projectionSource;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		Grid2D temp = imageProcessor;
		if (debug) stack = new Grid2D[tools.length];
		if (debug) System.out.println("Tools " + tools.length);
		int correctedIndex = imageIndex;
		try {
			MultiSweepTrajectory.getImageIndexInSingleSweepGeometry(imageIndex);
		} catch (Exception e){
			CONRAD.log("Warning: " +e.toString());
			//e.printStackTrace();
		}
		for (int i = 0; i < tools.length; i++){
			Grid2D in = temp;
			if (debug) System.out.println("Tool " + i);
			if (debug) System.out.println(tools[i].getToolName() + " " + imageIndex + " " + correctedIndex + " (tool " + i+ ")");
			tools[i].setImageIndex(correctedIndex);
			temp = null;
			temp = tools[i].applyToolToImage(in);
			in = null;
			
			if (debug) stack[i] = temp;
			if (debug) if (this.imageIndex < 15) {
				//if (i == tools.length - 1) VisualizationUtil.showImageProcessor(temp, tools[i].getToolName() + " Projection " + imageIndex);
			}
		}
		if (imageIndex % 20 == 0) CONRAD.gc();
		if (debug) System.out.println("Tools done: " + tools.length);
		return temp;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImagePipelineFilteringTool clone = new IndividualImagePipelineFilteringTool();
		clone.setPipeline(tools);
		return clone;
	}

	/**
	 * This method performs the filtering and the projection on one frame, i.e., one projection image. 
	 */
	@Override
	public void run() {
		projectionSource.getNextProjection(this);
		while ((this.imageProcessor != null)) {
			if (debug) System.out.println("Image Index: " + this.imageIndex);
			try {
				theFiltered = applyToolToImage(imageProcessor);
				sink.process(theFiltered, imageIndex);
				if (debug) stack = null;
			} catch (Exception e){
				CONRAD.log(e.getLocalizedMessage());
				//System.out.println(e.getLocalizedMessage());
				e.printStackTrace();
				for (int i = 0; i < tools.length; i++){
					String title = tools[i].getToolName() + " Projection " + imageIndex;
				}
			}
			if (debug) System.out.println("done: "+this.imageIndex);
			projectionSource.getNextProjection(this);
			if (debug) System.out.println("next: "+imageProcessor);
		}
		latch.countDown();
		imageProcessor = null;
		theFiltered = null;
		if (debug) System.out.println("done.");

	}

	@Override
	public void configure() throws Exception {
		configured = tools != null;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/