package edu.stanford.rsl.conrad.phantom.renderer;

import java.io.IOException;

import ij.ImagePlus;
import ij.ImageStack;
import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.apps.gui.GUIConfigurable;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.opencl.OpenCLMaterialPathLengthPhantomRenderer;
import edu.stanford.rsl.conrad.opencl.OpenCLProjectionPhantomRenderer;
import edu.stanford.rsl.conrad.opencl.OpenCLSurfacePhantomRenderer;
import edu.stanford.rsl.conrad.pipeline.IndividualImagePipelineFilteringTool;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;

/**
 * Abstract class to describe a numerical phantom.
 * 
 * @author akmaier
 *
 */
public abstract class PhantomRenderer implements ProjectionSource, Citeable, GUIConfigurable {

	protected boolean configured;
	protected int projectionNumber;
	protected boolean init = false;
	
	protected synchronized void init(){
		//if (!init){
		//	createPhantom();
		//	init = true;
		//}
	}
	
	public void initStream(String filename) throws IOException {
		//
	}
	
	/**
	 * Returns the name of the phantom
	 */
	public abstract String toString();
	
	/**
	 * Method to start the voxelization of the phantom.
	 */
	public abstract void createPhantom();
	

	@Override
	public boolean isConfigured() {
		return configured;
	}
	
	/**
	 * Returns a list of all known numerical phantoms.
	 * @return the list.
	 */
	public static PhantomRenderer [] getPhantoms(){
		PhantomRenderer [] phantoms = {new MetricPhantomRenderer(), new ParallelProjectionPhantomRenderer(), new OpenCLProjectionPhantomRenderer(), new OpenCLMaterialPathLengthPhantomRenderer(), new OpenCLSurfacePhantomRenderer()};
		return phantoms;
	}
	
	/**
	 * Creates an empty volume which can be used to render the phantom into.
	 * 
	 * @param title the title
	 * @param dimx the size in x direction
	 * @param dimy the size in y direction
	 * @param dimz the size in z direction (slice number)
	 * @return the empty volume
	 */
	public static ImagePlus createEmptyVolume(String title, int dimx, int dimy, int dimz){
		ImageStack stack = new ImageStack(dimx, dimy, dimz);
		for(int i = 0; i < dimz; i++){
			stack.setPixels(new float[dimx * dimy], i+1);
		}
		ImagePlus projectionVolume = new ImagePlus();
		projectionVolume.setStack(title, stack);
		return projectionVolume;
	}
	
	@Override
	public int getCurrentProjectionNumber() {
		return projectionNumber;
	}

	@Override
	public void getNextProjection(IndividualImagePipelineFilteringTool tool) {
		if (!init) init();
		tool.setImageProcessor(getNextProjection());
		tool.setImageIndex(projectionNumber);
		projectionNumber++;
	}

	public static Grid3D generateProjections(final PhantomRenderer phantom) throws Exception{
		Thread thread = new Thread(new Runnable(){public void run(){phantom.createPhantom();}});
		thread.start();
		ImagePlusDataSink sink = new ImagePlusDataSink();
		ParallelImageFilterPipeliner pipeliner = new ParallelImageFilterPipeliner(phantom, new ImageFilteringTool[]{}, sink);
		pipeliner.project(true);
		Grid3D result = sink.getResult();
		return result;
	}

	
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/