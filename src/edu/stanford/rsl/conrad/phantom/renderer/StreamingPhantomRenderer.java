package edu.stanford.rsl.conrad.phantom.renderer;

import ij.gui.GenericDialog;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;

public abstract class StreamingPhantomRenderer extends PhantomRenderer {

	protected int dimx = 256;
	protected int dimy = 256;
	protected int dimz = 256;
	protected double originIndexX = 128;
	protected double originIndexY = 128;
	protected double originIndexZ = 128;
	protected ImageGridBuffer buffer;
	
	

	@Override
	public void configure() throws Exception {
		projectionNumber = -1;
		buffer = new ImageGridBuffer();
		readDimensionsFromGlobalConfig();
		configured = true;
	}

	/**
	 * Reads the required dimensions from the global configuration.
	 */
	protected void readDimensionsFromGlobalConfig(){
		Configuration config = Configuration.getGlobalConfiguration();
		dimx = config.getGeometry().getReconDimensionX();
		dimy = (int) config.getGeometry().getReconDimensionY();
		dimz = (int) config.getGeometry().getReconDimensionZ();
		originIndexX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		originIndexY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		originIndexZ = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
	}
	
	
	/**
	 * Creates a GenericDialog with fields for the phantom dimensions.
	 * @return the dialog.
	 */
	protected GenericDialog createDimensionDialog (){
		Configuration config = Configuration.getGlobalConfiguration();
		GenericDialog gd = new GenericDialog("Phantom Configuration");
		gd.addNumericField("Phantom X Dimension:", config.getGeometry().getReconDimensionX(), 5);
		gd.addNumericField("Phantom Y Dimension: ", config.getGeometry().getReconDimensionY(), 5);
		gd.addNumericField("Phantom Z Dimension: ", config.getGeometry().getReconDimensionZ(), 5);
		originIndexX = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
		originIndexY = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
		originIndexZ = (int) Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
		gd.addNumericField("Index of Origin X:", originIndexX, 5);
		gd.addNumericField("Index of Origin Y: ", originIndexY, 5);
		gd.addNumericField("Index of Origin Z: ", originIndexZ, 5);
		return gd;
	}
	
	
	/**
	 * Reads the dimensions from the GenericDialog
	 * @param gd the GenericDialog
	 * 
	 * @see #createDimensionDialog()
	 */
	protected void readDimensions(GenericDialog gd){
		// read dialog window
		dimx = (int) gd.getNextNumber();
		dimy = (int) gd.getNextNumber();
		dimz = (int) gd.getNextNumber();
		originIndexX = (int) gd.getNextNumber();
		originIndexY = (int) gd.getNextNumber();
		originIndexZ = (int) gd.getNextNumber();
		Trajectory traj = Configuration.getGlobalConfiguration().getGeometry();
		traj.setOriginInPixelsX(originIndexX);
		traj.setOriginInPixelsY(originIndexY);
		traj.setOriginInPixelsZ(originIndexZ);
	}

	
	
	@Override
	public Grid2D getNextProjection() {
		init();
		Grid2D proc  = null;
		if (projectionNumber < dimz -1) {
			projectionNumber ++;
			while (proc == null){
				proc = buffer.get(projectionNumber);
			}
			buffer.remove(projectionNumber);

		}
		return proc;
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/