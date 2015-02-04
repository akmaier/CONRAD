package edu.stanford.rsl.conrad.reconstruction.iterative;


import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.reconstruction.ReconstructionFilter;
import edu.stanford.rsl.conrad.utils.Configuration;


/**
 * 
 * All iterative reconstruction algorithms are based on the iterative reconstruction filter
 * The iterative reconstruction is the abstract class that describe a general outline for a
 * iterative reconstruction algorithm. It splite the geometry computation into ray-driven, 
 * distance-driven and separabel footprints. The iteration methods also may be splitted into
 * ART, SART and some other model-based reconstruction.
 * @author Meng Wu
 *
 */

public abstract class IterativeReconstruction extends ReconstructionFilter {


	private static final long serialVersionUID = 1L;

	
	public abstract void iterativeReconstruct() throws Exception;
	
	@Override
	protected void reconstruct() throws Exception{
			try {
				iterativeReconstruct();
			} catch (Exception e){
				System.out.println("An error occured during iterations ");
			}
			
			if (Configuration.getGlobalConfiguration().getUseHounsfieldScaling()) 
				applyHounsfieldScaling();
			
			int [] size = projectionVolume.getSize();
			
			for (int k = 0; k < size[2]; k++){
				sink.process(projectionVolume.getSubGrid(k), k);
			}
			sink.close();
			init = false;
	}

	@Override
	public void configure() throws Exception {
		// TODO Auto-generated method stub
	}

	@Override
	public ImageFilteringTool clone() {
		return this;
	}

	@Override
	public boolean isDeviceDependent() {
		return false;
	}

}
/*
 * Copyright (C) 2010-2014 Meng Wu
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/