package edu.stanford.rsl.conrad.phantom.renderer;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageGridBuffer;


/**
 * will implement a version which streams projections as they are projected.
 * To be implemented at some point in the future....
 * @author akmaier
 *
 */
public class AnalyticPhantomProjector extends PhantomRenderer {

	AnalyticPhantom phantom;
	protected int stackSize;
	ImageGridBuffer buffer;
	
	protected void init(){
		if (!init) {
			stackSize = Configuration.getGlobalConfiguration().getGeometry().getProjectionStackSize();
			super.init();
		}
	}

	@Override
	public Grid2D getNextProjection() {
		init();
		Grid2D proc  = null;
		while (true) {
			if (buffer.get(projectionNumber) != null) {
				proc = buffer.get(projectionNumber);
				projectionNumber ++;
				buffer.remove(projectionNumber);
				break;
			} else {
				if (projectionNumber >= stackSize){
					break;
				}
				try {
					Thread.sleep(CONRAD.INVERSE_SPEEDUP);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		return proc;
	}

	@Override
	public void createPhantom() {
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public void configure() throws Exception {
		projectionNumber = -1;
		buffer = new ImageGridBuffer();
		super.configured = true;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/