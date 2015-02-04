package edu.stanford.rsl.conrad.phantom.renderer;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.ImagePlus;
import ij.process.ImageProcessor;


/**
 * Phantom of a homogeneous cylinder.
 * 
 * @author akmaier
 *
 */
public class CylinderPhantomRenderer extends StreamingPhantomRenderer {


	double offset = .2;
	
	@Override
	public void createPhantom() {
		Grid3D revan = new Grid3D(dimx, dimy, dimz);
		for (int k = 0; k < dimz; k++){
			Grid2D current = revan.getSubGrid(k);
			for (int i=0; i< dimx; i++) {
				for (int j = 0; j< dimy; j++){
					double value = -1024; // air;
					if ((Math.pow((dimx/2) - i,2) + Math.pow((dimy/2) - j,2) < Math.pow((dimx/2) * (1 - offset),2)) &&
							(Math.abs((dimz/2) - k) < (dimz/2) * (1-offset))){
						value = 0;
					}
					current.putPixelValue(i, j, value);
				}
			}
			buffer.add(current, k);
		}
	}

	@Override
	public String toString() {
		return "Cylinder Phantom";
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
		super.configure();
		offset = UserUtil.queryDouble("Enter margin for air: ", offset);
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/