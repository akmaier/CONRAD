package edu.stanford.rsl.tutorial.ringArtifactCorrection;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import ij.ImageJ;
import ij.io.OpenDialog;

public class RingArtifactCorrectionExample {

	public static void main(String[] args) {
		new ImageJ();
		OpenDialog od = new OpenDialog("Please select the reconstruction file.");
		RingArtifactCorrector rac = new RingArtifactCorrector();
		Grid3D reko = rac.loadReconstruction(od.getDirectory()+od.getFileName());
		Grid2D rekoS = reko.getSubGrid(0); 
		rekoS.show("Original Image");
		
		//option examples
		//rac.setShowSteps(true); //displays all interim steps of the correction process
		//rac.setRadialFilterWidth(50); //sets the filter width in radial direction
		//setAzimuthalFilterWidth (50); //sets the filter width in azimuthal direction
		
		Grid2D correctedReko = rac.getRingArtifactCorrectedImage(rekoS);
		correctedReko.show("Ring artifact corrected");
	}

}

/*
 * Copyright (C) 2010-2015 Florian Gabsteiger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/