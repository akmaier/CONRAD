package edu.stanford.rsl.tutorial.truncation;

import ij.ImageJ;
import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;

/**
 * Test implementation to show that a constant offset inside the ROI in the artifact image is not possible.
 * Enforcing the 0 attenuation inside the FOV and a constant value in the artifact image converges to 0.
 * @author akmaier
 *
 */
public class NonZeroArtifactImage extends Phantom implements Citeable {

	public String getBibtexCitation() {
		return "@article{Yu2009, \n"+
		"  author = {Yu, Hengyong and Yang, Jiansheng and Jiang, Ming and Wang, Ge},\n"+
		"  doi = {10.1088/0031-9155/54/18/N04.Supplemental},\n"+
		"  file = {:C$\backslash$:/Users/z002xsyz/Desktop/supplemental.pdf:pdf},\n"+
		"  institution = {CT Laboratory, Biomedical Imaging Division, VT-WFU School of Biomedical Engineering, Virginia Tech, Blacksburg, VA 24061, USA. hengyong-yu@ieee.org},\n"+
		"  journal = {Physics in Medicine and Biology},\n"+
		"  keywords = {artifact image,compressed sensing,computed tomography,ct,interior tomography,minimization,total variation},\n"+
		"  number = {18},\n"+
		"  pages = {N425--N432},\n"+
		"  publisher = {IOP Publishing},\n"+
		"  title = {{Supplemental analysis on compressed sensing based interior tomography.}},\n"+
		"  volume = {54},\n"+
		"  year = {2009}\n"+
		"  }";
	}

	public String getMedlineCitation() {
		return "Yu, H., Yang, J., Jiang, M., & Wang, G. (2009). Supplemental analysis on compressed sensing based interior tomography. Physics in Medicine and Biology, 54(18), N425–N432. doi:10.1088/0031-9155/54/18/N04.Supplemental";
	}

	
	public NonZeroArtifactImage(int x, int y) {
		super(x, y, "NonZeroArtifactImage");
		boolean showProgress = false;

		// Create circle in the grid.
		double radius = 0.40 * x;
		double radius2 = 0.50 * x;
		int xcenter = x / 2;
		int ycenter = y / 2;
		float val = 1.f;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
					super.setAtIndex(i, j, val);
				} else {
					if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius2 * radius2)){
						super.setAtIndex(i, j, -4.f);
					}
				}
			}
		}
		int iterations = 1001;
		int truncationSize = x/4;
		radius = truncationSize;
		int numS = x;
		double deltaS = x / (double)(x+1);
		int numTheta = 360;

		ParallelProjector2D projector2d = new ParallelProjector2D(Math.PI, Math.PI/(numTheta-1), numS, deltaS);
		Grid2D sinogram = projector2d.projectRayDrivenCL(this);
		if (showProgress) sinogram.show("Cylinder Sinogram");
		Grid2D truncatedSinogram = new Grid2D(sinogram);

		for (int iter = 1; iter < iterations; iter++){
			if (showProgress) if (iter%50 == 0) truncatedSinogram.show("Artifact Sinogram " + iter);
			truncatedSinogram = new Grid2D(truncatedSinogram);
			for (int i=0; i < numTheta; i++) {
				for (int j = truncationSize; j < numS - truncationSize; j++){
					truncatedSinogram.setAtIndex(i, j, 0);
				}
			}
			RamLakKernel ramLakFilter = new RamLakKernel((int) (numS / deltaS), deltaS);
			for (int theta = 0; theta < sinogram.getSize()[0]; ++theta) {
				ramLakFilter.applyToGrid(truncatedSinogram.getSubGrid(theta));
			}
			ParallelBackprojector2D backprojector2d = new ParallelBackprojector2D(numS, numS, 1, 1);
			Grid2D truncatedImage = backprojector2d.backprojectPixelDriven(truncatedSinogram);
			if (showProgress) if (iter%50 == 0) truncatedImage.show("Artifact Image" + iter);
			truncatedImage = new Grid2D(truncatedImage);
			double sum = 0;
			long number = 0;
			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {
					if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
						sum += truncatedImage.getAtIndex(i, j);
						number++;
					}
					if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) > (radius2 * radius2)){
						truncatedImage.setAtIndex(i, j, 0);
					}
				}
			}
			double mean = sum / number;
			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {
					if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
						truncatedImage.setAtIndex(i, j, (float)mean);
					} 
				}
			}
			truncatedSinogram = projector2d.projectRayDrivenCL(truncatedImage);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		new ImageJ();
		Grid2D phantom = new NonZeroArtifactImage(200, 200);
		phantom.show("Phantom");
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/