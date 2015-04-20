package edu.stanford.rsl.tutorial.truncation;

import ij.ImageJ;
import edu.stanford.rsl.apps.gui.Citeable;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.Phantom;

/**
 * In this class we investigate the paper
 * Yu, H., Yang, J., Jiang, M., & Wang, G. (2009). Supplemental analysis on compressed sensing based interior tomography. Physics in Medicine and Biology, 54(18), N425–N432. doi:10.1088/0031-9155/54/18/N04.Supplemental
 * 
 * The paper correctly derives that an artifact image whose projection inside the truncated area is zero while having a constant value inside the ROI must be filled with zero.
 * In the class we derive a class of antifact images that is filled with zero and has zero forward projections.  
 * 
 * @author akmaier
 *
 */

public class ConstantValueZeroArtifactImage extends Phantom implements Citeable  {

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

	public ConstantValueZeroArtifactImage (int sizeX, int truncationDiameter, float value){
		super(sizeX, sizeX, "Constant Value Zero Projection Artifact Phantom " + truncationDiameter + " " + value);
		for (int i = 0; i<sizeX; i++){
			double j = Math.sqrt(-Math.pow(i-sizeX/2.0, 2)+Math.pow((truncationDiameter+1)-sizeX/2.0, 2));
			if (Double.isNaN(j)) continue;
			InterpolationOperators.addInterpolateLinear(this.getSubGrid(i), (sizeX/2.0 - j), -value);
			InterpolationOperators.addInterpolateLinear(this.getSubGrid(i), (sizeX/2.0 + j), -value);
			j = Math.sqrt(-Math.pow(i-sizeX/2.0, 2)+Math.pow(truncationDiameter-sizeX/2.0, 2));
			if (Double.isNaN(j)) continue;
			InterpolationOperators.addInterpolateLinear(this.getSubGrid(i), (sizeX/2.0 - j), value);
			InterpolationOperators.addInterpolateLinear(this.getSubGrid(i), (sizeX/2.0 + j), value);
		}
		for (int i = 0; i<sizeX; i++){
			double j = Math.sqrt(-Math.pow(i-sizeX/2.0, 2)+Math.pow((truncationDiameter+1)-sizeX/2.0, 2));
			if (Double.isNaN(j)) continue;
			InterpolationOperators.addInterpolateLinear(this, i, (sizeX/2.0 - j), -value);
			InterpolationOperators.addInterpolateLinear(this, i, (sizeX/2.0 + j), -value);
			j = Math.sqrt(-Math.pow(i-sizeX/2.0, 2)+Math.pow(truncationDiameter-sizeX/2.0, 2));
			if (Double.isNaN(j)) continue;
			InterpolationOperators.addInterpolateLinear(this, i, (sizeX/2.0 - j), value);
			InterpolationOperators.addInterpolateLinear(this, i, (sizeX/2.0 + j), value);
		}
	}
	
	public static void main(String [] args){
		new ImageJ();
		int numS = 400;
		double deltaS = 200.0 / 201.0;
		int numTheta = 360;
		int truncationSize = 100;
		ConstantValueZeroArtifactImage phantom = new ConstantValueZeroArtifactImage(numS, truncationSize, 200);
		phantom.show("Phantom");
		ParallelProjector2D projector2d = new ParallelProjector2D(Math.PI, Math.PI/(numTheta-1), numS, deltaS);
		Grid2D sinogram = projector2d.projectRayDrivenCL(phantom);
		sinogram.show("Sinogram");
	}


}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/