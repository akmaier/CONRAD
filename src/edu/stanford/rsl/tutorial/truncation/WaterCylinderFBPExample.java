package edu.stanford.rsl.tutorial.truncation;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.tutorial.RamLakKernel;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.UniformCircleGrid2D;


/**
 * Example Class to show the linearity of the truncation artifacts.
 * If we split the projection image into the truncated and the non-truncated part,
 * we can see that the filtered backprojection of both combined in image space
 * gives a correct reconstruction. Furthermore, the forward projection of the artifact
 * image gives nearly 0 in the area where the truncation took place. Note that this
 * is only perfect in continuous domain. With discrete sampling a slight residual artifact
 * remains.
 * 
 * @author akmaier
 *
 */
public class WaterCylinderFBPExample {


	public static void main (String [] args){
		new ImageJ();
		int numS = 200;
		double deltaS = 200.0 / 201.0;
		int numTheta = 360;
		int truncationSize = 50;
		UniformCircleGrid2D cylinder = new UniformCircleGrid2D(numS, numS);
		ParallelProjector2D projector2d = new ParallelProjector2D(Math.PI, Math.PI/(numTheta-1), numS, deltaS);
		Grid2D sinogram = projector2d.projectRayDrivenCL(cylinder);
		sinogram.show("Cylinder Sinogram");
		Grid2D truncatedSinogram = new Grid2D(sinogram);
		for (int i=0; i < numTheta; i++) {
			for (int j = 0; j < truncationSize; j++){
				truncatedSinogram.setAtIndex(i, j, 0);
			}
			for (int j = numS - truncationSize+1; j < numS; j++){
				truncatedSinogram.setAtIndex(i, j, 0);
			}
		}
		truncatedSinogram.show("Truncated Image");
		Grid2D artifactSinogram = new Grid2D(sinogram);
		NumericPointwiseOperators.multiplyBy(sinogram, -1);
		artifactSinogram = (Grid2D)NumericPointwiseOperators.addedBy(truncatedSinogram, sinogram);
		NumericPointwiseOperators.multiplyBy(sinogram, -1);
		NumericPointwiseOperators.multiplyBy(artifactSinogram, -1);
		artifactSinogram.show("Artifact Image");
		RamLakKernel ramLakFilter = new RamLakKernel((int) (numS / deltaS), deltaS);
		for (int theta = 0; theta < sinogram.getSize()[0]; ++theta) {
			ramLakFilter.applyToGrid(artifactSinogram.getSubGrid(theta));
			ramLakFilter.applyToGrid(truncatedSinogram.getSubGrid(theta));
		}
		ParallelBackprojector2D backprojector2d = new ParallelBackprojector2D(numS, numS, 1, 1);
		Grid2D truncatedImage = backprojector2d.backprojectPixelDriven(truncatedSinogram);
		truncatedImage.show("Truncated Image");
		Grid2D artifactImage = backprojector2d.backprojectPixelDriven(artifactSinogram);
		artifactImage.show("Artifact Image");
		cylinder = (UniformCircleGrid2D)NumericPointwiseOperators.addedBy(artifactImage, truncatedImage);
		cylinder.show("Combination of both");
		Grid2D FPofArtifactImage = projector2d.projectRayDrivenCL(artifactImage);
		FPofArtifactImage.show("FP of Artifact Image");
		Grid2D FPofTruncImage = projector2d.projectRayDrivenCL(truncatedImage);
		FPofTruncImage.show("FP of Truncation Image");
		Grid2D combinedSinogram = new Grid2D(sinogram);
		combinedSinogram = (Grid2D)NumericPointwiseOperators.addedBy(FPofArtifactImage, FPofTruncImage);
		combinedSinogram.show("Combination of FP");
	}
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/