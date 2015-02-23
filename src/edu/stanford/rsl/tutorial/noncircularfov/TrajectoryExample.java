/*
 * Copyright (C) 2014 Madgalena Herbst
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.tutorial.noncircularfov;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.fan.CosineFilter;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.FanBeamProjector2D;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.phantoms.Phantom;


/**
 * Short example of how to use the Correspondences class.
 * 
 * @author Magdalena
 *
 */
public class TrajectoryExample {
	
	public static void main(String[] args) {
		
		// required parameters
		int imgSzXMM = 256, // [mm]
		imgSzYMM = imgSzXMM; // [mm]
		float pxSzXMM = 1.0f, // [mm]
		pxSzYMM = pxSzXMM; // [mm]
		int imgSzXGU = (int) Math.floor(imgSzXMM / pxSzXMM), // [GU]
		imgSzYGU = (int) Math.floor(imgSzYMM / pxSzYMM); // [GU]
		
		double gammaM =2* 11.768288932020647 * Math.PI / 180, maxT = 501, deltaT = 1.0, focalLength = (maxT / 2.0 - 0.5)
				* deltaT / Math.tan(gammaM), maxBeta = 360 * Math.PI / 180, // +gammaM*2,
		deltaBeta = maxBeta / 360;
		
		// create Phantom
		System.out.println("Creating phantom...");
		Phantom phantom = new TwoCirclesGrid(imgSzXMM,imgSzYMM);
		phantom.setSpacing(pxSzXMM, pxSzYMM);
		// origin is given in (negative) world coordinates
		phantom.setOrigin(-(imgSzXGU * phantom.getSpacing()[0]) / 2,
				-(imgSzYGU * phantom.getSpacing()[1]) / 2);
		
		new ImageJ();
		
		phantom.show("Phantom");
		
		// make projections
		System.out.println("projecting...");
		FanBeamProjector2D fanBeamProjector = new FanBeamProjector2D(focalLength, maxBeta, deltaBeta, maxT, deltaT);
		
		Grid2D projectionP = new Grid2D(phantom);

		Grid2D sino = fanBeamProjector.projectRayDriven(projectionP);
		sino.clone().show("Sinogram");
		
		// computing the minimal set
		System.out.println("Computing the minimal set...");
		
		double detSize = 130;	// [mm], pixelsize deltaT
		int detSizeGU = (int) (detSize/deltaT);
		
		Correspondences c = new Correspondences(focalLength, sino);
		
		int[] minimalSet = c.evaluateNumer(sino, detSizeGU);
		
		System.out.println("Minimal required rotation: "  + minimalSet[0]*deltaBeta  + "degress, corresp. start angle: " + minimalSet[1]*deltaBeta);
		
		// displaying the minimal set
		System.out.println("Display minimal set");
		
		int width = sino.getWidth();
		int height = sino.getHeight();
		
		Grid2D mask = new Grid2D(width, height);
		Grid2D data = new Grid2D(width, height);
		
		int start = minimalSet[1];
		int rot = minimalSet[0];
		
		for (int y = start; y < start + rot; y++) {
			for (int x = 0; x < width; x++) {
				//if (sino.getAtIndex(x, y%height) > 0) {
				if((x>150) && (x < 150+detSize)){
					for (int i = 0; i < detSize; i++) {
						if (x + i < width) {
							mask.setAtIndex(x + i, y%height, 1.f);
							data.setAtIndex(x + i, y%height, sino.getAtIndex(x + i, y%height));
						}
					}
					break;
				}
			}
		}
		
		Grid2D res = c.findCorrespondences(mask, sino);
		res.setSpacing(deltaT, deltaBeta);	// required for reconstruction
		Grid2D dou = c.findRedundancies(mask, sino);
		Grid2D diff = new Grid2D(width, height);
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				
				// count missing data
				if(sino.getAtIndex(x, y) > 0 && res.getAtIndex(x, y) == 0) {
					diff.setAtIndex(x, y, 1.f);
				}
			}
		}
		
		data.show("Acquired Data");
		res.show("Reconstructed Sinogram");
		dou.show("redundant areas");
		diff.show("Missing Pixels");
		mask.show("Mask");
		
		// resonstruct the completed sinogram
		System.out.println("Reconstruct the completed sinogram...");
		
		RamLakKernel ramLak = new RamLakKernel((int) (maxT / deltaT), deltaT);
		CosineFilter cKern = new CosineFilter(focalLength, maxT, deltaT);
		// Apply filtering
		for (int theta = 0; theta < res.getSize()[1]; ++theta) {
			cKern.applyToGrid(res.getSubGrid(theta));

		}

		for (int theta = 0; theta < res.getSize()[1]; ++theta) {
			ramLak.applyToGrid(res.getSubGrid(theta));
		}
		
		res.show("After Filtering");
		
		// Do the backprojection
		FanBeamBackprojector2D fbp = new FanBeamBackprojector2D(focalLength,
				deltaT, deltaBeta, imgSzXMM, imgSzYMM);

		Grid2D reco = fbp.backprojectPixelDriven(res);
		reco.show("Reconstructed result");
		
		//finished
		System.out.println("Everything done!");
	}

}
