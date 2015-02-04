package edu.stanford.rsl.conrad.reconstruction;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.geometry.Projection;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import ij.process.FloatProcessor;


/**
 * The most simple implementation of the backprojection using no speed-up methods. The name is derived from the example given at <a href="http://www.rabbitct.com/">RabbitCT</a>.
 * Does not include the computation of a volume of interest.
 * @author akmaier
 *
 */
public class LolaBunnyBackprojector extends VOIBasedReconstructionFilter {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4887607829631536252L;
	
	public LolaBunnyBackprojector(){
		super();
		useVOImap = false;
	}

	@Override
	public void backproject(Grid2D projection, int projectionNumber) {
		boolean debug = false;
		int count = 0;
		initialize(projection);
		FloatProcessor currentProjection = new FloatProcessor(projection.getWidth(), projection.getHeight(), projection.getBuffer(), null);
		
		// Constant part of distance weighting (D^2) + additional weighting for arbitrary scan ranges
		double D =  getGeometry().getSourceToDetectorDistance();
		currentProjection.multiply(10* D*D * 2* Math.PI * getGeometry().getPixelDimensionX() / getGeometry().getNumProjectionMatrices());
		
		int p = projectionNumber;
		double[] voxel = new double [4];
		voxel[3] = 1;
		for (int k = 0; k < maxK ; k++){ // for all slices
			voxel[2] = (this.getGeometry().getVoxelSpacingZ() * k) - offsetZ;
			for (int i=0; i < maxI; i++){ // for all lines
				voxel[0] = (this.getGeometry().getVoxelSpacingX() * i) - offsetX;
				for (int j = 0; j < maxJ; j++){ // for all voxels
					// compute real world coordinates in homogenious coordinates;
					//if (voiMap[i][j][k]){
						voxel[1] = (this.getGeometry().getVoxelSpacingY() * j) - offsetY;
						SimpleVector point3d = new SimpleVector(voxel);
						//Compute coordinates in projection data.
						Projection Pcurr = getGeometry().getProjectionMatrix(p);
						SimpleVector point2d = SimpleOperators.multiply(Pcurr.computeP(), point3d);
						double coordX = point2d.getElement(0) / point2d.getElement(2);
						double coordY = point2d.getElement(1) / point2d.getElement(2);
						
						// back project						
						double distanceWeight = point2d.getElement(2);
						double increment = currentProjection.getInterpolatedValue(coordX + lineOffset, coordY)/(distanceWeight*distanceWeight);
						if (Double.isNaN(increment)){
							debug = true;
							if (count < 10) System.out.println("NAN Happened at i = " + i + " j = " + j + " k = " + k + " projection = " + projectionNumber + " x = " + coordX + " y = " + coordY  );
							increment = 1;
							count ++;
						}
						updateVolume(i, j, k, increment);
					//}
				}
			}
		}
		if (debug) {
			throw new RuntimeException("Encountered NaN in projection!");
		}
	}
	

	@Override
	public String getMedlineCitation(){
		return "Rohkohl C, Keck B, Hofmann H, Hornegger J. RabbitCT - an open platform for benchmarking 3D cone-beam reconstruction algorithms. Med Phys 2009 36(9):3940-4.";
	}
	
	@Override
	public String getToolName(){
		return "Lola Bunny CPU-based Backprojector";
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/