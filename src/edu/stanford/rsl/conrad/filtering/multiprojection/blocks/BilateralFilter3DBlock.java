package edu.stanford.rsl.conrad.filtering.multiprojection.blocks;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.UserUtil;


/**
 * Class implements the processing for a straight forward 3D bilateral filter.
 * Implementation does not use any kind of speed ups. Use only with small values for width!
 * Otherwise the filter may take extremely long to finish the computation.
 * 
 * @author akmaier
 *
 */
public class BilateralFilter3DBlock extends ImageProcessingBlock {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7875829253535505054L;
	private double sigma_d = 2.0;
	private double sigma_r = 0.001;
	private int width = 5;

	@Override
	public ImageProcessingBlock clone() {
		BilateralFilter3DBlock clone = new BilateralFilter3DBlock();
		clone.sigma_d = sigma_d;
		clone.sigma_r = sigma_r;
		clone.width = width;
		return clone;
	}

	@Override
	protected void processImageBlock() {
		int[] size = inputBlock.getSize();
		outputBlock = new Grid3D(size[0], size[1], size[2], true);
		for (int z=0; z < size[2]; z++) {
			for (int y=0; y < size[1]; y++) {
				for (int x=0; x < size[0]; x++) {
					float val = computeFilteredPixel(inputBlock, x, y, z);
					outputBlock.setAtIndex(x, y, z, val);
				}
			}
		}
	}

	private double computeGeometricCloseness(int i, int j, int k, int x, int y, int z){
		return Math.exp(- 0.5 * Math.pow((computeEuclidianDistance(i,j,k,x,y,z) / sigma_d), 2));
	}

	private double computeEuclidianDistance(int i, int j, int k, int x, int y, int z){
		return Math.sqrt(Math.pow(i-x ,2) + Math.pow((j-y),2) + Math.pow((k-z),2));
	}


	private double computePhotometricDistance(float cur, float ref) {
		return Math.exp(-0.5f * Math.pow(Math.abs(cur - ref) / sigma_r, 2));
	}

	private float computeFilteredPixel(Grid3D image, int x, int y, int z){
		int[] size = image.getSize();
		
		double sumWeight = 0;
		double sumFilter = 0;
		// No filtering at the image boudaries;
		if ((x < (width/2)) || (x+(width/2)+1 >= size[0])
				|| (y < (width/2)) || (y+(width/2)+1 >= size[1])
				|| (z < (width/2)) || (z+(width/2)+1 >= size[2])
		){
			sumWeight = 1;
			sumFilter = image.getAtIndex(x, y, z);
		} else {
			float ref = image.getSubGrid(z).getAtIndex(x, y);
			
			//ImageProcessor ref = image.getStack().getProcessor(z+1);
			for (int k = z-(width/2); k < z+(width/2)+1; k++){
				Grid2D currentSlice = image.getSubGrid(k);
				for (int j = y-(width/2); j < y+(width/2)+1; j++){
					for (int i = x-(width/2); i < x+(width/2)+1; i++){
						float current = currentSlice.getAtIndex(i, j);
						double currentWeight = computePhotometricDistance(current, ref) * computeGeometricCloseness(i,j,k,x,y,z);
						sumWeight += currentWeight;
						sumFilter += currentWeight * current;
					}
				}
			}
		}
		return (float) (sumFilter / sumWeight);
	} 


	public void configure() throws Exception {
		width = UserUtil.queryInt("Enter Width: ", width);
		sigma_r = UserUtil.queryDouble("Sigma for photometric distance: ", sigma_r);
		sigma_d = UserUtil.queryDouble("Sigma for geometric distance: ", sigma_d);
		configured = true;
	}


}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/