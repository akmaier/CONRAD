package edu.stanford.rsl.tutorial.phantoms;

import edu.stanford.rsl.conrad.data.PointwiseIterator;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.Grid4D;
import edu.stanford.rsl.conrad.data.numeric.NumericGridOperator;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;

public class MockupFlow4D {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		CONRAD.setup();
		
		IJ.open("/Users/maier/Documents/data/ERC Grant/Vessels/segment_result_clean2_small.tif");
		ImagePlus vessel = IJ.getImage();
		
		IJ.open("/Users/maier/Documents/data/ERC Grant/intensity_corrected_small.tif");
		ImagePlus bone = IJ.getImage();
		
		
		Grid3D boneGrid = ImageUtil.wrapImagePlus(bone);
		Grid3D vesselGrid = ImageUtil.wrapImagePlus(vessel);
		
		NumericGridOperator op = boneGrid.getGridOperator();
		double max = op.max(boneGrid);
		op.divideBy(boneGrid, (float)max);
		int time = 10;
		double fraction = 1.0 / (time-1);
		
		Grid4D flowMockup = new Grid4D(boneGrid.getSize()[0], boneGrid.getSize()[1], boneGrid.getSize()[2], time);
		for (int i = 0; i < time; i++){
			System.out.println("Time" + i);
			Grid3D grid = new Grid3D(vesselGrid);
			for (int j = grid.getSize()[2]; j > grid.getSize()[2]*i*fraction; --j){
				op.fill(grid.getSubGrid(j-1),0);
			}
			op.addBy(grid, boneGrid);
			flowMockup.setSubGrid(i, grid);
		}
		
		ImagePlus flowImage = ImageUtil.wrapGrid4D(flowMockup, "4D Flow Image Mockup");
		flowImage.show();
	}

}
