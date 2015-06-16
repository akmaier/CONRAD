/*
 * Copyright (C) 2010-2014 - Mathias Unberath 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;


public class Replace_Intensity_Value_Tool  implements PlugIn {

	public Replace_Intensity_Value_Tool() {
		
	}
	
	/**
	 * 
	 */
	@Override
	public void run(String arg) {
		ImagePlus image = IJ.getImage();
		
		int val = 0;
	    int rep = 0;
	    
	    GenericDialog gd = new GenericDialog("Replace intensity values.");
	    gd.addNumericField("Value to be replaced: ", val, 0);
	    gd.addNumericField("Replace with: ", rep, 0);
	    gd.showDialog();
	    if (gd.wasCanceled()) return;
	    val = (int)gd.getNextNumber();
	    rep = (int)gd.getNextNumber();
	     
	    System.out.println("Replacing "+val+" with "+rep+".");
	    
	    Grid3D img = ImageUtil.wrapImagePlus(image, true);
	    int[] size = img.getSize();
	    for(int i = 0; i < size[0]; i++){
	    	for(int j = 0; j < size[1]; j++){
	    		for(int k = 0; k < size[2]; k++){
	    			if(img.getAtIndex(i, j, k) == val){
	    				img.setAtIndex(i, j, k, rep);
	    			}
	    		}
	    	}
	    }
	    ImagePlus processed = ImageUtil.wrapGrid(img, "Processed image");
		processed.show();
	}

}
