import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;



public class Normalize_Image_MinMax implements PlugIn {

	public Normalize_Image_MinMax(){
		
	}
	
	@Override
	public void run(String arg) {
		ImagePlus image = IJ.getImage();
		ImageUtil.normalizeImagePlusMinMax(image);
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/