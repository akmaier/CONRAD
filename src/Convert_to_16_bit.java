import edu.stanford.rsl.conrad.utils.ImageUtil;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.Calibration;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import ij.process.StackConverter;



public class Convert_to_16_bit implements PlugIn {

	public Convert_to_16_bit(){
		
	}
	
	@Override
	public void run(String arg) {
		ImagePlus image = IJ.getImage();
		// compute min
		double min = ImageUtil.minOfImagePlusValues(image);
		if (min < -32000) min = -32000;
		// set minimum to 0;
		ImageUtil.addToImagePlusValues(image, -min);
		// store conversion setting
		boolean scaling = ImageConverter.getDoScaling();
		// turn conversion off
		ImageConverter.setDoScaling(false);
		// convert to 16 bit
		StackConverter convert = new StackConverter(image);
		convert.convertToGray16();
		// set scaling to previous value
		ImageConverter.setDoScaling(scaling);
		// set calibration
		double [] coeff = {min, 1.0};
		image.getCalibration().setFunction(Calibration.STRAIGHT_LINE, coeff, "Grey Value");
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/