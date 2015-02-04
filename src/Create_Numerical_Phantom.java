import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.io.ImagePlusDataSink;
import edu.stanford.rsl.conrad.phantom.renderer.PhantomRenderer;
import edu.stanford.rsl.conrad.pipeline.ParallelImageFilterPipeliner;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;
import ij.ImagePlus;
import ij.measure.Calibration;
import ij.plugin.PlugIn;


public class Create_Numerical_Phantom implements PlugIn {

	public static PhantomRenderer phantom;

	public Create_Numerical_Phantom(){

	}

	@Override
	public void run(String arg) {
		try{
			phantom = (PhantomRenderer) UserUtil.chooseObject("Select Phantom: ", "Phantom Selection", PhantomRenderer.getPhantoms(), PhantomRenderer.getPhantoms()[0]);
			phantom.configure();
			long time = System.currentTimeMillis();
			Grid3D result = PhantomRenderer.generateProjections(phantom);
			ImagePlus image = ImageUtil.wrapGrid3D(result, phantom.toString());
			Calibration cal = image.getCalibration();
			cal.xOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsX();
			cal.yOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsY();
			cal.zOrigin = Configuration.getGlobalConfiguration().getGeometry().getOriginInPixelsZ();
			cal.pixelWidth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingX();
			cal.pixelHeight = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingY();
			cal.pixelDepth = Configuration.getGlobalConfiguration().getGeometry().getVoxelSpacingZ();
			image.show();
			time = System.currentTimeMillis() - time;
			System.out.println("Runtime: " + time/1000.0 + " s");
		} catch (Exception e){
			e.printStackTrace();
		}
	}

}

/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
