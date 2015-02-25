package TruncationPolynomial;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.filtering.CosineWeightingTool;
import edu.stanford.rsl.conrad.filtering.ImageFilteringTool;
import edu.stanford.rsl.conrad.filtering.RampFilteringTool;
import edu.stanford.rsl.conrad.filtering.TruncationCorrectionTool;
import edu.stanford.rsl.conrad.filtering.rampfilters.RamLakRampFilter;
import edu.stanford.rsl.conrad.filtering.redundancy.ParkerWeightingTool;
import edu.stanford.rsl.conrad.geometry.trajectories.Trajectory;
import edu.stanford.rsl.conrad.io.TiffProjectionSource;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.FileUtil;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.cone.ConeBeamBackprojector;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;

public class ConebeamPipeline {

	public static void main(String[] args) {
		ImagePlus imp;
		Grid3D impAsGrid = null;

		try {
			// we need ImageJ in the following
			new ImageJ();
			String filenameString = "/proj/ciptmp/recoData/DensityProjection_No248_Static60_0.8deg_REFERENCE.tif";
			// call the ImageJ routine to open the image:
			imp = IJ.openImage(filenameString);
			// Convert from ImageJ to Grid3D. Note that no data is copied here.
			// The ImageJ container is only wrapped. Changes to the Grid will also affect the ImageJ ImagePlus.
			impAsGrid = ImageUtil.wrapImagePlus(imp);
			// Display the data that was read from the file.
			impAsGrid.show("Data from file");
		} catch (Exception e) {
			e.printStackTrace();
		}

		Configuration conf = Configuration.loadConfiguration("/proj/ciptmp/recoData/CONRADsettings.xml");
		Configuration.setGlobalConfiguration(conf);

		Trajectory geo = conf.getGeometry();

		CosineWeightingTool cosineWT = new CosineWeightingTool();
		cosineWT.configure();
		ParkerWeightingTool parkerWT = new ParkerWeightingTool(geo);
		RampFilteringTool rampFT = new RampFilteringTool();
		
		try {
			rampFT.configure();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		try {
			parkerWT.configure();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		ImageFilteringTool[] filters = {cosineWT, parkerWT, rampFT};
		Grid3D filtered = ImageUtil.applyFiltersInParallel(impAsGrid, filters);
		
		
		ConeBeamBackprojector coneBeamBP = new ConeBeamBackprojector();
		Grid3D result = coneBeamBP.backprojectPixelDrivenCL(filtered);
		
		result.show("Manual Pipeline");


	}

}
