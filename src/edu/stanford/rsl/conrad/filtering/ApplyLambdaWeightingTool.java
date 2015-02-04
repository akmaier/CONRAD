package edu.stanford.rsl.conrad.filtering;

import ij.ImagePlus;

import javax.swing.JOptionPane;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.utils.BilinearInterpolatingDoubleArray;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.conrad.utils.UserUtil;



/**
 * This class applies the beam hardening correction after Joseph & Spittal in projection domain.
 * The algorith works as follows:<BR>
 * <li>
 * A water corrected reconstruction is required. 
 * <li>Next, the the density values of the hard material</li>
 * have to be corrected using a VolumeAttenuationFactorCorrectionTool</li>
 * <li>After that only the hard material is left in the volume. This has to be forward projected</li>
 * <li>The ApplyLambdaWeightingTool has to be applied to correct the beam hardening in projection domain. A calibration table is required for this step. It can be generated using the Generate_Beam_Hardening_Lookup_Table plugin.</li>
 * <li>The projections can be reconstructed to the final image.</li>
 * <BR><BR>
 * @author akmaier
 * @see edu.stanford.rsl.conrad.cuda.CUDAForwardProjector
 * @see VolumeAttenuationFactorCorrectionTool
 *
 */
public class ApplyLambdaWeightingTool extends IndividualImageFilteringTool{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8909824388187619416L;
	private double measureHard = 2.7;
	private double measureSoft = 1;
	private double lambda0;
	private BilinearInterpolatingDoubleArray lut;
	private double offset = 0.0;
	private double projectionXShift = 0;
	private double projectionYShift = 0;
	private double slope = 1;
	private Grid3D hardMaterial;
	
	@Override
	public ApplyLambdaWeightingTool clone() {
		ApplyLambdaWeightingTool clone = new ApplyLambdaWeightingTool();
		clone.lut = lut;
		clone.offset = offset;
		clone.slope = slope;
		clone.measureHard = measureHard;
		clone.measureSoft = measureSoft;
		clone.projectionXShift = projectionXShift;
		clone.projectionYShift = projectionYShift;
		clone.lambda0 = lambda0;
		clone.hardMaterial = hardMaterial;
		return clone;
	}

	@Override
	public void prepareForSerialization(){
		super.prepareForSerialization();
		hardMaterial = null;
		configured = false;
		lut = null;
	}

	@Override
	public String getToolName() {
		return "Apply Beam Hardening Correction (Lambda Weighting)";
	}
	
	@Override
	public void configure() throws Exception {
		slope = UserUtil.queryDouble("Enter slope of correction transformatoin", slope);
		//offset = UserUtil.queryDouble("Enter offset of correction transformation", offset);
		measureHard = UserUtil.queryDouble("Measurement of hard material [HU]", measureHard);
		measureSoft= UserUtil.queryDouble("Measurement of soft material [HU]", measureSoft);
		projectionXShift = UserUtil.queryDouble("Enter projection x shift", projectionXShift);
		projectionYShift = UserUtil.queryDouble("Enter projection y shift", projectionYShift);
		lambda0 = VolumeAttenuationFactorCorrectionTool.getLambda0(measureHard, measureSoft);
		lut = Configuration.getGlobalConfiguration().getBeamHardeningLookupTable();
		ImagePlus [] images = ImageUtil.getAvailableImagePlusAsArray();
		hardMaterial = ImageUtil.wrapImagePlus(
				(ImagePlus) JOptionPane.showInputDialog(null, "Select projections with hard material: ", "Select source", JOptionPane.PLAIN_MESSAGE, null, images, images[0]));
		configured = true;
	}

	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor)
			throws Exception {
		Grid2D secondImageProcessor = imageProcessor;
		Grid2D firstImageProcessor = hardMaterial.getSubGrid(imageIndex);
		int xoffset = (firstImageProcessor.getWidth() - secondImageProcessor.getWidth()) / 2;
		int yoffset = (firstImageProcessor.getHeight() - secondImageProcessor.getHeight()) / 2;
		for (int i = 0; i < firstImageProcessor.getWidth(); i++) {
			for (int j = 0; j < firstImageProcessor.getHeight(); j++) {
				double value = slope * InterpolationOperators.interpolateLinear(firstImageProcessor, i+projectionXShift + xoffset, j+projectionYShift + yoffset);
				double value2 = secondImageProcessor.getPixelValue(i, j);
				double lambda = lambda0;
				try {
					if(value > 0.02) {
						lambda = lut.getValue(value2, value);
					}
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				secondImageProcessor.putPixelValue(i,j, value2 + (lambda0-lambda) * value);
			}
		}
		return secondImageProcessor;
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getBibtexCitation() {
		String bibtex = "@article{Joseph78-AMF,\n" +
		"  author = {{Joseph}, P. M. and {Spital}, R. D.},\n" +
		"  title = {{A method for correcting bone induced artifacts in computed tomography scanners}},\n" +
		"  journal = {{Journal of Computer Assisted Tomography}},\n" +
		"  volume = {2}, \n" +
		"  pages= {100-108},\n" +
		"  year = {1978}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Joseph PM, Spital RD. A method for correcting bone induced artifacts in computed tomography scanners, JCAT 1978;2:100-8.";
	}

}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
