package edu.stanford.rsl.conrad.filtering;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public class KinectBasedDetectorSaturationCorrectionTool extends IndividualImageFilteringTool {
	
	/**	 
	 * 
	 * @author Jang-Hwan Choi, Johannes Rausch
	 *
	 */
	private static final long serialVersionUID = 450437133500604807L;
	
	Configuration config = Configuration.getGlobalConfiguration();
	private boolean isDisplay = true;
		
	public KinectBasedDetectorSaturationCorrectionTool (){
		configured = true;
	}
			
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) throws Exception {
		FloatProcessor imp = new FloatProcessor(imageProcessor.getWidth(),imageProcessor.getHeight());
		imp.setPixels(imageProcessor.getBuffer());
		
		ImageProcessor imp1 = imp.duplicate();	// original
		SimpleMatrix mat = config.getGeometry().getProjectionMatrix(imageIndex).computeP();
				
		double [][] KinectDots3D = {{10, 10, 10}, {50, 50, 50}, {-20, -20, -20}}; // [beadNo][x,y,z]		
		double [] uv = new double[1];

		for (int i=0; i< KinectDots3D.length; i++){
			
			if (KinectDots3D[i][0] != 0 || KinectDots3D[i][1] != 0 || KinectDots3D[i][2] != 0){ 
				
				uv = compute2DCoordinates(KinectDots3D[i], mat);
				
				if (isDisplay) {
					imp1.setValue(2);
					imp1.drawLine((int) Math.round(uv[0]-10), (int) Math.round(uv[1]-10), (int) Math.round(uv[0]+10), (int) Math.round(uv[1]+10));
					imp1.drawLine((int) Math.round(uv[0]-10), (int) Math.round(uv[1]+10), (int) Math.round(uv[0]+10), (int) Math.round(uv[1]-10));
				}
				
			}			
		}

		if (isDisplay) {
			for (int x=0; x< config.getGeometry().getDetectorWidth(); x+=100)
				imp1.drawLine(x, 0, x, config.getGeometry().getDetectorHeight());
			for (int y=0; y< config.getGeometry().getDetectorHeight(); y+=100)
				imp1.drawLine(0, y, config.getGeometry().getDetectorWidth(), y);
		}
				
		Grid2D result = new Grid2D((float[]) imp1.getPixels(), imp1.getWidth(), imp1.getHeight());

		return result;
	}
	
	private double [] compute2DCoordinates(double [] point3D, SimpleMatrix pMatrix){
		
		// Compute coordinates in projection data.
		SimpleVector homogeneousPoint = SimpleOperators.multiply(pMatrix, new SimpleVector(point3D[0], point3D[1], point3D[2], 1));
		// Transform to 2D coordinates
		double coordU = homogeneousPoint.getElement(0) / homogeneousPoint.getElement(2);
		double coordV = homogeneousPoint.getElement(1) / homogeneousPoint.getElement(2);
		
		//double pxlSize = config.getGeometry().getPixelDimensionX();

		return new double [] {coordU, coordV};
	}
		
	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new KinectBasedDetectorSaturationCorrectionTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Kinect-based Detector Saturation Correction Tool";
	}

	@Override
	public void configure() throws Exception {
		setConfigured(true);
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

}

/*
 * Copyright (C) 2010-2014 - Jang-hwan Choi, Johannes Rausch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
