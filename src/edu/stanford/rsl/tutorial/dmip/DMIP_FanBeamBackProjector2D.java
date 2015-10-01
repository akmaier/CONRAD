package edu.stanford.rsl.tutorial.dmip;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
import edu.stanford.rsl.conrad.filtering.redundancy.ParkerWeightingTool;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.StraightLine;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.utils.ImageUtil;
import edu.stanford.rsl.tutorial.fan.FanBeamBackprojector2D;
import edu.stanford.rsl.tutorial.fan.redundancy.ParkerWeights;
import edu.stanford.rsl.tutorial.filters.RamLakKernel;
import edu.stanford.rsl.tutorial.filters.SheppLoganKernel;
import ij.IJ;
import ij.ImageJ;

/**
 * Exercise 7 of Diagnostic Medical Image Processing (DMIP)
 * @author Marco Boegel
 *
 */
public class DMIP_FanBeamBackProjector2D {
	
	//source to detector distance
	private double focalLength;
	//increment of rotation angle beta between two projections in [rad]
	private double betaIncrement;
	//number of pixels on the detector
	private int detectorPixels;
	//number of acquired projections
	private int numProjs;
	//spacing - mm per pixel
	private double detectorSpacing;
	//detector length in [mm] 
	private double detectorLength;
	//max rotation angle beta in [rad]
	private double maxBeta;
	
	public enum RampFilterType {NONE, RAMLAK, SHEPPLOGAN};
	
	public DMIP_FanBeamBackProjector2D( )
	{
		
	}
	
	/**
	 * Initialize all relevant parameters for the reconstruction
	 * @param sino the sinogram
	 * @param focalLength source to detector distance
	 * @param maxRot maximum rotation angle in [rad] at which the sinogram was acquired
	 */
	public void setSinogramParams(Grid2D sino, double focalLength, double maxRot)
	{
		this.focalLength = focalLength;
		this.numProjs = sino.getHeight();
		this.detectorPixels = sino.getWidth();
		this.detectorSpacing = sino.getSpacing()[0];
		this.detectorLength = detectorSpacing*detectorPixels;
		
		
		double halfFanAngle = 0;//TODO
		System.out.println("Half fan angle: " + halfFanAngle*180.0/Math.PI);
		//TODO
		this.betaIncrement = maxBeta /(double) numProjs;
		System.out.println("Short-scan range: " + maxBeta*180/Math.PI);
	}
	
	/**
	 * A pixel driven backprojection algorithm. Cosine, Redundancy and Ramp filters need to be applied separately beforehand
	 * @param sino the filtered sinogram
	 * @param recoSize the dimension of the output image
	 * @param spacing the spacing of the output image
	 * @return the reconstruction
	 */
	public Grid2D backproject(Grid2D sino, int[] recoSize, double[] spacing)
	{
		Grid2D result = new Grid2D(recoSize[0], recoSize[1]);
		result.setSpacing(spacing[0], spacing[1]);
		
		for(int p = 0; p < numProjs; p++)
		{
			//First, compute the rotation angle beta and pre-compute cos(beta), sin(beta)
			float beta = (float) (betaIncrement * p);
			float cosBeta = (float) Math.cos(beta);
			float sinBeta = (float) Math.sin(beta);
			
			//Compute direction and normal of the detector at the current rotation angle
			final PointND detBorder = new PointND();//TODO
			final SimpleVector dirDet = new SimpleVector();//TODO
			final StraightLine detLine = new StraightLine(detBorder, dirDet);
			
			//Compute rotated source point
			final PointND source = new PointND(focalLength * cosBeta, focalLength*sinBeta, 0.d);
			
			//pick current projection
			Grid1D currProj = sino.getSubGrid(p);
			
			//pixel driven BP: iterate over all output image pixels
			for(int x = 0; x < recoSize[0]; x++)
			{
				//transform the image pixel coordinates to world coordinates
				float wx =0; //TODO
				
				for(int y = 0; y < recoSize[1]; y++)
				{
					float wy = 0;//TODO
					
					final PointND reconstructionPointWorld = new PointND(wx, wy, 0.d);

					//intersect the projection ray with the detector
					//TODO
					final PointND detPixel = new PointND();//TODO
					
					float valueTemp;
					
					if(detPixel != null)
					{
						//calculate position of the projected point on the 1D detector
						final SimpleVector pt = SimpleOperators.subtract(detPixel.getAbstractVector(), detBorder.getAbstractVector());
						double length = pt.normL2();
						//deal with truncation
						if(pt.getElement(0)*dirDet.getElement(0)+pt.getElement(1)*dirDet.getElement(1)<0)
						{
							length -= length;
						}
						double t = (length - 0.5d)/detectorSpacing;
						
						//check if projection of this world point hit the detector
						if(currProj.getSize()[0] <= t+1 || t< 0)
							continue;
						
						float value = InterpolationOperators.interpolateLinear(currProj, t);
					
						//Apply distance weighting
						//see Fig 1a) exercise sheet
						//TODO
						//TODO
						float dWeight = 0;//TODO
						valueTemp = (float) (value / (dWeight*dWeight));
					}
					else
					{
						//projected pixel lies outside the detector
						valueTemp = 0.f;
					}
					
					result.addAtIndex(x, y, valueTemp);
				}
			}
		}
		
		//adjust scale. required because of shortscan
		float normalizationFactor = (float) (numProjs / Math.PI);
		NumericPointwiseOperators.divideBy(result, normalizationFactor);

		
		return result;
	}
	
	/**
	 * Cosine weighting for 2D sinograms
	 * @param sino the sinogram
	 * @return the cosine weighted sinogram
	 */
	public Grid2D applyCosineWeights(Grid2D sino)
	{
		Grid2D result = new Grid2D(sino);
		
		//Create 1D kernel (identical for each projection)
		Grid1D cosineKernel = new Grid1D(detectorPixels);
		for(int i=0; i<detectorPixels; ++i){
			//TODO
			cosineKernel.setAtIndex(i, 9999);//TODO
		}
		
		//apply cosine weights to each projection
		for(int p = 0; p < numProjs; p++)
		{
			NumericPointwiseOperators.multiplyBy(result.getSubGrid(p), cosineKernel);
		}
		return result;
	}
	
	/**
	 * Parker redundancy weights for a 2D sinogram
	 * @param sino the sinogram
	 * @return parker weighted sinogram
	 */
	public Grid2D applyParkerWeights(Grid2D sino)
	{
		Grid2D result = new Grid2D(sino);
		Grid2D parker = new Grid2D(sino.getWidth(), sino.getHeight());
		
		// Initialize parameters
		
		double delta = (double) (Math.atan((detectorLength / 2.d)/ focalLength) );
		double beta, gamma;

		// iterate over the detector elements
		for (int t = 0; t < detectorPixels; ++t) {
			// compute alpha of the current ray (detector element)
			gamma = Math.atan((t * detectorSpacing - detectorLength / 2.d + 0.5*detectorSpacing) / focalLength);
			
			// iterate over the projection angles
			for (int b = 0; b < numProjs; ++b) {
				beta = b * betaIncrement;
				
				// Shift weights such that they are centered (Important for maxBeta < pi + 2 * gammaM)
					beta += (Math.PI+2*delta-maxBeta)/2.0;
				
				// Adjust beta if out of range [0, 2*pi]
				if (beta < 0) {
					continue;
				}
				if (beta > Math.PI *2.d) {
					continue;
				}

				// implement the conditions as described in Parker's paper
				if (beta <= 2 * (delta - gamma)) {
					float val = 0; //TODO
					if (Double.isNaN(val)){
						continue;
					}
					parker.setAtIndex(t, b , val);

				} else if (beta < Math.PI - 2.d * gamma) {
					parker.setAtIndex(t, b , 1);
				}
				else if (beta <= (Math.PI + 2.d * delta) + 1e-12) {
					float val = 0;//TODO
					if (Double.isNaN(val)){
						continue;
					}
					parker.setAtIndex(t, b , val);
				}
			}
		}
		
		// Correct for scaling due to varying angle
		NumericPointwiseOperators.multiplyBy(parker, (float)( maxBeta / (Math.PI)));
		parker.show();
		
		//apply the parker weights to the sinogram
		NumericPointwiseOperators.multiplyBy(result, parker);
		
		return result;
	}
	
	
	public static void main(String arg[])
	{
		ImageJ ij = new ImageJ();
		
		double focalLength = 600.f;
		//maximum rotation angle in [rad]
		double maxRot = Math.PI;
		//choose the ramp filter (none, ramlak, shepplogan)
		RampFilterType filter = RampFilterType.RAMLAK;
		
		DMIP_FanBeamBackProjector2D fbp = new DMIP_FanBeamBackProjector2D();

		//Load and visualize the projection image data
		String filename = "D:/02_lectures/DMIP/exercises/2014/6/Sinogram0.tif";
		Grid2D sino = ImageUtil.wrapImagePlus(IJ.openImage(filename)).getSubGrid(0);
		sino.show("Sinogram");
		
		//getSubGrid() only yields rows. -> Transpose so that each row is a projection
		sino = sino.getGridOperator().transpose(sino);
		sino.show();
		
		
		//Initialize parameters
		double detectorSpacing = 0.5;
		sino.setSpacing(detectorSpacing, 0);
		fbp.setSinogramParams(sino, focalLength, maxRot);
		sino.setSpacing(fbp.detectorSpacing, fbp.betaIncrement);
		
		//apply cosine weights
		Grid2D cosineWeighted = fbp.applyCosineWeights(sino);
		cosineWeighted.show("After Cosine Weighting");
		
		//apply parker redundancy weights
		Grid2D parkerWeighted = fbp.applyParkerWeights(cosineWeighted);
		parkerWeighted.show("After Parker Weighting");
		
		//apply ramp filter
		switch(filter){
		case RAMLAK:
			RamLakKernel ramLak = new RamLakKernel(fbp.detectorPixels, fbp.detectorSpacing);
			for(int i = 0; i < fbp.numProjs; i++)
			{
				ramLak.applyToGrid(parkerWeighted.getSubGrid(i));
			}
			parkerWeighted.show("After RamLak Filter");
			break;
		case SHEPPLOGAN:
			SheppLoganKernel sheppLogan = new SheppLoganKernel(fbp.detectorPixels, fbp.detectorSpacing);
			for(int i = 0; i < fbp.numProjs; i++)
			{
				sheppLogan.applyToGrid(parkerWeighted.getSubGrid(i));
			}
			parkerWeighted.show("After Shepp-Logan Filter");
			break;
		case NONE:
		default:
			
		}
		
		
		//setup reconstruction image size
		int[] recoSize = new int[]{128, 128};
		double[] spacing = new double[]{1.0, 1.0};
		
		Grid2D reconstruction = fbp.backproject(parkerWeighted, recoSize, spacing);
		reconstruction.show("Reconstructed image");
		
		
		//Compare our backprojection algorithm to the tutorial code
		FanBeamBackprojector2D fbp2 = new FanBeamBackprojector2D(focalLength, fbp.detectorSpacing, fbp.betaIncrement,recoSize[0], recoSize[1]);
		fbp2.backprojectPixelDriven(parkerWeighted).show("Tutorial reconstruction");
	
	}
}


