package edu.stanford.rsl.conrad.angio.preprocessing.segmentation.hessian.tools;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Roi;
import ij.measure.Calibration;
import ij.process.FloatProcessor;

public class VesselnessProcessor extends HessianEvalueProcessor {

	private double gammaThreshold = 0.1;
	private double gammaMaxVal = 0.1;
	
	
	private double gammaMult = 1f;
	private double gamma = 1f;
	private double beta = 0.5f;
	
	private Roi roi = null;
	
	public VesselnessProcessor(double sigma, double gamma, boolean useCalibration) {
		this.gammaMult = gamma;
		this.sigma = sigma;
		this.useCalibration = useCalibration;
	}

	@Override
	public ImagePlus generateImage(ImagePlus original) {

		Calibration calibration=original.getCalibration();

		@SuppressWarnings("unused")
		float sepX = 1, sepY = 1, sepZ = 1;
		if( useCalibration && (calibration!=null) ) {
			sepX = (float)calibration.pixelWidth;
			sepY = (float)calibration.pixelHeight;
			sepZ = (float)calibration.pixelDepth;
		}

		double minimumSeparation = Math.min(sepX,
						    Math.min(sepY,sepX));

		ComputeCurvatures c = new ComputeCurvatures(original, sigma, this, useCalibration);
		IJ.showStatus("Convolving with Gaussian \u03C3="+sigma+" (min. pixel separation: "+minimumSeparation+")...");
		c.run();

		int width = original.getWidth();
		int height = original.getHeight();
		int depth = original.getStackSize();

		ImageStack stack = new ImageStack(width, height);

		float[] evalues = new float[3];

		IJ.showStatus("Calculating Hessian eigenvalues at each point...");

		if(roi == null){
			this.roi = new Roi(0,0,width+1,height+1);
		}
		
		float minResult = Float.MAX_VALUE;
		float maxResult = Float.MIN_VALUE;

		if( depth == 1 ) {

			float[] e1 = new float[width * height];
			float[] e2 = new float[width * height]; // this is the larger one
			float[] structureness = new float[width * height];

			for (int y = 1; y < height - 1; ++y) {
				for (int x = 1; x < width - 1; ++x) {
					
					float value = 0;
					int index = y * width + x;					
					if(roi.contains(x, y)){
						boolean real = c.hessianEigenvaluesAtPoint2D(x, y,
										     true, // order absolute
										     evalues,
										     normalize,
										     false,
										     sepX,
										     sepY);
						if( real ){
							structureness[index] = (float)structurenessFromEvalues(evalues);
							e1[index] = evalues[0];
							e2[index] = evalues[1];
						}
					}				
					
					if( value < minResult )
						minResult = value;
					if( value > maxResult )
						maxResult = value;

				}				
			}
			double maxMag = -Double.MAX_VALUE;
			for (int y = 1; y < height - 1; ++y) {
				for (int x = 1; x < width - 1; ++x) {
					int index = y * width + x;					
					if(roi.contains(x, y)){
						maxMag = Math.max(maxMag, structureness[index]);
					}
				}
			}
			FloatProcessor sp = new FloatProcessor(width, height);
			sp.setPixels(structureness);
			int[] magnitudes = sp.convertToShort(true).getHistogram();
			int mL = magnitudes.length;
			int nAll = roi.getBounds().width * roi.getBounds().height;
			int cM = 0;
			int histIndex = 0;
			for(int i = 1; i < mL; i++){
				if(((float)cM)/nAll < gammaMult){
					cM += magnitudes[i];
					histIndex = i;
				}else{
					break;
				}
			}
			double histVal = maxMag * ((histIndex))/(mL-1);
			if(gammaMult > 1){
				histVal = histVal * gammaMult;
			}
			this.gamma = histVal;
			//System.out.println(gamma);
			if(gamma > gammaThreshold){
				gamma = gammaMaxVal;
			}
			
			
			float[] slice = new float[width * height];
			for (int y = 1; y < height - 1; ++y) {
				for (int x = 1; x < width - 1; ++x) {
					int index = y * width + x;
					float value = measureFromEvalues2D(new float[]{e1[index],e2[index]});
					slice[index] = value; 
					if( value < minResult )
						minResult = value;
					if( value > maxResult )
						maxResult = value;
				}				
			}
			FloatProcessor fp = new FloatProcessor(width, height);
			fp.setPixels(slice);
			stack.addSlice(null, fp);
		} else {
			System.out.println("Not implemented for 3D, yet.");
		}

		IJ.showProgress(1.0);

		ImagePlus result=new ImagePlus("processed " + original.getTitle(), stack);
		result.setCalibration(calibration);


		result.getProcessor().setMinAndMax(minResult,maxResult);
		result.updateAndDraw();

		return result;
	}
	
	private double structurenessFromEvalues(float[] evalues) {
		double s = 0;
		for(int i = 0; i < evalues.length; i++){
			s += evalues[i]*evalues[i];
		}
		return Math.sqrt(s);
	}

	public float measureFromEvalues2D( float [] evalues ) {

		/* If either of the two principle eigenvalues is
		   positive then the curvature is in the wrong
		   direction - towards higher instensities rather than
		   lower. */

		if (evalues[1] >= 0)
			return 0;
		else{
			float rB = evalues[0] / evalues[1];
			double s = structurenessFromEvalues(evalues);
			return (float)( Math.exp(-rB*rB / (2*beta*beta)) * (1 - Math.exp(-s*s / (2*gamma*gamma))) );
		}
	}

	public float measureFromEvalues3D( float [] evalues ) {

		/* If either of the two principle eigenvalues is
		   positive then the curvature is in the wrong
		   direction - towards higher instensities rather than
		   lower. */

		if ((evalues[1] >= 0) || (evalues[2] >= 0))
			return 0;
		else
			return (float)Math.sqrt(evalues[2] * evalues[1]);
	}

	public Roi getRoi() {
		return roi;
	}

	public void setRoi(Roi roi) {
		this.roi = roi;
	}
	
	public void setGammaThreshold(double[] gammaVals) {
		this.gammaThreshold = gammaVals[0];
		this.gammaMaxVal = gammaVals[1];
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}
}