package TruncationPolynomial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

public class FanBeamProjection extends Grid2D{

	
	public FanBeamProjection(int width, int height) {
		super(width, height);
		// TODO Auto-generated constructor stub
	}

	public static void main(String args[]) {
		new ImageJ();
		int phantomgroesse = 128;
		
		Phantom phantom = new Phantom(phantomgroesse, phantomgroesse);
		phantom.show();
		Phantom sinogram = Phantom.getSinogram(phantom);
//		sinogram.show();
		double D = 1*(int)Math.sqrt(Math.pow(phantom.getWidth(),2) + Math.pow(phantom.getHeight(),2));
		Phantom fanogram = getFanogram(phantom, D, 360);
//		fanogram.show();
//		
//		NumericPointwiseOperators.subtractedBy(sinogram, fanogram).show();
		Phantom rebinnedSinogram = rebin(fanogram, D);
//		rebinnedSinogram.show();
		
//		NumericPointwiseOperators.subtractedBy(sinogram, rebinnedSinogram).show();
		
		Phantom ramlacSinogram = BackProjection.ramlac(rebinnedSinogram);
//		ramlacSinogram.show();
		
		Phantom reconstructed = BackProjection.backproject(ramlacSinogram);
		reconstructed.show();
		
		Phantom shortFanogram = getShortFanogram(phantom, D, 5000);
//		shortFanogram.show();
		Phantom rebinnedShortSinogram = shortRebin(shortFanogram, D);
//		rebinnedShortSinogram.show();
		Phantom ramlacShortSinogram = BackProjection.ramlac(rebinnedShortSinogram);
		//ramlacShortSinogram.show();
		Phantom shortReconstructed = shortBackproject(ramlacShortSinogram, D);
//		NumericPointwiseOperators.divideBy(shortReconstructed, 500);
		shortReconstructed.show();
		
		NumericPointwiseOperators.subtractedBy(phantom, shortReconstructed).show();
	}

	private static Phantom getShortFanogram(Phantom phantom, double D, int steps) {
		
		Phantom fanogram = new Phantom((int)Math.sqrt(Math.pow(phantom.getWidth(),2) + Math.pow(phantom.getHeight(),2)), steps);
		
		double scanAngle = Math.PI + 2*Math.atan(fanogram.getWidth()/(2*D));
		
		for (int betaGrad = 0; betaGrad < steps; betaGrad++) {
			double beta = (betaGrad)* scanAngle/steps;
			
			for (int t = -fanogram.getWidth()/2; t < fanogram.getWidth()/2; t++) {
				double gamma = Math.atan((t/D));
				double theta = gamma + beta;
				double s = D * Math.sin(gamma);
				
				double sum = 0;
				
				if ((theta <= Math.PI*0.25) || (theta >= Math.PI*0.75 && theta <= Math.PI*1.25) || (theta >= 1.75 * Math.PI)) {
					for (int y = 0; y < phantom.getHeight(); y++) {
						double x = (s - (y - phantom.getHeight()/2) * Math.sin(theta))/(Math.cos(theta)) + phantom.getWidth()/2;
						float interpValue = InterpolationOperators.interpolateLinear(phantom, x, y);
						sum += interpValue;
					}
				} else {
					for (int x = 0; x < phantom.getWidth(); x++) {
						double y = (s - (x - phantom.getWidth()/2) * Math.cos(theta))/(Math.sin(theta)) + phantom.getHeight()/2;
						float interpValue = InterpolationOperators.interpolateLinear(phantom, x, y);
						sum += interpValue;
					}
				}
				
				fanogram.putPixelValue(t+fanogram.getWidth()/2, betaGrad, sum);
			}
			
			
		}
		
		
		return fanogram;
	}
	
	public static Phantom getFanogram(Phantom phantom, double D, int steps) {
		
		Phantom fanogram = new Phantom((int)Math.sqrt(Math.pow(phantom.getWidth(),2) + Math.pow(phantom.getHeight(),2)), steps);
		
		for (int betaGrad = 0; betaGrad < steps; betaGrad++) {
			double beta = (betaGrad) * 2 * Math.PI/steps;
			
			for (int t = -fanogram.getWidth()/2; t < fanogram.getWidth()/2; t++) {
				double gamma = Math.atan((t/D));
				double theta = gamma + beta;
				double s = D * Math.sin(gamma);
				
				double sum = 0;
				
				if ((theta <= Math.PI*0.25) || (theta >= Math.PI*0.75 && theta <= Math.PI*1.25) || (theta >= 1.75 * Math.PI)) {
					for (int y = 0; y < phantom.getHeight(); y++) {
						double x = (s - (y - phantom.getHeight()/2) * Math.sin(theta))/(Math.cos(theta)) + phantom.getWidth()/2;
						float interpValue = InterpolationOperators.interpolateLinear(phantom, x, y);
						sum += interpValue;
					}
				} else {
					for (int x = 0; x < phantom.getWidth(); x++) {
						double y = (s - (x - phantom.getWidth()/2) * Math.cos(theta))/(Math.sin(theta)) + phantom.getHeight()/2;
						float interpValue = InterpolationOperators.interpolateLinear(phantom, x, y);
						sum += interpValue;
					}
				}
				
				fanogram.putPixelValue(t+fanogram.getWidth()/2, betaGrad, sum);
			}
			
			
		}
		
		
		return fanogram;
	}
	
	public static Phantom rebin(Phantom fanogram, double D) {
		
		Phantom sinogram = new Phantom(fanogram.getWidth(), fanogram.getHeight());
		
		for (int s = -sinogram.getWidth()/2; s < sinogram.getWidth()/2; s++) {
			for (int thetaGrad = 0; thetaGrad < sinogram.getHeight(); thetaGrad++) {
				double theta = (thetaGrad)* 2 * Math.PI/360;
				
				double gamma = Math.asin(s/D);
				double t = D * Math.tan(gamma);
				double beta = theta-gamma;
				double betaGrad = beta * 360 /(2*Math.PI);
				
				if (betaGrad > 359){
					betaGrad -= 359;
				}
				if (betaGrad < 0) {
					betaGrad += 359;					
				}
				
				float interpValue = InterpolationOperators.interpolateLinear(fanogram, t+fanogram.getWidth()/2, betaGrad);
				sinogram.putPixelValue(s+sinogram.getWidth()/2, thetaGrad, interpValue);
				
				
			}
		}
		
		
		return sinogram;
	}
	
	public static Phantom shortRebin(Phantom fanogram, double D) {
		
		Phantom sinogram = new Phantom(fanogram.getWidth(), fanogram.getHeight());
		double scanAngle = Math.PI + 2*Math.atan(fanogram.getWidth()/(2*D));
		
		for (int s = -sinogram.getWidth()/2; s < sinogram.getWidth()/2; s++) {
			for (int thetaGrad = 0; thetaGrad < sinogram.getHeight(); thetaGrad++) {
				double theta = (thetaGrad)* scanAngle/fanogram.getHeight();// 2 * Math.PI/360;
				
				double gamma = Math.asin(s/D);
				double t = D * Math.tan(gamma);
				double beta = theta-gamma;
				double betaGrad = beta * fanogram.getHeight()/scanAngle;//360 /(2*Math.PI);
				
				if (betaGrad < 0){// - 1.0/fanogram.getHeight()) {
					betaGrad = betaGrad - (-2.0*gamma - Math.PI)*(fanogram.getHeight()/scanAngle);//(fanogram.getHeight()-1);
					t = -t;
				} else if (betaGrad > fanogram.getHeight()-1) {
					betaGrad = betaGrad + (2.0*gamma - Math.PI)*(fanogram.getHeight()/scanAngle);//(fanogram.getHeight()-1);					
					t = -t;
				}
				
				float interpValue = InterpolationOperators.interpolateLinear(fanogram, t+fanogram.getWidth()/2, betaGrad);
				sinogram.putPixelValue(s+sinogram.getWidth()/2, thetaGrad, interpValue);
				
				
			}
		}
		
		
		return sinogram;
	}
	
	public static Phantom shortBackproject(Phantom sinogram, double D) {
		
		int a = Math.round((float)(sinogram.getWidth()/Math.sqrt(2)));
		Phantom reconstructed = new Phantom(a,a);
		
		double scanAngle = Math.PI + 2*Math.atan(sinogram.getWidth()/(2*D));
		
		for (int y = 0; y < reconstructed.getHeight(); y++) {
			for (int x = 0; x < reconstructed.getWidth(); x++) {
				
				for (double theta = Math.atan(sinogram.getWidth()/(2*D)); theta < Math.PI + Math.atan(sinogram.getWidth()/(2*D)); theta += scanAngle/sinogram.getHeight()) {
					//double theta = (thetaGrad)* scanAngle/sinogram.getHeight();
											
					double s = (x-reconstructed.getWidth()/2) * Math.cos(theta) + (y-reconstructed.getHeight()/2) * Math.sin(theta);
					// float interpValue =sinogram.getAtIndex((int)(s+sinogram.getWidth()/2), thetaGrad);
					double thetaGrad = theta * (sinogram.getHeight()/scanAngle);
					float interpValue = InterpolationOperators.interpolateLinear(sinogram, s + sinogram.getWidth()/2, thetaGrad);
					reconstructed.addAtIndex(x, y, interpValue);
				}
				
				
			}
		}
		
		NumericPointwiseOperators.divideBy(reconstructed, NumericPointwiseOperators.max(reconstructed));
		
		return reconstructed;
	}
}
