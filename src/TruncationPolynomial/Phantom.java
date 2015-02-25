package TruncationPolynomial;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;

public class Phantom extends Grid2D{

	public Phantom(int width, int height) {
		super(width, height);
		
		this.draw(this);
//		for (int i = 20; i < 100; i++) {
//			for (int j = 20; j < 100; j++) {
//				this.putPixelValue(i, j, 1);
//			}
//			
//		}
		// a, b, midPointx, midPointy, intesity, theta
		
	}
	
	public Phantom(int width, int height, boolean a) {
		super(width, height);
		
		if (a) {
			this.draw(this);	
		} else {
			for (int i = (int) Math.round(0.6*height); i < (int) Math.round(0.9*height); i++) {
				for (int j = (int) Math.round(0.6*width); j < (int) Math.round(0.9*width); j++) {
					this.putPixelValue(i, j, 1);
				}
			
			}
		}

		// a, b, midPointx, midPointy, intesity, theta
		
	}

	public static void main(String args[]) {
		Phantom test = new Phantom(128,128);
		test.show();
		Phantom sinogram = getSinogram(test);
		sinogram.show();
	}
	
	public static Phantom getSinogram(Phantom bio) {
		double grad = 180;
		Phantom sinogram = new Phantom((int)Math.sqrt(Math.pow(bio.getWidth(),2) + Math.pow(bio.getHeight(),2)),(int)grad);
		
		for (int s = -sinogram.getWidth()/2; s < sinogram.getWidth()/2; s++) {
			for (int thetaGrad = 0; thetaGrad < sinogram.getHeight(); thetaGrad++) {
				double theta = (thetaGrad)* Math.PI/grad;
				
				double sum = 0;
				
				if (theta <= Math.PI*0.25 || theta >= Math.PI*0.75) {
					for (int y = 0; y < bio.getHeight(); y++) {
						double x = (s - (y - bio.getHeight()/2) * Math.sin(theta))/(Math.cos(theta)) + bio.getWidth()/2;
						float interpValue = InterpolationOperators.interpolateLinear(bio, x, y);
						sum += interpValue;
					}
				} else {
					for (int x = 0; x < bio.getWidth(); x++) {
						double y = (s - (x - bio.getWidth()/2) * Math.cos(theta))/(Math.sin(theta)) + bio.getHeight()/2;
						float interpValue = InterpolationOperators.interpolateLinear(bio, x, y);
						sum += interpValue;
					}
				}
				
				sinogram.putPixelValue(s+sinogram.getWidth()/2, thetaGrad, sum);
			
			}
//			System.out.println(s);
		}
		
		
		return sinogram;
	}

	public void draw(Phantom orig) {
		
		for (int row = 0; row < orig.getHeight(); row++) {
			for (int col = 0; col < orig.getWidth(); col++) {

//				orig.drawEllipse(orig, Math.round((float) 0.5*orig.getHeight()), Math.round((float)0.5*orig.getWidth()), orig.getHeight()/2, orig.getWidth()/2, 1, 0, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.42*orig.getHeight()), Math.round((float)0.37*orig.getWidth()), Math.round((float)orig.getHeight()/2), Math.round((float)orig.getWidth()/2), 0.3, 0, row, col);
				orig.drawEllipse(orig, Math.round((float) 0.45*orig.getHeight()), Math.round((float)0.45*orig.getWidth()), orig.getHeight()/2, orig.getWidth()/2, 1, 0, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.44*orig.getHeight()), Math.round((float)0.44*orig.getWidth()), orig.getHeight()/2, orig.getWidth()/2, 0, 0, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.06*orig.getHeight()), Math.round((float)0.03*orig.getWidth()), Math.round((float)3*orig.getHeight()/5), Math.round((float)5*orig.getWidth()/7), 0.9, Math.PI/3, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.2*orig.getHeight()), Math.round((float)0.05*orig.getWidth()), Math.round((float)3*orig.getHeight()/5), Math.round((float)1*orig.getWidth()/3), 0.8, -Math.PI/3, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.2*orig.getHeight()), Math.round((float)0.2*orig.getWidth()), Math.round((float)orig.getHeight()/2), Math.round((float)orig.getWidth()/2), 0.1, 0, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.02*orig.getHeight()), Math.round((float)0.03*orig.getWidth()), Math.round((float)5*orig.getHeight()/7), Math.round((float)3*orig.getWidth()/7), 0, Math.PI/5, row, col);
//				orig.drawEllipse(orig, Math.round((float) 0.03*orig.getHeight()), Math.round((float)0.04*orig.getWidth()), Math.round((float)4*orig.getHeight()/5), Math.round((float)4*orig.getWidth()/7), 1, Math.PI/4, row, col);
											
			}
		}
				
		return;
	}
	
	public void drawEllipse(Phantom orig, int a, int b, double midPointx, double midPointy, double intensity, double theta, int row, int col) {
		double xTransRot = Math.cos(theta) * (col-midPointx) - Math.sin(theta) * (row - midPointy);
		double yTransRot = Math.sin(theta) * (col-midPointx) + Math.cos(theta) * (row - midPointy);
		
		if ((Math.pow(xTransRot,2)/(a*a)) + (Math.pow(yTransRot,2)/(b*b)) <= 1.0) {
			orig.putPixelValue(row, col, intensity);
		}
	}


}
