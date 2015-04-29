package edu.stanford.rsl.tutorial.truncation;

import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.tutorial.parallel.ParallelBackprojector2D;
import edu.stanford.rsl.tutorial.parallel.ParallelProjector2D;
import edu.stanford.rsl.tutorial.phantoms.SimpleGridsForTruncationCorrection;
import ij.ImageJ;

//Name, was macht Klasse

public class TruncationCorrection {

	public TruncationCorrection(){}
	
	public static void main(String[] args) {

		new ImageJ();

		TruncationCorrection T = new TruncationCorrection();
		
		// create Phantom
		int imageSize = 200;
		double maxS = 200.0;
		double deltaS = 1.0;
		int numTheta = 180;
		int truncationSize = 40;
		SimpleGridsForTruncationCorrection cylinder = new SimpleGridsForTruncationCorrection(imageSize, imageSize, 2);
		cylinder.show("original image");

		// get Sinogram
		ParallelProjector2D projector2d = new ParallelProjector2D(Math.PI, Math.PI/(numTheta-1), maxS, deltaS);
		Grid2D sinogram = projector2d.projectRayDrivenCL(cylinder);
		sinogram.show("original sinogram");

		// apply ramlak filter to sinogram 
		Grid2D origSino = new Grid2D(sinogram);		
		T.ramlak(origSino);		
		origSino.show("filtered original sinogram");
		
		
		// backproject original
		ParallelBackprojector2D backprojector = new ParallelBackprojector2D(imageSize, imageSize, 1, 1);
		Grid2D backproj = backprojector.backprojectRayDriven(origSino);
		backproj.show("filtered original backprojected");
		

		// truncate Sinogram
		Grid2D truncatedSinogram = T.getTruncatedSinogram(sinogram, truncationSize);
		truncatedSinogram.show("Truncated Sinogram");
		// backproject truncated sinogram
//		Grid2D truncBackproj = backprojector.backprojectRayDriven(truncatedSinogram);
//		truncBackproj.show("truncated backprojected");
		
		Grid2D correctedSinogram = T.polynomialTruncationCorrection(truncatedSinogram);
		correctedSinogram.setSpacing(sinogram.getSpacing());
		correctedSinogram.show("corrected Sinogram");
		
		Grid2D broaderCorrectedSinogram = T.addBlackOnSides(correctedSinogram, sinogram);
		broaderCorrectedSinogram.setSpacing(sinogram.getSpacing());
		broaderCorrectedSinogram.show("corrected sinogram, black added on sides");
		
		T.ramlak(broaderCorrectedSinogram);
//		RamLakKernel ramLak = new RamLakKernel((int) maxS, deltaS);
////		
//		for (int theta = 0; theta < origSino.getHeight(); ++theta) {
//			ramLak.applyToGrid(broaderCorrectedSinogram.getSubGrid(theta));
//		}
			
		broaderCorrectedSinogram.show("broader Corrected after RamLak");
		broaderCorrectedSinogram.setSpacing(sinogram.getSpacing());
		
		ParallelBackprojector2D backprojector2 = new ParallelBackprojector2D(imageSize, imageSize, 1, 1);
		Grid2D correctedBackproj = backprojector2.backprojectRayDriven(broaderCorrectedSinogram);
		correctedBackproj.show("filtered corrected Reconstructed");

	}

	// correct truncation with a polynomial in each row
	private Grid2D polynomialTruncationCorrection(Grid2D truncatedSinogram) {
		
		// get L for each angle
		int[] l = getL(truncatedSinogram);
		int maxL = max(l);

		// get derivative of truncated Sinogram
		Grid2D firstDerivativeTruncatedSinogram = derive(truncatedSinogram);

		// get width of the truncated sinogram
		int truncatedSinogramWidth = getSinogramWidth(truncatedSinogram);

		int R = truncatedSinogramWidth/2;		
		Grid2D correctedSinogram = new Grid2D(2*maxL, truncatedSinogram.getHeight());

		int posDerivative = (int) Math.floor(0.5f*(truncatedSinogram.getWidth()-truncatedSinogramWidth));

		// correct Sinogram on right and left side
		for (int sinoRow = 0; sinoRow < correctedSinogram.getHeight(); sinoRow++) {
			
			//get L, p', p for computation of a, b, c
			double L = l[sinoRow];

			double pPrime_left = firstDerivativeTruncatedSinogram.getAtIndex(posDerivative, sinoRow);
			double p_left = truncatedSinogram.getAtIndex(posDerivative, sinoRow);

			double pPrime_right = firstDerivativeTruncatedSinogram.getAtIndex(firstDerivativeTruncatedSinogram.getWidth()-posDerivative-1, sinoRow);
			double p_right = truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-posDerivative-1, sinoRow);

			double a_right = (p_right*p_right - 2*p_right*pPrime_right*(R-L))/(3*R*R-L*L-2*R*L);
			double b_right = 2*p_right*pPrime_right + 2*a_right*R;
			double c_right = -a_right*L*L - b_right*L;
			
			double a_left = (p_left*p_left - 2*p_left*pPrime_left*((-R)-(-L)))/(3*(-R)*(-R)-(-L)*(-L)-2*(-R)*(-L));
			double b_left = 2*p_left*pPrime_left + 2*a_left*(-R);
			double c_left = -a_left*(-L)*(-L) - b_left*(-L); 

			// add polynomial on left and right side
			for (int col = 0; col < L-truncatedSinogramWidth/2; col++) {
				correctedSinogram.setAtIndex(maxL-R-col, sinoRow, (float) Math.sqrt(a_left*(-R-col)*(-R-col)+b_left*(-R-col)+c_left));
				correctedSinogram.setAtIndex(maxL+R+col, sinoRow, (float) Math.sqrt(a_right*(R+col)*(R+col)+b_right*(R+col)+c_right));
			}
			// take original values in between
			for (int col = 0; col < truncatedSinogramWidth; col++) {
				correctedSinogram.setAtIndex(maxL-truncatedSinogramWidth/2 + col, sinoRow, truncatedSinogram.getAtIndex(posDerivative+col, sinoRow));
			}


		}

		return correctedSinogram;
	}
	
	// truncate sinogram: add zero values starting at left and right side
	private Grid2D getTruncatedSinogram(Grid2D sinogram, int truncationWidth) {

		Grid2D truncatedSinogram = new Grid2D(sinogram);
		for (int row = 0; row < sinogram.getHeight(); row++) {
			for (int col = 0; col < truncationWidth; col++) {
				truncatedSinogram.setAtIndex(col, row, 0.0f);
				truncatedSinogram.setAtIndex(truncatedSinogram.getWidth()-1-col, row, 0.0f);
			}
		}


		return truncatedSinogram;
	}

	// add black on the sides of the corrected sinogram to acheive the same width as in the original
	private Grid2D addBlackOnSides(Grid2D correctedSinogram, Grid2D originalSinogram) {
		
//		Grid2D result = new Grid2D(originalSinogram.getWidth(), correctedSinogram.getHeight());

		Grid2D result = new Grid2D(correctedSinogram.getWidth() + 40, correctedSinogram.getHeight());
		
		for (int row = 0; row < result.getHeight(); row ++) {
			for (int col = 0; col < result.getWidth(); col++) {
				result.setAtIndex(col, row, 0.0f);
			}
		}
		
		int begin = (result.getWidth()-correctedSinogram.getWidth())/2;
		for (int row = 0; row < result.getHeight(); row++) {
			for (int col = 0; col < correctedSinogram.getWidth(); col++) {
				result.setAtIndex(col+begin, row, correctedSinogram.getAtIndex(col, row));
			}
		}
		
		return result;
	}

	// get actual width of sinogram (from first to last non-zero value)
	private int getSinogramWidth(Grid2D truncatedSinogram) {
		int start = truncatedSinogram.getWidth();
		int end = 0;
		for (int row = 0; row < truncatedSinogram.getHeight(); row++) {
			boolean rowStartFound = false;
			boolean rowEndFound = false;
			for (int col = 0; col < truncatedSinogram.getWidth(); col++) {
				if (rowStartFound == false
						&& truncatedSinogram.getAtIndex(col, row) != 0.0f
						&& !Double.isNaN(truncatedSinogram.getAtIndex(col, row))
						) {
					if (col < start) {
						start = col;
					}
					rowStartFound = true;
				}
				if (rowEndFound == false
						&& truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-1-col, row) != 0.0f
						&& !Double.isNaN(truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-col, row))
						){
					if (truncatedSinogram.getWidth()-col > end) {
						end = truncatedSinogram.getWidth()-col;
					}
					rowEndFound = true;
				}
				if (rowStartFound && rowEndFound) break;
			}
		}
//		int start = 0;
//		int end = 0;
//		for (int i = 0; i < truncatedSinogram.getWidth(); i++) {
//			if (start == 0
//					&& truncatedSinogram.getAtIndex(i, 0) != 0.0f
//					&& !Double.isNaN(truncatedSinogram.getAtIndex(i, 0))
//					) {
//				start = i;
//			}
//			if (end == 0
//					&& truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-i, 0) != 0.0f
//					&& !Double.isNaN(truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-i, 0))
//					) {
//				end = truncatedSinogram.getWidth()-i;
//			}
//		}
		return end-start;
	}

	private Grid2D derive(Grid2D sinogram) {
		Grid2D firstDerivative = new Grid2D(sinogram.getWidth()-1, sinogram.getHeight());
		for (int row = 0; row < sinogram.getHeight(); row++) {
			for (int col = 0; col < firstDerivative.getWidth(); col++) {
				firstDerivative.setAtIndex(col, row, sinogram.getAtIndex(col+1, row) - sinogram.getAtIndex(col, row));
			}
		}
		return firstDerivative;
	}

	private int max(int[] l) {
		int curr_max = 0;
		for (int i = 0; i < l.length; i++) {
			if (l[i] > curr_max) {
				curr_max = l[i];
			}
		}
		return curr_max;
	}

	// get width L of polynomial for each row
	private int[] getL(Grid2D sinogram) {

//		// calculate average mu
//		double muSum = 0.0f;
//		double nonNullEntries = 0.0f;
//		for (int col = 0; col < sinogram.getWidth(); col++) {
//			for (int row = 0; row < sinogram.getHeight(); row++) {
//				muSum += sinogram.getAtIndex(col, row);
//				if (sinogram.getAtIndex(col, row) != 0.0) {
//					nonNullEntries++;
//				}
//			}
//		}
//		double mu_average = 1.0f;// muSum/(sinogram.getWidth()*sinogram.getHeight());//(nonNullEntries);
//		// 1.0f fuer wasserzylinder Mitte
//		// 1.7f fuer mickey
//		// 0.5f fuer wasserzylinder verschoben
//		
//		
//		// calculate l-vector
//		double[] l_double = new double[sinogram.getHeight()];
//		int[] l = new int[sinogram.getHeight()];
//		for (int theta = 0; theta < sinogram.getHeight(); theta++) {
//
//			double maximum = 0.0f;
//			for (int thetaPrime = theta-90; thetaPrime < theta+90; thetaPrime++) {
//				if (thetaPrime < 0) {
//					thetaPrime +=360;
//					for (int s = 0;  s < sinogram.getWidth(); s++) {
//						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime)) / mu_average;
//						if (val > maximum) {
//							maximum = val;
//						}
//					}
//					thetaPrime -= 360;
//				} else if (thetaPrime >= 360) {
//					thetaPrime -=360;
//					for (int s = 0;  s < sinogram.getWidth(); s++) {
//						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime)) / mu_average;
//						if (val > maximum) {
//							maximum = val;
//						}
//					}
//					thetaPrime += 360;
//				} else {
//					for (int s = 0;  s < sinogram.getWidth(); s++) {
//						double val = sinogram.getAtIndex(thetaPrime, s) * Math.sin(Math.abs(theta - thetaPrime)) / mu_average;
//						if (val > maximum) {
//							maximum = val;
//						}
//					}
//				}
//
//			}
//			l_double[theta] = maximum*0.5;
//			l[theta] = (int) Math.round(maximum*0.5);
//		}			
		
		
//		int[] l = new int[sinogram.getHeight()];
//		
//		for (int i = 0; i < 75; i++) {
//			l[i] = 0;
//		}
//		for (int i = 75; i < 180; i++) {
//			l[i] = (int) Math.round(61 + (i-75) * 0.18);
//		}
//		for (int i = 180; i < 285; i++) {
//			l[i] = l[i] = (int) Math.round(80- (i-180) * 0.18);;
//		}
//		for (int i = 286; i < sinogram.getHeight(); i++) {
//			l[i] = 0;
//		}
		
		double deltaTheta = 180.0f/sinogram.getHeight();
		
		int[] L = new int[sinogram.getHeight()];
		
		double mu_average = 1.0f;
		
		for (int row = 0; row < sinogram.getHeight(); row++) {
			double theta = row * deltaTheta;
			double thetaRad = theta * (Math.PI/180);
			
			double max = 0.0f;
			
			for (int rowPrime = (int) (row - 90/deltaTheta); rowPrime < (int) (row + 90/deltaTheta); rowPrime++) {
				
				boolean changedThetaPrimeMin = false;
				boolean changedThetaPrimePlus = false;
				if (rowPrime >= sinogram.getHeight()) {
					rowPrime -= sinogram.getHeight();
					changedThetaPrimeMin = true;
				} else if (rowPrime < 0) {
					rowPrime += sinogram.getHeight();
					changedThetaPrimePlus = true;
				}

				
				double thetaPrime = rowPrime * deltaTheta;
				double thetaPrimeRad = thetaPrime * (Math.PI/180);
				
				for (int s = 0; s < sinogram.getWidth(); s++) {
					
					
					double curr_L = sinogram.getAtIndex(s, rowPrime) * Math.sin(thetaRad - thetaPrimeRad) / mu_average;
					
					if (Math.abs(curr_L) > max) {
						max = Math.abs(curr_L);
					}					
				}
				
				if (changedThetaPrimeMin) {
					rowPrime += sinogram.getHeight();
				} else if (changedThetaPrimePlus) {
					rowPrime -= sinogram.getHeight();
				}
				
			}
			
			L[row] = (int) max/2;
			
		}
		
		
//		double[] x = new double[sinogram.getHeight()];
//		double[] Ldouble = new double[L.length];
//		for (int i = 0; i < x.length; i++) {
//			x[i] = i;
//			Ldouble[i] = L[i];
//		}
//
//		Plot2DPanel panel = new Plot2DPanel();
//		panel.addLinePlot("Line", x, Ldouble);
//		
//		JFrame  frame= new JFrame("Histogram");
//		frame.setContentPane(panel);
//		frame.setSize(500, 600);
//		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//		frame.setVisible(true);


		return L;
	}
	
	public void ramlak(Grid2D sinogram) {
		
		Grid1DComplex[] sinogramTransform = new Grid1DComplex[(sinogram.getHeight())];
		Grid1D[] filteredSinogram = new Grid1D[(sinogram.getHeight())];
		Grid1DComplex ramlacFilter = new Grid1DComplex(sinogram.getWidth());
		
		int middle = (int) ramlacFilter.getSize()[0]/2;
		ramlacFilter.setRealAtIndex(middle, 1.0f/4);
		
		for (int i = middle + 1; i < ramlacFilter.getSize()[0]; i+=2) {
			ramlacFilter.setRealAtIndex(i, (float) (-1.0f/((i-middle)*(i-middle)*Math.PI*Math.PI)));
			ramlacFilter.setImagAtIndex(i, 0);
		}
		
		for (int i = middle - 1; i >= 0; i-=2) {
			ramlacFilter.setRealAtIndex(i, (float) (-1.0f/((i-middle)*(i-middle)*Math.PI*Math.PI)));
			ramlacFilter.setImagAtIndex(i, 0);
		}
		
		ramlacFilter.transformForward();

		for (int row = 0; row < sinogram.getHeight(); row++) {
			sinogramTransform[row] = new Grid1DComplex(sinogram.getSubGrid(row));
			sinogramTransform[row].transformForward();
			for (int i = 0; i < ramlacFilter.getSize()[0]; i++) {
				sinogramTransform[row].multiplyAtIndex(i, ramlacFilter.getAtIndex(i));
			}
			sinogramTransform[row].transformInverse();
			filteredSinogram[row] = sinogramTransform[row].getRealSubGrid(0,sinogram.getWidth());
		}

		for (int row = 0; row < sinogram.getHeight(); row++) {
			for (int col = 0; col < sinogram.getWidth(); col++) {
				sinogram.setAtIndex(col, row, filteredSinogram[row].getAtIndex(col));
			}
		}
	}
}

