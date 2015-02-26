package TruncationPolynomial;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.ImageJ;

public class PolynomialTruncationCorrection {

	public static void main(String[] args) {

		new ImageJ();

		Phantom phantom = new Phantom(256, 256);
		phantom.show("original");

		int DWeit = 10000;
		Phantom fanogram = FanBeamProjection.getFanogram(phantom, DWeit, 360);
		Phantom sinogram = FanBeamProjection.rebin(fanogram, DWeit);
//		Phantom sinogram = Phantom.getSinogram(phantom);
		sinogram.show("Sinogram");		

//		int DNah = 245;
//		Phantom truncatedFanogram = FanBeamProjection.getFanogram(phantom, DNah, 360);
//		Phantom truncatedSinogram = FanBeamProjection.rebin(truncatedFanogram, DNah);
//		truncatedSinogram.show("truncated Sinogram");
		Phantom truncatedSinogram = getTruncatedSinogram(sinogram, 20);
		truncatedSinogram.show("truncated Sinogram");

		Phantom correctedSinogram = correctTruncationPolynomial(truncatedSinogram);
		correctedSinogram.show("corrected Sinogram");

		Phantom reconstructedSinogram = BackProjection.backproject(sinogram);
		Phantom reconstructedTruncatedSinogram = BackProjection.backproject(truncatedSinogram);
		Phantom reconstructedCorrectedSinogram = BackProjection.backproject(correctedSinogram);
		
//		reconstructedSinogram.show("original reconstructed");
//		reconstructedTruncatedSinogram.show("truncated reconstructed");
//		reconstructedCorrectedSinogram.show("corrected reconstructed");
		
		
	}

	private static Phantom getTruncatedSinogram(Phantom sinogram, int truncationWidth) {
		
		Phantom truncatedSinogram = new Phantom(sinogram);
		
		for (int row = 0; row < sinogram.getHeight(); row++) {
			for (int col = 0; col < truncationWidth; col++) {
				
				truncatedSinogram.setAtIndex(col, row, 0.0f);
				truncatedSinogram.setAtIndex(truncatedSinogram.getWidth()-1-col, row, 0.0f);
				
			}
		}

		
		return truncatedSinogram;
	}

	private static Phantom correctTruncationPolynomial(Phantom truncatedSinogram) {

		int anfang = 0;
		int ende = 0;
		for (int i = 0; i < truncatedSinogram.getWidth(); i++) {
			if (anfang == 0
					&& truncatedSinogram.getAtIndex(i, 0) != 0.0f
					&& !Double.isNaN(truncatedSinogram.getAtIndex(i, 0))
					) {
				anfang = i;
			}
			if (ende == 0
					&& truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-i, 0) != 0.0f
					&& !Double.isNaN(truncatedSinogram.getAtIndex(truncatedSinogram.getWidth()-i, 0))
					) {
				ende = truncatedSinogram.getWidth()-i;
			}
		}

		int N_s = ende - anfang + 1;
		int N_ext = (int) Math.round(N_s/15.0);

		Phantom correctedSinogram = new Phantom(N_s + 2 * N_ext, truncatedSinogram.getHeight(), false);
		
		for (int row = 0; row < correctedSinogram.getHeight(); row++) {
			for (int col = N_ext; col < N_ext + N_s; col++) {		
				correctedSinogram.setAtIndex(col, row, truncatedSinogram.getAtIndex(anfang+col-N_ext, row));
			}
		}
		for (int row = 0; row < correctedSinogram.getHeight(); row++) {
			for (int col = 0; col < N_ext; col++) {
				correctedSinogram.setAtIndex(col, row, 0.0f);
				correctedSinogram.setAtIndex(correctedSinogram.getWidth() - col - 1, row, 0.0f);
			}
		}

		
		correctedSinogram.show("before correction");

		int N_fit = (int) Math.round(N_s/20.0);

		for (int sinoRow = 0; sinoRow < correctedSinogram.getHeight(); sinoRow++) {

			int polynomgrad = 5;
			
			Matrix X = new Matrix(2*N_fit, polynomgrad);
			Matrix Y = new Matrix(2*N_fit, 1);

			for (int row = 0; row < N_fit; row++) {
				for (int col = 0; col < X.getColumnDimension(); col++) {

					X.set(row, col, Math.pow(N_ext+row, X.getColumnDimension()-1-col));
					X.set(X.getRowDimension()-1-row, col, Math.pow(N_ext+N_s-row, X.getColumnDimension()-1-col));

				}

				Y.set(row, 0, correctedSinogram.getAtIndex(N_ext+row, sinoRow));
				Y.set(X.getRowDimension()-1-row, 0, correctedSinogram.getAtIndex(N_ext+N_s-row, sinoRow));
			}
			
			Matrix n = X.solve(Y);
			System.out.println(n.getRowDimension());

			for (int sinoCol = 0; sinoCol < N_ext; sinoCol++) {
				
				Matrix xLeft = new Matrix(1, n.getRowDimension());
				Matrix xRight = new Matrix(1, n.getRowDimension());
				
				for (int pow = 0; pow < n.getRowDimension(); pow++) {
					xLeft.set(0, pow, Math.pow(sinoCol, n.getRowDimension()-1-pow));
					xRight.set(0, pow, Math.pow(correctedSinogram.getWidth()-1-sinoCol, n.getRowDimension()-1-pow));
				}
				
				Matrix yLeft = xLeft.times(n);
				Matrix yRight = xRight.times(n);
				
//				correctedSinogram.show("before correction");
			
				correctedSinogram.setAtIndex(sinoCol, sinoRow, (float) yLeft.get(0, 0));
				correctedSinogram.setAtIndex(correctedSinogram.getWidth()-1-sinoCol, sinoRow, (float) yRight.get(0, 0));
				
//				correctedSinogram.show(Integer.toString(sinoCol));
			}
//			correctedSinogram.show(Integer.toString(sinoRow));
			
			
			
		}
		
		return correctedSinogram;


	}
}

