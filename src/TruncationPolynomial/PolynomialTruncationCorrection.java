package TruncationPolynomial;

import Jama.Matrix;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import ij.ImageJ;

public class PolynomialTruncationCorrection {

	public static void main(String[] args) {

		Matrix a = new Matrix(3, 5);
		System.out.println(a.getColumnDimension());

		new ImageJ();

		Phantom phantom = new Phantom(256, 256);
		phantom.show("original");

		int DWeit = 10000;
		Phantom fanogram = FanBeamProjection.getFanogram(phantom, DWeit, 360);
		Phantom sinogram = FanBeamProjection.rebin(fanogram, DWeit);
		sinogram.show("Sinogram");		

		int DNah = 130;
		Phantom truncatedFanogram = FanBeamProjection.getFanogram(phantom, DNah, 360);
		Phantom truncatedSinogram = FanBeamProjection.rebin(truncatedFanogram, DNah);
		truncatedSinogram.show("truncated Sinogram");

		Phantom correctedSinogram = correctTruncationPolynomial(truncatedSinogram);
		correctedSinogram.show("corrected Sinogram");
		System.out.println("Done.");
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
		//		System.out.println("Anfang: " + anfang);
		//		System.out.println("Ende: " + ende);

		int N_s = ende - anfang + 1;
		//		System.out.println("N_s: " + N_s);
		int N_ext = (int) Math.round(N_s/15.0);
		//		System.out.println("N_ext: " + N_ext);

		Phantom correctedSinogram = new Phantom(N_s + 2 * N_ext, truncatedSinogram.getHeight(), false);

		for (int row = 0; row < correctedSinogram.getHeight(); row++) {
			for (int col = 0; col < N_ext; col++) {
				correctedSinogram.setAtIndex(col, row, 0.0f);
				correctedSinogram.setAtIndex(correctedSinogram.getWidth() - col - 1, row, 0.0f);
			}
		}
		for (int row = 0; row < correctedSinogram.getHeight(); row++) {
			for (int col = N_ext; col < N_ext + N_s; col++) {		
				correctedSinogram.setAtIndex(col, row, truncatedSinogram.getAtIndex(anfang+col-N_ext, row));
			}
		}

		int N_fit = (int) Math.round(N_s/20.0);

		for (int sinoRow = 0; sinoRow < correctedSinogram.getHeight(); sinoRow++) {
			Matrix n = new Matrix(3, 1);
			Matrix X = new Matrix(2*N_fit, 3);
			Matrix Y = new Matrix(2*N_fit, 1);

			for (int row = 0; row < N_fit; row++) {
				for (int col = 0; col < X.getColumnDimension(); col++) {

					X.set(row, col, Math.pow(N_ext+row, X.getColumnDimension()-1-col));
					X.set(X.getRowDimension()-1-row, col, Math.pow(N_ext+N_s-row, X.getColumnDimension()-1-col));

				}

				Y.set(row, 0, correctedSinogram.getAtIndex(N_ext+row, sinoRow));
				Y.set(X.getRowDimension()-1-row, 0, correctedSinogram.getAtIndex(N_ext+N_s-row, sinoRow));
			}

			
			
		}
		return correctedSinogram;


	}
}

