package TruncationPolynomial;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid1D;
import edu.stanford.rsl.conrad.data.numeric.Grid1DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.InterpolationOperators;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;
public class BackProjection extends Grid2D {
	Phantom phantom;
	public BackProjection(int width, int height) {
		super(width, height);
		this.phantom = new Phantom(width, height);
	}
	public static void main(String args[]) {
		new ImageJ();
		// bild laden
		BackProjection bp = new BackProjection(256, 256);
		bp.phantom.show();
		//sinogram berechnen
		Phantom sinogram = Phantom.getSinogram(bp.phantom);
//		sinogram.show();
		// reconstruct sinogram
		Phantom reconstructed = backproject(sinogram);
//		 reconstructed.show();
		// filter sinogram mit ramp
		Phantom filteredSinogram = filter(sinogram);
//		filteredSinogram.show();
		// reconstruct ramp filtered sinogram
		Phantom reconstructedFiltered = backproject(filteredSinogram);
//		reconstructedFiltered.show();
		// filter sinogram with ramlac
		Phantom ramlacSinogram = ramlac(sinogram);
//		ramlacSinogram.show();
		// reconstruct ramlac filtered sinogram
		Phantom reconstructedRamlac = backproject(ramlacSinogram);
		reconstructedRamlac.show();
//		NumericPointwiseOperators.subtractedBy(reconstructedFiltered, reconstructedRamlac).show();
	}
	
	public static Phantom filter(Phantom sinogram) {
		Grid1DComplex[] sinogramTransform = new Grid1DComplex[(sinogram.getHeight())];
		Grid1D[] filteredSinogram = new Grid1D[(sinogram.getHeight())];
		for (int row = 0; row < sinogram.getHeight(); row++) {
			sinogramTransform[row] = new Grid1DComplex(sinogram.getSubGrid(row));
			sinogramTransform[row].transformForward();
			double deltaS = 1.0/sinogramTransform[row].getSize()[0];
			for (int w = 0; w < sinogramTransform[row].getSize()[0]; w++) {
				if(w < sinogramTransform[row].getSize()[0]/2) {
					sinogramTransform[row].setRealAtIndex(w, (float) (sinogramTransform[row].getRealAtIndex(w) * (w * deltaS)));
					sinogramTransform[row].setImagAtIndex(w, (float) (sinogramTransform[row].getImagAtIndex(w) * (w * deltaS)));
				} else {
					sinogramTransform[row].setRealAtIndex(w, (float) (sinogramTransform[row].getRealAtIndex(w) * (sinogramTransform[row].getSize()[0] - w) * deltaS));
					sinogramTransform[row].setImagAtIndex(w, (float) (sinogramTransform[row].getImagAtIndex(w) * (sinogramTransform[row].getSize()[0] - w) * deltaS));
				}
			}
			sinogramTransform[row].transformInverse();
			filteredSinogram[row] = sinogramTransform[row].getRealSubGrid(0,sinogram.getWidth());
		}
		Phantom result = new Phantom(sinogram.getWidth(), sinogram.getHeight());
//		System.out.println(result.getHeight() + "/" + result.getWidth());
		for (int row = 0; row < result.getHeight(); row++) {
			for (int col = 0; col < result.getWidth(); col++) {
				result.setAtIndex(col, row, filteredSinogram[row].getAtIndex(col));
			}
		}
		return result;
	}
	
	public static Phantom backproject(Phantom sinogram) {
		
		int a = sinogram.getWidth();//256;//Math.round((float)(sinogram.getWidth()/Math.sqrt(2)));
		Phantom reconstructed = new Phantom(a,a);
		
		for (int y = 0; y < reconstructed.getHeight(); y++) {
			for (int x = 0; x < reconstructed.getWidth(); x++) {
				
				for (int thetaGrad = 0; thetaGrad < sinogram.getHeight(); thetaGrad++) {
					double theta = (thetaGrad)* Math.PI/sinogram.getHeight();
					double s = (x-reconstructed.getWidth()/2) * Math.cos(theta) + (y-reconstructed.getHeight()/2) * Math.sin(theta);
					// float interpValue =sinogram.getAtIndex((int)(s+sinogram.getWidth()/2), thetaGrad);
					float interpValue = InterpolationOperators.interpolateLinear(sinogram, s + sinogram.getWidth()/2, thetaGrad);
					reconstructed.addAtIndex(x, y, interpValue);
				}
			}
		}
		return reconstructed;
	}
	
	
	public static Phantom ramlac(Phantom sinogram) {
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
		// ramlacFilter.getMagSubGrid(0, ramlacFilter.getSize()[0]).show();
		for (int row = 0; row < sinogram.getHeight(); row++) {
			sinogramTransform[row] = new Grid1DComplex(sinogram.getSubGrid(row));
			sinogramTransform[row].transformForward();
			for (int i = 0; i < ramlacFilter.getSize()[0]; i++) {
				sinogramTransform[row].multiplyAtIndex(i, ramlacFilter.getAtIndex(i));
			}
			// double deltaS = 1.0/sinogramTransform[row].getSize()[0];
			//
			// for (int w = 0; w < sinogramTransform[row].getSize()[0]; w++) {
			// if(w < sinogramTransform[row].getSize()[0]/2) {
			// sinogramTransform[row].setRealAtIndex(w, (float) (sinogramTransform[row].getRealAtIndex(w) * (w * deltaS)));
			// sinogramTransform[row].setImagAtIndex(w, (float) (sinogramTransform[row].getImagAtIndex(w) * (w * deltaS)));
			// } else {
			// sinogramTransform[row].setRealAtIndex(w, (float) (sinogramTransform[row].getRealAtIndex(w) * (sinogramTransform[row].getSize()[0] - w) * deltaS));
			// sinogramTransform[row].setImagAtIndex(w, (float) (sinogramTransform[row].getImagAtIndex(w) * (sinogramTransform[row].getSize()[0] - w) * deltaS));
			// }
			// }
			sinogramTransform[row].transformInverse();
			filteredSinogram[row] = sinogramTransform[row].getRealSubGrid(0,sinogram.getWidth());
		}
		Phantom result = new Phantom(sinogram.getWidth(), sinogram.getHeight());
//		System.out.println(result.getHeight() + "/" + result.getWidth());
		for (int row = 0; row < result.getHeight(); row++) {
			for (int col = 0; col < result.getWidth(); col++) {
				result.setAtIndex(col, row, filteredSinogram[row].getAtIndex(col));
			}
		}
		return result;
	}
}