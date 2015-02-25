package TruncationPolynomial;

import ij.ImageJ;

public class PolynomialTruncationCorrection {
	
	public static void main(String[] args) {
		
		new ImageJ();
		
		Phantom phantom = new Phantom(256, 256);
		phantom.show("original");
		
		int DWeit = 10000;
		Phantom fanogramWeitWeg = FanBeamProjection.getFanogram(phantom, DWeit, 360);
		fanogramWeitWeg.show("weit weg");
		Phantom sinogramWeitWeg = FanBeamProjection.rebin(fanogramWeitWeg, DWeit);
		sinogramWeitWeg.show("rebinned Sinogram weit weg");
		
		int DNah = 100;
		Phantom fanogramNahDran = FanBeamProjection.getFanogram(phantom, DNah, 360);
		fanogramNahDran.show("nah dran");
		
		Phantom sinogramNahDran = FanBeamProjection.rebin(fanogramNahDran, DNah);
		sinogramNahDran.show("rebinned Sinogram nah dran");
		
		
		
	}
	
	
	

}
