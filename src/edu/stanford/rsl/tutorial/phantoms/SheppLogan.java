package edu.stanford.rsl.tutorial.phantoms;

import ij.ImageJ;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;

public class SheppLogan extends Phantom{
	
	final private SimpleMatrix Ellipses; 
	
	
	public SheppLogan(int xy, boolean type) {
		super(xy,xy,"Shepp-Logan-Phantom");
		
		if (type)
			Ellipses = ShepOrig();
		else
			Ellipses = ShepMod();
		CreatePhantom();
	}
	
	
	public SheppLogan(int xy) {
		super(xy,xy,"Shepp-Logan-Phantom");
		Ellipses = ShepMod();
		CreatePhantom();
	}
	
	private void CreatePhantom()
	{
	
		// Iterate over all pixels and sum up intensity values of all corresponding ellipses
		double sizeX = (double)super.getSize()[0];
		double sizeY = (double)super.getSize()[1];
		
		for (int i=0; i < super.getSize()[0]; ++i)
		{
			double x = ((double)i-(sizeX-1)/2.0) / ((sizeX-1)/2.0);
			for (int j=0; j < super.getSize()[1]; ++j)
			{
				double y = ((double)j-(sizeY-1)/2.0) / ((sizeY-1)/2.0);
				super.setAtIndex(i, super.getSize()[1]-j-1, 0.f);
				for (int k=0; k < Ellipses.getRows(); ++k)
				{
					// Extract the ellipse properties here
					double xc = x - Ellipses.getElement(k, 3);
					double yc = y - Ellipses.getElement(k, 4);
					double phi = Ellipses.getElement(k, 5)*Math.PI/180.0;
					double cos = Math.cos(phi);
					double sin = Math.sin(phi);
					double asq = Ellipses.getElement(k, 1)*Ellipses.getElement(k, 1);
					double bsq = Ellipses.getElement(k, 2)*Ellipses.getElement(k, 2);
					double Val = Ellipses.getElement(k, 0);
					
					// Check if this pixel is part of the ellipse, if yes, add the given intensity value to it
					double help = Math.pow((xc*cos + yc*sin),2.0);
					double help2 = Math.pow((yc*cos - xc*sin),2.0);
					if ( help/asq + help2/bsq <= 1.0 )
						super.setAtIndex(i, super.getSize()[1]-j-1, super.getAtIndex(i, super.getSize()[1]-j-1) + (float)Val);
				}
				
			}
		}
	}
	
	private SimpleMatrix ShepOrig()
	{
		// Original Shepp-Logan Phantom according to:
		
		// Shepp, L. A., & Logan, B. F. (1974). 
		// The Fourier reconstruction of a head section-LA Shepp. 
		// IEEE Transactions on Nuclear Science, NS-21, 21�43.
		
		// One row describes properties for a single ellipse
		// Colum Values: A    a     b    x0    y0    phi
		
		SimpleMatrix Shep = new SimpleMatrix(10,6);
		
		Shep.setRowValue(0, new SimpleVector(new double[] {2.0, 0.69, 0.92, 0, 0, 0}));
		Shep.setRowValue(1, new SimpleVector(new double[] {-0.98, 0.6624, 0.8740, 0, -0.0184, 0}));
		Shep.setRowValue(2, new SimpleVector(new double[] {-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0}));
		Shep.setRowValue(3, new SimpleVector(new double[] {-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0}));
		Shep.setRowValue(4, new SimpleVector(new double[] {0.01, 0.2100, 0.2500, 0, 0.35, 0}));
		Shep.setRowValue(5, new SimpleVector(new double[] {0.01, 0.0460, 0.0460, 0, 0.1, 0}));
		Shep.setRowValue(6, new SimpleVector(new double[] {0.01, 0.0460, 0.0460, 0, -0.1, 0}));
		Shep.setRowValue(7, new SimpleVector(new double[] {0.01, 0.0460, 0.0230, -0.08, -0.605, 0}));
		Shep.setRowValue(8, new SimpleVector(new double[] {0.01, 0.0230, 0.0230, 0, -0.606, 0}));
		Shep.setRowValue(9, new SimpleVector(new double[] {0.01, 0.0230, 0.0460,  0.06, -0.605, 0}));
		return Shep;
	}
	
	private SimpleMatrix ShepMod()
	{
	// Modified (better contrast) Shepp-Logan Phantom according 
	// to P. A. Toft, "The Radon Transform, Theory and Implementation" 
	// (unpublished dissertation), p. 199.
	
	// One row describes properties for a single ellipse
	// Colum Values: A    a     b    x0    y0    phi
	
	SimpleMatrix Shep = new SimpleMatrix(10,6);
	
	Shep.setRowValue(0, new SimpleVector(new double[] {1.0, 0.69, 0.92, 0, 0, 0}));
	Shep.setRowValue(1, new SimpleVector(new double[] {-0.8, 0.6624, 0.8740, 0, -0.0184, 0}));
	Shep.setRowValue(2, new SimpleVector(new double[] {-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0}));
	Shep.setRowValue(3, new SimpleVector(new double[] {-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0}));
	Shep.setRowValue(4, new SimpleVector(new double[] {0.1, 0.2100, 0.2500, 0, 0.35, 0}));
	Shep.setRowValue(5, new SimpleVector(new double[] {0.1, 0.0460, 0.0460, 0, 0.1, 0}));
	Shep.setRowValue(6, new SimpleVector(new double[] {0.1, 0.0460, 0.0460, 0, -0.1, 0}));
	Shep.setRowValue(7, new SimpleVector(new double[] {0.1, 0.0460, 0.0230, -0.08, -0.605, 0}));
	Shep.setRowValue(8, new SimpleVector(new double[] {0.1, 0.0230, 0.0230, 0, -0.606, 0}));
	Shep.setRowValue(9, new SimpleVector(new double[] {0.1, 0.0230, 0.0460,  0.06, -0.605, 0}));
	return Shep;
	}
	
	
	public static void main (String [] args){
		new ImageJ();
		SheppLogan test = new SheppLogan(256);
		test.show("Shepp Logan");
		
		SheppLogan test2 = new SheppLogan(256,true);
		test2.show("Shepp Logan");
		
		}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/