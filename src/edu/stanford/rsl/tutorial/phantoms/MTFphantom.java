package edu.stanford.rsl.tutorial.phantoms;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.VisualizationUtil;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.PointRoi;

/**
 * Creates an MTF phantom with five high density beads in a homogeneous cylinder
 * 
 * @author Martin Berger
 * 
 */
public class MTFphantom extends Phantom {
	
	
	private PointRoi[] beadCoords = null; 
	
	/**
	 * The constructor takes two arguments to initialize the grid. The cylinder
	 * will be in the center and have a radius of r1% of the x-dimension. Thus
	 * we recommend a grid size that is uniform in both directions. The outer beads
	 * will be at a circle defined by radius of r2% of the r1 radius.
	 * 
	 * @param x Size in X direction
	 * @param y Size in Y direction
	 * @param r1 Cylinder radius as a fraction of the X size
	 * @param r2 Outer bead radius as a fraction of the cylinder radius
	 * @param cylinderDensity Density value for the cylinder
	 * @param beadDensity Density value for the beads
	 * 
	 */
	public MTFphantom(int x, int y, double r1, double r2, float cylinderDensity, float beadDensity) {
		super(x, y, "MTF Phantom - Bead Coordinates: (" + Integer.toString((x/2)-((int)Math.round(r1*r2*(double)x/2.0))) + "," + Integer.toString((y/2)) + "),(" +
														Integer.toString((x/2)) + "," + Integer.toString((y/2)) + "),(" +
														Integer.toString((x/2)+((int)Math.round(r1*r2*(double)x/2.0))) + "," + Integer.toString((y/2)) + "),(" +
														Integer.toString((x/2)) + "," + Integer.toString((y/2)+((int)Math.round(r1*r2*(double)x/2.0))) + "),(" +
														Integer.toString((x/2)) + "," + Integer.toString((y/2)-((int)Math.round(r1*r2*(double)x/2.0))) + ")");

		int xcenter = x / 2;
		int ycenter = y / 2;
		int h1 = (int)Math.round(r1*r2*(double)x/2.0);
		beadCoords = new PointRoi[] {new PointRoi(xcenter-h1,ycenter), 
									 new PointRoi(xcenter,ycenter), 
									 new PointRoi(xcenter+h1,ycenter),
									 new PointRoi(xcenter,ycenter+h1),
									 new PointRoi(xcenter,ycenter-h1)};
		
		// Create circle in the grid.
		double radius1 = r1 * (double)x/2.0;
		int beadRadius = (int)Math.round(r2*radius1);


		for (int i = -xcenter; i < -xcenter+x; i++) {
			for (int j = -ycenter; j < -ycenter+y; j++) {
				if (Math.pow(i, 2) + Math.pow(j, 2) <= (radius1 * radius1)) {
					if( (j == 0 && Math.abs(i) == beadRadius) || (i == 0 && Math.abs(j) == beadRadius) || (Math.pow(i,2) + Math.pow(j, 2) <= 0) )
					{
						super.setAtIndex(i+xcenter, j+ycenter, beadDensity);
					}
					else
					{
						super.setAtIndex(i+xcenter, j+ycenter, cylinderDensity);
					}
				}
				else
					super.setAtIndex(i+xcenter, j+ycenter, 0.f);
			}
		}
	}
	
	public PointRoi[] getBeadPointsROI()
	{
		return this.beadCoords;
	}
		
		public static void main(String [] args)
		{
			new ImageJ();
			MTFphantom test = new MTFphantom(1024, 1024, 0.95, 0.8, 1.f, 2.f);
			Grid2D phantom= new Grid2D(test);
			ImagePlus gi = VisualizationUtil.showGrid2D(phantom, test.getTitle());
			ImagePlus ip = gi.duplicate();
			ip.show();
			ip.getProcessor().setRoi(test.getBeadPointsROI()[0]);
		}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/