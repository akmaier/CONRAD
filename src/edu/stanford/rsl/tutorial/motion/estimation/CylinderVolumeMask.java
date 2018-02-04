package edu.stanford.rsl.tutorial.motion.estimation;

import ij.ImageJ;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.Grid3D;
import edu.stanford.rsl.tutorial.phantoms.SimpleCubes3D;

/**
 * This class applies a cylinder mask to the reconstructed volume Grid in order
 * to get rid of errors outside the VOI
 * 
 * @author Marco Boegel
 * 
 */
public class CylinderVolumeMask {

	private Grid2D mask;
	private int sizeY;
	private int sizeX;

	public CylinderVolumeMask(int sizeX, int sizeY, int xcenter, int ycenter, double radius) {
		this.sizeX = sizeX;
		this.sizeY = sizeY;
		mask = new Grid2D(sizeX, sizeY);
		initialize(radius, sizeX, sizeY, xcenter, ycenter);
	}

	private void initialize(double radius, int sizeX, int sizeY, int xcenter, int ycenter) {

		for (int i = 0; i < sizeX; i++) {
			for (int j = 0; j < sizeY; j++) {
				if (Math.pow(i - xcenter, 2) + Math.pow(j - ycenter, 2) <= (radius * radius)) {
					mask.setAtIndex(i, j, 1.f);
				}
			}
		}
	}

	public void applyToGrid(Grid3D input) {
		int size = input.getSize()[2];
		for (int i = 0; i < size; i++)
			applyToGrid(input.getSubGrid(i));
	}

	/**
	 * Applies cylinder in a limited z-Range
	 * 
	 * @param input
	 *            input grid
	 * @param start
	 *            start z-coord in volume coordinates
	 * @param end
	 *            end z-coord in volume coordinates
	 * @param keepRest
	 *            if true leaves the rest of the volume as is, otherwise sets to
	 *            zero
	 */
	public void applyToGrid(Grid3D input, int start, int end, boolean keepRest) {
		int size = input.getSize()[2];

		for (int i = start; i < end; i++)
			applyToGrid(input.getSubGrid(i));

		if (keepRest == false) {
			for (int i = 0; i < start; i++) {
				for (int x = 0; x < sizeX; x++) {
					for (int y = 0; y < sizeY; y++) {

						input.getSubGrid(i).setAtIndex(x, y, 0);
					}
				}
			}
			for (int i = end; i < size; i++) {
				for (int x = 0; x < sizeX; x++) {
					for (int y = 0; y < sizeY; y++) {

						input.getSubGrid(i).setAtIndex(x, y, 0);
					}
				}
			}
		}
	}

	public void applyToGrid(Grid2D input) {

		for (int i = 0; i < sizeX; i++) {
			for (int j = 0; j < sizeY; j++) {
				input.multiplyAtIndex(i, j, mask.getAtIndex(i, j));
			}
		}

	}

	public static void main(String args[]) {
		CylinderVolumeMask m = new CylinderVolumeMask(200, 200, 100, 100, 100);
		Grid3D s = new SimpleCubes3D(200, 200, 300);
		new ImageJ();
		m.applyToGrid(s);

		s.show();
	}
}
/*
 * Copyright (C) 2010-2014 Marco Bögel
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/