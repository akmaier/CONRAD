package edu.stanford.rsl.tutorial.phantoms;

import ij.IJ;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.io.IOException;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.io.FileProjectionSource;
import edu.stanford.rsl.conrad.pipeline.ProjectionSource;

public class FilePhantom extends Phantom {

	public FilePhantom(int imgSzXGU, int imgSzYGU, String filename) {
		super(imgSzXGU, imgSzYGU, filename);
		FloatProcessor fip =null;
		try {
			ProjectionSource pSource = FileProjectionSource.openProjectionStream(filename);
			Grid2D iProcessor = pSource.getNextProjection();
			fip =  new FloatProcessor(iProcessor.getWidth(), iProcessor.getHeight(), iProcessor.getBuffer(), null);

		} catch (IOException e1) {
			IJ.open(filename);
			fip = IJ.getImage().getProcessor().toFloat(0, null);
		}
		if (fip.getWidth()!= imgSzXGU || fip.getHeight()!=imgSzYGU){
			ImageProcessor resize = fip.resize(imgSzXGU, imgSzXGU);
			for (int j=0;j<imgSzYGU; j++){
				for (int i=0;i<imgSzXGU; i++){
					setAtIndex(i, j, resize.getf(i, j));
				}
			}
		}
		else
		{
			for (int j=0;j<imgSzYGU; j++){
				for (int i=0;i<imgSzXGU; i++){
					setAtIndex(i, j, fip.getf(i,j));
				}
			}
		}
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/