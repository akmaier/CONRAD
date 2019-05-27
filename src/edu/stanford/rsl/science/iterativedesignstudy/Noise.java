package edu.stanford.rsl.science.iterativedesignstudy;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;


/**
 * 
 * Allows to add different types of noise
 * 
 * 
 * 
 * @author Karoline Kallis, Anja Jaeger, Tilmann Huebner
 * 
 */
public class Noise {
	public static Grid2D addRandomNoise(Grid2D clearImage, float percentage) {
		int height = clearImage.getHeight();
		int width = clearImage.getWidth();
		float amount = percentage * width * height / 100;
		Grid2D noisyImage = (Grid2D) clearImage.clone();
		for (int i = 0; i < amount; i++) {
			int xSalt = (int) (Math.random() * width);
			int ySalt = (int) (Math.random() * height);

			int xPepper = (int) (Math.random() * width);
			int yPepper = (int) (Math.random() * height);

			noisyImage.setAtIndex(xSalt, ySalt,
					NumericPointwiseOperators.max(clearImage));
			noisyImage.setAtIndex(xPepper, yPepper,
					NumericPointwiseOperators.min(clearImage));
		}

		return noisyImage;
	}

	public static Grid2D addNoise(Grid2D originalSinogram) {
		// Noise
		Grid2D noise = new Grid2D(originalSinogram);
		NumericPointwiseOperators.fill(noise, (float) 0.0);

		float max = NumericPointwiseOperators.max(originalSinogram);
		float min = NumericPointwiseOperators.min(originalSinogram);

		for (int i = 0; i < noise.getWidth(); i++) {
			for (int j = 0; j < noise.getHeight(); j++) {
				noise.addAtIndex(i, j, (float) (min + Math.random() / 5
						* (max - min)));
			}
		}
		NumericPointwiseOperators.addBy(originalSinogram, noise);
		return originalSinogram;
	}
}

/*
 * Copyright (C) 2010-2014 Karoline Kallis, Anja Jaeger, Tilmann Huebner CONRAD
 * is developed as an Open Source project under the GNU General Public License
 * (GPL).
 */
