package edu.stanford.rsl.tutorial.modelObserver;



import java.util.Random;
import edu.stanford.rsl.conrad.data.numeric.Grid2DComplex;
import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericPointwiseOperators;

/**
 * This class provides all supporting function for Model_observer demo.
 * All original functions are written by Adam Wunderlich in MATLAB
 * Available online, url: https://github.com/DIDSR/IQmodelo  
 * @author Priyal Patel
 * @Supervisors: Frank Schebesch, Andreas Maier
 */
public class CreateImages extends Grid2D {

	private int height = 500, width = 500;
	private float sigscale = 2.5f; // signal scale parameter (Gaussian stdev)
	private int A = 10;
	
	public int getwidth() {
		return this.width;
	}
	
	public int getamplitude() {
		return this.A;
	}
	
	public int getheight() {
		return this.height;
	}
	
	public CreateImages() {
		super(null);
	}

	public CreateImages(int width, int height, double[] spacing) {
		super(width, height);
		// TODO Auto-generated constructor stub
		this.setSpacing(spacing[0], spacing[1]);
		this.setOrigin(-(width * spacing[0] / 2), -(height * spacing[1] / 2));
		this.height = height;
		this.width = width;

		configuration(height, width);

	}

	/**
	 * Sets all pixel values in image.
	 * @param height: Height of an image
	 * @param width:  Width of an image
	 */
	public void configuration(int height, int width) {
		Random x = new Random();
		float rand_value;
		for (int i = 0; i < height; i++)

		{
			for (int j = 0; j < width; j++) {
				rand_value = (float) x.nextGaussian() * 5;
				setAtIndex(i, j, rand_value);
			}
		}

	}
    /**
     * Generates Signal present image
     * @param filter_image: Filtered(cone-filter) noise images
     * @param signal:       Gaussian signal           
     * @return Signal-present images
     */
	public Grid2D sp(Grid2D[] filter_image, Grid2D signal) {
		Grid2D image = new Grid2D(filter_image[0].getWidth(), filter_image[0].getHeight());
		NumericPointwiseOperators.addBy(image, signal);
		NumericPointwiseOperators.addBy(image, filter_image[1]);
		NumericPointwiseOperators.addBy(image, filter_image[0]);
		return image;// returns image with background noise and signal
	}

	/**
	 * Generates noise-absent image
	 * @param filter_image:   Filtered(cone-filter) noise images
	 * @return Signal-absent image
	 */
	public Grid2D sa(Grid2D[] filter_image) {
		Grid2D image1 = new Grid2D(filter_image[0].getWidth(), filter_image[1].getHeight());
		NumericPointwiseOperators.addBy(image1, filter_image[1]);
		NumericPointwiseOperators.addBy(image1, filter_image[0]);
		return image1;// returns image with background noise and signal
	}
	/**
	 * Cone-filter 
	 * @param image: Image to be filtered(noise image) 
	 * @param bg:    Amplitude of noise component
	 * @return Filtered image
	 */
	public Grid2D conefilter(Grid2D image, float bg){
		
		Grid2DComplex fouriertransform = new Grid2DComplex(image, true);
		fouriertransform.transformForward();
		int beta = 2;// constant of cone.filter
		// Transfer values from 3rd quadrant value to 1st, 4th to 2nd and vice-versa
		fouriertransform.fftshift();
		
		// implementation of cone filter Grid
		
		float[][] con_x = new float[fouriertransform.getHeight()][fouriertransform.getWidth()];
		float i1 = -fouriertransform.getWidth() / 2, a;
		for (int i = 0; i < fouriertransform.getWidth(); i++) {
			a = i1 / fouriertransform.getWidth();
			for (int j = 0; j < fouriertransform.getWidth(); j++) {
				con_x[i][j] = a;
			}
			i1 += 1;
		}
		
		float[][] con_y = new float[fouriertransform.getHeight()][fouriertransform.getWidth()];
		i1 = -fouriertransform.getHeight() / 2;
		for (int i = 0; i < fouriertransform.getHeight(); i++) {

			a = i1 / fouriertransform.getWidth();
			for (int j = 0; j < fouriertransform.getHeight(); j++) {
				con_y[j][i] = a;
			}
			i1 += 1;
		}

		
		float[][] cone_filter = new float[fouriertransform.getHeight()][fouriertransform.getWidth()];
		for (int i = 0; i < fouriertransform.getWidth(); i++) {
			for (int j = 0; j < fouriertransform.getHeight(); j++) {
				cone_filter[i][j] = (float) Math.pow((Math.pow(con_x[i][j], 2) + Math.pow(con_y[i][j], 2)), -beta / 4f);
				if (Math.abs(con_x[i][j]) < (3.0 / fouriertransform.getWidth())
						&& Math.abs(con_y[i][j]) < (3.0 / fouriertransform.getHeight())) {
					cone_filter[i][j] = 0;
				}
			}
		}
		
		/*// generating con_filter grid in order to visualize it.
		Grid2DComplex filter_grid = new Grid2DComplex(image.getWidth(), image.getHeight(), true);
		for (int i = 0; i < fouriertransform.getWidth(); i++) {

			for (int j = 0; j < fouriertransform.getHeight(); j++) {
				filter_grid.setAtIndex(i, j, (float) cone_filter[i][j]);
			}
		}*/
		
		// applying grid on image
		for (int i = 0; i < fouriertransform.getHeight(); i++) {
			for (int j = 0; j < fouriertransform.getWidth(); j++) {

				fouriertransform.setRealAtIndex(i, j, cone_filter[i][j] * fouriertransform.getRealAtIndex(i, j));
				fouriertransform.setImagAtIndex(i, j, fouriertransform.getImagAtIndex(i, j) * cone_filter[i][j]);
			}
		}
		fouriertransform.ifftshift();
		fouriertransform.transformInverse();
		for (int i = 0; i < image.getHeight(); i++) {
			for (int j = 0; j < image.getWidth(); j++) {
				image.setAtIndex(i, j, fouriertransform.getRealAtIndex(i, j));
			}
		}
		float max, min;
		max = NumericPointwiseOperators.max(image);
		min = NumericPointwiseOperators.min(image);
		float div = max - min;
		NumericPointwiseOperators.subtractBy(image, min);
		NumericPointwiseOperators.divideBy(image, div);
		NumericPointwiseOperators.subtractBy(image, 1 / 2);
		NumericPointwiseOperators.multiplyBy(image, bg);
		return image;
	}

	/**
	 * 
	 * @return Signal
	 */
	public Grid2D create_signal() {
		float k;
		float i11 = (float) (1 / (2 * Math.pow(sigscale, 2)));
		float[] x11 = new float[this.getWidth()], y11 = new float[this.getWidth()];
		float a = -this.getWidth() / 2;
		float b = -this.getHeight() / 2;
		for (int i = 0; i < this.getWidth(); i++) {
			x11[i] = a;
			++a;
		}
		for (int i = 0; i < this.getWidth(); i++) {
			y11[i] = b;
			++b;
		}
		Grid2D grid = new Grid2D(x11.length, y11.length);
		for (int i = 0; i < x11.length; i++) {
			for (int j = 0; j < y11.length; j++) {
				k = (float) (A * Math.exp(-((x11[i] * x11[i] * i11) + (y11[j] * y11[j] * i11))));
				grid.setAtIndex(i, j, k);
			}
		}
		grid.setSpacing(spacing[0], spacing[1]);
		grid.setOrigin(-(width * spacing[0] / 2), -(height * spacing[1] / 2));
		return grid;

	}

	/**
	 * 
	 * @param width:		     Width of an image
	 * @param height:		     Height of an image
	 * @param templet_value:     Siganl intensity value
	 * @param a:                 Decides the shape of reference signal("Disk"or "Gaussian")   
	 * @return Reference signal
	 */
	public Grid2D tamplets(int width, int height, float templet_value, String a) {
		Grid2D tamplet = new Grid2D(width, height);
		tamplet.setSpacing(1.0f, 1.0f);
		tamplet.setOrigin(-tamplet.getWidth() / 2 + 0.5, +tamplet.getHeight() / 2 + 0.5);
		sigscale = 2.5f;
		float i1 = (float) (1 / (2 * Math.pow(sigscale, 2)));
		if (a == "Disk") {
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					if ((Math.pow(i - (width / 2), 2) + Math.pow(j - (height / 2), 2)) <= Math.pow(sigscale, 2)) {
						tamplet.setAtIndex(i, j, templet_value);
					}
					
					// templet_value++;
				}
			}
			return tamplet;
		} else {
			for (int i = 1; i <= width; i++) {
				for (int j = 1; j <= height; j++) {
					// templet_value = (float)(Math.exp(-A/i1));
					//if ((Math.pow(i - (width / 2), 2) + Math.pow(j - (height / 2), 2)) <= Math.pow(sigscale, 2)){
					templet_value = (float) (1000 * Math
							.exp(-(i*i + j*j)*i1));
					tamplet.setAtIndex(i, j, templet_value);
					//}
				}
			}
			return tamplet;
		}
	}
}


/*
 * Copyright (C) 2010-2016 -Priyal Patel,Frank Schebesch,Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License
 * (GPL).
 */
